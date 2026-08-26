import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.TFPS_layers import TFPSDomainBranch


class ChannelwisePredictionHead(nn.Module):
    def __init__(self, enc_in, c_out, input_dim, pred_len):
        super().__init__()
        self.source_channels = [enc_in - 1] if c_out == 1 else list(range(enc_in))
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, pred_len) for _ in self.source_channels]
        )

    def forward(self, features):
        forecasts = [
            head(features[:, channel].flatten(start_dim=1))
            for channel, head in zip(self.source_channels, self.heads)
        ]
        return torch.stack(forecasts, dim=-1)


class Model(nn.Module):
    """Time-Frequency Pattern-Specific experts for long-term forecasting."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != 'long_term_forecast':
            raise ValueError(
                "TFPS currently supports task_name='long_term_forecast' only."
            )

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self._validate_core_config(configs)
        self.patch_num_in = self._patch_count(self.seq_len)
        self.patch_num_out = self._patch_count(self.pred_len)
        feature_dim = self.enc_in * configs.d_model

        time_subspace_dim = self._subspace_dim(
            feature_dim, configs.tfps_t_num_experts, configs.tfps_subspace_dim
        )
        frequency_subspace_dim = self._subspace_dim(
            feature_dim, configs.tfps_f_num_experts, configs.tfps_subspace_dim
        )
        expert_hidden_dim = configs.tfps_expert_hidden
        if expert_hidden_dim == 0:
            expert_hidden_dim = 4 * feature_dim

        common = dict(
            channels=self.enc_in,
            patch_num_in=self.patch_num_in,
            patch_num_out=self.patch_num_out,
            patch_len=self.patch_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            activation=configs.activation,
            expert_hidden_dim=expert_hidden_dim,
            eta=configs.tfps_eta,
            beta=configs.tfps_beta,
        )
        self.time_branch = TFPSDomainBranch(
            domain='time',
            num_experts=configs.tfps_t_num_experts,
            top_k=configs.tfps_t_top_k,
            subspace_dim=time_subspace_dim,
            **common,
        )
        self.frequency_branch = TFPSDomainBranch(
            domain='frequency',
            num_experts=configs.tfps_f_num_experts,
            top_k=configs.tfps_f_top_k,
            subspace_dim=frequency_subspace_dim,
            **common,
        )
        self.prediction_head = ChannelwisePredictionHead(
            enc_in=self.enc_in,
            c_out=self.c_out,
            input_dim=2 * configs.d_model * self.patch_num_out,
            pred_len=self.pred_len,
        )
        self.time_loss_weight = configs.tfps_time_loss_weight
        self.frequency_loss_weight = configs.tfps_frequency_loss_weight

    def _validate_core_config(self, configs):
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError('seq_len and pred_len must both be > 0.')
        if self.patch_len <= 0:
            raise ValueError(f'patch_len must be > 0, got {self.patch_len}.')
        if self.stride <= 0:
            raise ValueError(f'stride must be > 0, got {self.stride}.')
        if self.patch_len > min(self.seq_len, self.pred_len):
            raise ValueError(
                'patch_len must not exceed seq_len or pred_len, got '
                f'patch_len={self.patch_len}, seq_len={self.seq_len}, '
                f'pred_len={self.pred_len}.'
            )
        if configs.d_model <= 0 or configs.d_ff <= 0:
            raise ValueError('d_model and d_ff must both be > 0.')
        if configs.n_heads <= 0 or configs.d_model % configs.n_heads != 0:
            raise ValueError(
                f'd_model ({configs.d_model}) must be divisible by '
                f'n_heads ({configs.n_heads}).'
            )
        if configs.e_layers <= 0:
            raise ValueError(f'e_layers must be > 0, got {configs.e_layers}.')
        if configs.activation not in {'relu', 'gelu'}:
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {configs.activation!r}."
            )
        if self.c_out not in {1, self.enc_in}:
            raise ValueError(
                'TFPS supports c_out=enc_in for multivariate forecasting or '
                f'c_out=1 for MS forecasting, got enc_in={self.enc_in}, '
                f'c_out={self.c_out}.'
            )
        if configs.tfps_subspace_dim < 0:
            raise ValueError('tfps_subspace_dim must be >= 0.')
        if configs.tfps_expert_hidden < 0:
            raise ValueError('tfps_expert_hidden must be >= 0.')
        if configs.tfps_time_loss_weight < 0 or configs.tfps_frequency_loss_weight < 0:
            raise ValueError('TFPS auxiliary loss weights must be >= 0.')

        for domain, experts, top_k in (
            ('time', configs.tfps_t_num_experts, configs.tfps_t_top_k),
            ('frequency', configs.tfps_f_num_experts, configs.tfps_f_top_k),
        ):
            if experts <= 0:
                raise ValueError(f'{domain} num_experts must be > 0, got {experts}.')
            if top_k <= 0 or top_k > experts:
                raise ValueError(
                    f'{domain} top_k must be in [1, {experts}], got {top_k}.'
                )

    def _patch_count(self, length):
        return (length + self.stride - self.patch_len) // self.stride + 1

    @staticmethod
    def _subspace_dim(feature_dim, num_experts, configured_dim):
        if configured_dim > 0:
            return configured_dim
        if feature_dim % num_experts != 0:
            raise ValueError(
                'enc_in * d_model must be divisible by num_experts when '
                'tfps_subspace_dim=0; set --tfps_subspace_dim explicitly '
                f'for feature_dim={feature_dim}, num_experts={num_experts}.'
            )
        return feature_dim // num_experts

    def forecast(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f'x_enc must have shape [B, {self.seq_len}, {self.enc_in}], '
                f'got {tuple(x_enc.shape)}.'
            )

        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev

        padded = F.pad(normalized.permute(0, 2, 1), (0, self.stride), mode='replicate')
        patches = padded.unfold(-1, self.patch_len, self.stride)
        time_output = self.time_branch(patches)
        frequency_output = self.frequency_branch(patches)
        fused = torch.cat([time_output, frequency_output], dim=-2)
        channel_forecast = self.prediction_head(fused)

        if self.c_out == 1:
            means = means[..., -1:]
            stdev = stdev[..., -1:]
        return channel_forecast * stdev + means

    def auxiliary_loss(self):
        return (
            self.time_loss_weight * self.time_branch.auxiliary_loss()
            + self.frequency_loss_weight * self.frequency_branch.auxiliary_loss()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
