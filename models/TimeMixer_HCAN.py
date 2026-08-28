import torch
import torch.nn as nn

from layers.HCAN import HCAN
from models.TimeMixer import Model as TimeMixer


class Model(nn.Module):
    """TimeMixer with HCAN attached to prediction-aligned pre-head features."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != 'long_term_forecast':
            raise ValueError("TimeMixer_HCAN supports task_name='long_term_forecast' only.")
        if not configs.channel_independence:
            raise ValueError('TimeMixer_HCAN currently requires --channel_independence 1.')
        if getattr(configs, 'use_multi_gpu', False):
            raise ValueError('TimeMixer_HCAN currently supports single-device training only.')
        self.forecast_loss_name = getattr(configs, 'loss', 'mse').lower()
        if self.forecast_loss_name not in {'mse', 'ps', 'facl'}:
            raise ValueError(
                'TimeMixer_HCAN supports --loss mse, ps, or facl. '
                'MSE is the paper-faithful default.'
            )
        if configs.hcan_beta < 0 or configs.hcan_gamma < 0:
            raise ValueError('hcan_beta and hcan_gamma must be non-negative.')
        if configs.hcan_annealing_steps <= 0:
            raise ValueError('hcan_annealing_steps must be positive.')

        self.configs = configs
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.features = configs.features
        if self.features == 'MS':
            self.output_channels = 1
        elif self.features in {'M', 'S'}:
            self.output_channels = configs.c_out
        else:
            raise ValueError(f'Unsupported features mode: {self.features!r}.')
        if self.features == 'M' and configs.c_out != configs.enc_in:
            raise ValueError('TimeMixer_HCAN requires c_out=enc_in for features=M.')
        if self.features == 'S' and configs.enc_in != 1:
            raise ValueError('TimeMixer_HCAN requires enc_in=1 for features=S.')

        self.backbone = TimeMixer(configs)
        self.hcan = HCAN(
            channels=self.output_channels,
            pred_len=configs.pred_len,
            hidden_dim=configs.hcan_hidden_dim,
            coarse_classes=configs.hcan_kc,
            fine_classes=configs.hcan_kf,
            regression_weight=configs.hcan_alpha,
        )
        self.hcan_beta = float(configs.hcan_beta)
        self.hcan_gamma = float(configs.hcan_gamma)
        self.hcan_annealing_steps = int(configs.hcan_annealing_steps)
        self._projection_inputs = []
        self._boundary_chunks = None
        self._auxiliary_targets = None
        self._training_step = 0
        self.last_loss_components = None
        self._projection_hook = self.backbone.projection_layer.register_forward_pre_hook(
            self._capture_projection_input
        )

    def _capture_projection_input(self, module, inputs):
        del module
        if inputs:
            self._projection_inputs.append(inputs[0])

    def begin_auxiliary_fit(self):
        if bool(self.hcan.mapper.boundaries_fitted.item()):
            return False
        self._boundary_chunks = []
        return True

    def update_auxiliary_fit(self, targets):
        if self._boundary_chunks is None:
            raise RuntimeError('begin_auxiliary_fit() must be called before update_auxiliary_fit().')
        if targets.ndim != 3 or targets.shape[-1] != self.output_channels:
            raise ValueError(
                f'Expected fitting targets [B, pred_len, {self.output_channels}], '
                f'got {tuple(targets.shape)}.'
            )
        self._boundary_chunks.append(targets.detach().reshape(-1, self.output_channels).cpu())

    def finalize_auxiliary_fit(self):
        if self._boundary_chunks is None:
            raise RuntimeError('begin_auxiliary_fit() must be called before finalize_auxiliary_fit().')
        if not self._boundary_chunks:
            raise ValueError('No training targets were supplied for HCAN boundary fitting.')
        self.hcan.mapper.fit(torch.cat(self._boundary_chunks, dim=0))
        self._boundary_chunks = None

    def set_training_progress(self, current_step, total_steps):
        if total_steps <= 0 or not 0 <= current_step < total_steps:
            raise ValueError('Training progress must satisfy 0 <= current_step < total_steps.')
        self._training_step = int(current_step) + 1

    def set_auxiliary_targets(self, targets):
        if targets.ndim != 3 or targets.shape[-1] != self.output_channels:
            raise ValueError(
                f'Expected auxiliary targets [B, pred_len, {self.output_channels}], '
                f'got {tuple(targets.shape)}.'
            )
        self._auxiliary_targets = targets

    def _annealing_coefficient(self):
        return min(1.0, self._training_step / self.hcan_annealing_steps)

    def auxiliary_loss(self):
        if self._auxiliary_targets is None:
            raise RuntimeError('HCAN auxiliary_loss() requires training targets for the current batch.')
        components = self.hcan.training_losses(
            self._auxiliary_targets, self._annealing_coefficient()
        )
        self.last_loss_components = {
            name: value.detach() for name, value in components.items()
        }
        return components['hierarchy'] + self.hcan_beta * components['consistency']

    def compose_training_loss(self, base_loss):
        # ``base_loss`` is the framework-selected forecast objective.  With MSE,
        # this is exactly the HCAN paper objective; PS/FACL are optional studies.
        auxiliary = self.auxiliary_loss()
        total = auxiliary + self.hcan_gamma * base_loss
        self._auxiliary_targets = None
        return total

    def _select_target_features(self, features):
        if self.features == 'MS':
            return features[:, :, -1:, :]
        return features

    def _denormalize(self, prediction):
        normalizer = self.backbone.normalize_layers[0]
        if normalizer.non_norm:
            return prediction
        if self.features == 'MS':
            channel_slice = slice(-1, None)
        else:
            channel_slice = slice(None)
        if normalizer.affine:
            weight = normalizer.affine_weight[channel_slice].view(1, 1, -1)
            bias = normalizer.affine_bias[channel_slice].view(1, 1, -1)
            prediction = (prediction - bias) / (weight + normalizer.eps * normalizer.eps)
        prediction = prediction * normalizer.stdev[..., channel_slice]
        if normalizer.subtract_last:
            return prediction + normalizer.last[..., channel_slice]
        return prediction + normalizer.mean[..., channel_slice]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        self._projection_inputs.clear()
        self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        expected_scales = self.configs.down_sampling_layers + 1
        if len(self._projection_inputs) != expected_scales:
            raise RuntimeError(
                f'Expected {expected_scales} TimeMixer projection features, '
                f'captured {len(self._projection_inputs)}.'
            )

        batch_size = x_enc.shape[0]
        scale_features = []
        for feature in self._projection_inputs:
            feature = feature.reshape(
                batch_size, self.enc_in, self.pred_len, feature.shape[-1]
            ).permute(0, 2, 1, 3).contiguous()
            scale_features.append(feature)
        fused_feature = torch.stack(scale_features, dim=-1).sum(dim=-1)
        target_feature = self._select_target_features(fused_feature)
        forecasting_feature = target_feature.mean(dim=-1).transpose(1, 2).contiguous()
        normalized_prediction = self.hcan(forecasting_feature)
        return self._denormalize(normalized_prediction)
