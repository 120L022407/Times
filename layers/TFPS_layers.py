import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class FourierEncoderLayer(nn.Module):
    """FNet-style encoder layer used by the TFPS frequency branch."""

    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        activation_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.fourier_dropout = nn.Dropout(dropout)
        self.fourier_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fourier = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = self.fourier_norm(x + self.fourier_dropout(fourier))
        return self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))


class TimeEncoderLayer(nn.Module):
    """PatchTST-style attention block with batch normalization."""

    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()
        activation_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + self.attention_dropout(attended))
        return self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))


class PatchDomainEncoder(nn.Module):
    """Patch projection plus either the time or frequency TFPS encoder."""

    def __init__(self, domain, patch_num, patch_len, d_model, n_heads, d_ff,
                 e_layers, dropout, activation):
        super().__init__()
        if domain not in {'time', 'frequency'}:
            raise ValueError(f"domain must be 'time' or 'frequency', got {domain!r}.")

        self.domain = domain
        self.patch_num = patch_num
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, patch_num, d_model))
        self.embedding_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        layer_type = TimeEncoderLayer if domain == 'time' else FourierEncoderLayer
        self.encoder = nn.ModuleList(
            [
                layer_type(d_model, n_heads, d_ff, dropout, activation)
                if domain == 'time'
                else layer_type(d_model, d_ff, dropout, activation)
                for _ in range(e_layers)
            ]
        )

    def forward(self, patches):
        # patches: [batch, channels, patch_num, patch_len]
        batch_size, channels, patch_num, _ = patches.shape
        if patch_num != self.patch_num:
            raise ValueError(
                f'Expected {self.patch_num} input patches, got {patch_num}. '
                'Check seq_len, patch_len, and stride.'
            )

        encoded = self.patch_projection(patches)
        encoded = encoded.reshape(batch_size * channels, patch_num, -1)
        encoded = self.embedding_dropout(encoded + self.position_embedding)

        for layer in self.encoder:
            encoded = layer(encoded)

        return encoded.reshape(batch_size, channels, patch_num, -1)


class SubspacePatternIdentifier(nn.Module):
    """Subspace affinity assignment and the PI loss from the TFPS paper."""

    def __init__(self, feature_dim, num_patterns, subspace_dim, eta, beta,
                 regularization_weight=1e-3):
        super().__init__()
        if num_patterns <= 0:
            raise ValueError(f'num_patterns must be > 0, got {num_patterns}.')
        if subspace_dim <= 0:
            raise ValueError(f'subspace_dim must be > 0, got {subspace_dim}.')
        if eta <= 0:
            raise ValueError(f'eta must be > 0, got {eta}.')
        if beta < 0:
            raise ValueError(f'beta must be >= 0, got {beta}.')

        self.feature_dim = feature_dim
        self.num_patterns = num_patterns
        self.subspace_dim = subspace_dim
        self.eta = eta
        self.beta = beta
        self.regularization_weight = regularization_weight
        self.bases = nn.Parameter(
            torch.empty(feature_dim, num_patterns, subspace_dim)
        )
        nn.init.normal_(self.bases, mean=0.0, std=feature_dim ** -0.5)
        self._last_affinity = None
        self.last_affinity_shape = None

    def forward(self, features):
        # features: [batch, patch_num, feature_dim]
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f'Pattern feature size must be {self.feature_dim}, '
                f'got {features.shape[-1]}.'
            )
        projections = torch.einsum('bnq,qkd->bnkd', features, self.bases)
        energy = projections.square().sum(dim=-1)
        affinity = energy + self.eta * self.subspace_dim
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._last_affinity = affinity
        self.last_affinity_shape = tuple(affinity.shape)
        return affinity

    def auxiliary_loss(self):
        if self._last_affinity is None:
            raise RuntimeError('auxiliary_loss() requires a forward pass first.')

        affinity = self._last_affinity.reshape(-1, self.num_patterns).clamp_min(1e-12)
        refined = affinity.square() / affinity.sum(dim=0, keepdim=True).clamp_min(1e-12)
        refined = refined / refined.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cluster_loss = F.kl_div(
            affinity.log(), refined.detach(), reduction='batchmean'
        )

        flat_bases = self.bases.reshape(
            self.feature_dim, self.num_patterns * self.subspace_dim
        )
        gram = flat_bases.transpose(0, 1) @ flat_bases
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        within_pattern_mask = torch.zeros_like(gram)
        for index in range(self.num_patterns):
            start = index * self.subspace_dim
            end = start + self.subspace_dim
            within_pattern_mask[start:end, start:end] = 1

        norm_constraint = 0.5 * ((gram * identity) - identity).square().sum()
        separation_constraint = 0.5 * (gram * (1 - within_pattern_mask)).square().sum()
        regularization = self.regularization_weight * (
            norm_constraint + separation_constraint
        )
        return regularization + self.beta * cluster_loss


class PatternExpert(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TopKPatternExperts(nn.Module):
    """Sparse patch-wise experts routed by PI affinity scores."""

    def __init__(self, feature_dim, num_experts, top_k, hidden_dim, dropout):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f'top_k must be > 0, got {top_k}.')
        if top_k > num_experts:
            raise ValueError(
                f'top_k ({top_k}) cannot exceed num_experts ({num_experts}).'
            )
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be > 0, got {hidden_dim}.')

        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                PatternExpert(feature_dim, hidden_dim, dropout)
                for _ in range(num_experts)
            ]
        )
        self.last_topk_shape = None

    def forward(self, features, affinity):
        # features: [batch, patch_num, channels, d_model]
        batch_size, patch_num, channels, d_model = features.shape
        flat_features = features.reshape(batch_size, patch_num, -1)
        if affinity.shape != (batch_size, patch_num, self.num_experts):
            raise ValueError(
                'Affinity shape must match [batch, patch_num, num_experts], '
                f'got {tuple(affinity.shape)}.'
            )

        routing_logits = affinity
        if self.training:
            routing_logits = routing_logits + (
                torch.randn_like(routing_logits) * F.softplus(routing_logits)
            )
        topk_logits, topk_indices = routing_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        self.last_topk_shape = tuple(topk_indices.shape)

        flat_input = flat_features.reshape(-1, flat_features.shape[-1])
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_weights = topk_weights.reshape(-1, self.top_k)
        flat_output = torch.zeros_like(flat_input)

        for expert_index, expert in enumerate(self.experts):
            selected = flat_indices == expert_index
            token_mask = selected.any(dim=-1)
            if not token_mask.any():
                continue
            expert_weight = (flat_weights[token_mask] * selected[token_mask]).sum(
                dim=-1, keepdim=True
            )
            flat_output[token_mask] += expert(flat_input[token_mask]) * expert_weight

        return flat_output.reshape(batch_size, patch_num, channels, d_model)


class TFPSDomainBranch(nn.Module):
    def __init__(self, domain, channels, patch_num_in, patch_num_out, patch_len,
                 d_model, n_heads, d_ff, e_layers, dropout, activation,
                 num_experts, top_k, subspace_dim, expert_hidden_dim, eta, beta):
        super().__init__()
        feature_dim = channels * d_model
        self.domain = domain
        self.encoder = PatchDomainEncoder(
            domain=domain,
            patch_num=patch_num_in,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
        )
        self.patch_projection = nn.Linear(patch_num_in, patch_num_out)
        self.pattern_identifier = SubspacePatternIdentifier(
            feature_dim=feature_dim,
            num_patterns=num_experts,
            subspace_dim=subspace_dim,
            eta=eta,
            beta=beta,
        )
        self.pattern_experts = TopKPatternExperts(
            feature_dim=feature_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=expert_hidden_dim,
            dropout=0.1,
        )

    def forward(self, patches):
        encoded = self.encoder(patches)
        encoded = self.patch_projection(encoded.transpose(-1, -2)).transpose(-1, -2)
        encoded = encoded.permute(0, 2, 1, 3)
        affinity = self.pattern_identifier(encoded.flatten(start_dim=2))
        output = self.pattern_experts(encoded, affinity)
        output = output.permute(0, 2, 3, 1)
        if self.domain == 'frequency':
            output = torch.fft.ifft(
                torch.fft.ifft(output, dim=-2), dim=-1
            ).real
        return output

    def auxiliary_loss(self):
        return self.pattern_identifier.auxiliary_loss()
