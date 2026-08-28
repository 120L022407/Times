import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalBinMapper(nn.Module):
    """Training-only quantile boundaries and hierarchical target mapping."""

    def __init__(self, channels, coarse_classes=2, fine_classes=4, eps=1e-6):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}.')
        if coarse_classes < 2:
            raise ValueError(f'hcan_kc must be at least 2, got {coarse_classes}.')
        if fine_classes <= coarse_classes or fine_classes % coarse_classes != 0:
            raise ValueError(
                'hcan_kf must be greater than and divisible by hcan_kc, got '
                f'hcan_kc={coarse_classes}, hcan_kf={fine_classes}.'
            )
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}.')

        self.channels = int(channels)
        self.coarse_classes = int(coarse_classes)
        self.fine_classes = int(fine_classes)
        self.eps = float(eps)
        self.register_buffer(
            'coarse_boundaries', torch.full((channels, coarse_classes + 1), torch.nan)
        )
        self.register_buffer(
            'fine_boundaries', torch.full((channels, fine_classes + 1), torch.nan)
        )
        self.register_buffer('boundaries_fitted', torch.tensor(False, dtype=torch.bool))

    def fit(self, targets):
        if bool(self.boundaries_fitted.item()):
            raise RuntimeError('HCAN boundaries are already fitted and cannot be changed.')
        if targets.ndim < 2 or targets.shape[-1] != self.channels:
            raise ValueError(
                f'Boundary targets must end in {self.channels} channels, got {tuple(targets.shape)}.'
            )
        values = targets.detach().reshape(-1, self.channels).float()
        if values.shape[0] == 0:
            raise ValueError('Cannot fit HCAN boundaries from an empty training target set.')
        if not torch.isfinite(values).all():
            raise ValueError('HCAN boundary targets must contain only finite values.')

        coarse = self._equal_frequency_boundaries(values, self.coarse_classes)
        fine = self._equal_frequency_boundaries(values, self.fine_classes)
        self.coarse_boundaries.copy_(coarse.to(self.coarse_boundaries.device))
        self.fine_boundaries.copy_(fine.to(self.fine_boundaries.device))
        self.boundaries_fitted.fill_(True)

    @staticmethod
    def _equal_frequency_boundaries(values, classes):
        sorted_values = values.sort(dim=0).values
        positions = torch.arange(classes + 1, device=values.device, dtype=torch.float64)
        indices = torch.floor(positions * (values.shape[0] - 1) / classes).long()
        return sorted_values[indices].transpose(0, 1).contiguous()

    def map_targets(self, targets, level):
        if not bool(self.boundaries_fitted.item()):
            raise RuntimeError('HCAN boundaries must be fitted from training targets first.')
        if targets.ndim != 3 or targets.shape[-1] != self.channels:
            raise ValueError(
                f'HCAN targets must have shape [B, pred_len, {self.channels}], '
                f'got {tuple(targets.shape)}.'
            )
        if level == 'coarse':
            boundaries = self.coarse_boundaries
        elif level == 'fine':
            boundaries = self.fine_boundaries
        else:
            raise ValueError(f"level must be 'coarse' or 'fine', got {level!r}.")

        boundaries = boundaries.to(device=targets.device, dtype=targets.dtype)
        internal = boundaries[:, 1:-1].view(1, 1, self.channels, -1)
        labels = (targets.unsqueeze(-1) >= internal).sum(dim=-1).long()
        expanded = boundaries.view(1, 1, self.channels, -1).expand(
            targets.shape[0], targets.shape[1], -1, -1
        )
        lower = expanded.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        upper = expanded.gather(-1, (labels + 1).unsqueeze(-1)).squeeze(-1)
        width = upper - lower
        relative = torch.where(
            width.abs() > self.eps,
            (targets - lower) / width.clamp_min(self.eps),
            torch.zeros_like(targets),
        ).clamp(0.0, 1.0)
        return labels, relative

    def fine_to_coarse(self, fine_values):
        if fine_values.shape[-1] != self.fine_classes:
            raise ValueError(
                f'Expected {self.fine_classes} fine classes, got {fine_values.shape[-1]}.'
            )
        classes_per_group = self.fine_classes // self.coarse_classes
        return fine_values.reshape(
            *fine_values.shape[:-1], self.coarse_classes, classes_per_group
        ).mean(dim=-1)


class UncertaintyAwareClassifier(nn.Module):
    """Softplus-evidence UAC with uncertainty weighting and Dirichlet KL."""

    def __init__(self, classes):
        super().__init__()
        if classes < 2:
            raise ValueError(f'UAC requires at least two classes, got {classes}.')
        self.classes = int(classes)

    @staticmethod
    def _dirichlet_kl(alpha):
        prior = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        sum_prior = prior.sum(dim=-1, keepdim=True)
        log_normalizer = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            + torch.lgamma(prior).sum(dim=-1, keepdim=True)
            - torch.lgamma(sum_prior)
        )
        digamma_term = ((alpha - prior) * (
            torch.digamma(alpha) - torch.digamma(sum_alpha)
        )).sum(dim=-1, keepdim=True)
        return (log_normalizer + digamma_term).squeeze(-1)

    def forward(self, logits, labels, annealing_coefficient=1.0):
        if logits.shape[:-1] != labels.shape or logits.shape[-1] != self.classes:
            raise ValueError(
                f'UAC logits/labels mismatch: logits={tuple(logits.shape)}, '
                f'labels={tuple(labels.shape)}, classes={self.classes}.'
            )
        logits = logits.float()
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        one_hot = F.one_hot(labels, num_classes=self.classes).to(alpha.dtype)
        belief = evidence / strength
        uncertainty_weight = (1.0 - belief) * one_hot
        data_fit = (
            uncertainty_weight * (torch.digamma(strength) - torch.digamma(alpha))
        ).sum(dim=-1)

        adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
        annealing = float(max(0.0, min(1.0, annealing_coefficient)))
        loss = data_fit + annealing * self._dirichlet_kl(adjusted_alpha)
        probabilities = alpha / strength
        uncertainty = self.classes / strength.squeeze(-1)
        return loss.mean(), probabilities, uncertainty


def hierarchical_consistency_loss(coarse_logits, mapped_fine_logits, eps=1e-8):
    if coarse_logits.shape != mapped_fine_logits.shape:
        raise ValueError(
            'HCL inputs must have identical shapes, got '
            f'{tuple(coarse_logits.shape)} and {tuple(mapped_fine_logits.shape)}.'
        )
    coarse_probability = F.softmax(coarse_logits.float(), dim=-1).clamp_min(eps)
    fine_probability = F.softmax(mapped_fine_logits.float(), dim=-1).clamp_min(eps)
    coarse_probability = coarse_probability / coarse_probability.sum(dim=-1, keepdim=True)
    fine_probability = fine_probability / fine_probability.sum(dim=-1, keepdim=True)
    coarse_to_fine = (coarse_probability * (
        coarse_probability.log() - fine_probability.log()
    )).sum(dim=-1)
    fine_to_coarse = (fine_probability * (
        fine_probability.log() - coarse_probability.log()
    )).sum(dim=-1)
    return 0.5 * (coarse_to_fine + fine_to_coarse).mean()


class HierarchyAwareAttention(nn.Module):
    def __init__(self, pred_len, hidden_dim):
        super().__init__()
        self.coarse_projection = nn.Linear(pred_len, hidden_dim)
        self.fine_projection = nn.Linear(pred_len, hidden_dim)
        self.temporal_projection = nn.Linear(pred_len, hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, pred_len)
        self.prediction_head = nn.Linear(pred_len, pred_len)

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError(f'HAA expects [B, C, pred_len], got {tuple(features.shape)}.')
        coarse_features = self.coarse_projection(features)
        fine_features = self.fine_projection(features)
        temporal_features = self.temporal_projection(features)
        attention = torch.matmul(coarse_features, fine_features.transpose(1, 2))
        attention = F.softmax(attention, dim=-1)
        attended = torch.matmul(attention, temporal_features)
        fused = self.attention_output(attended) + features
        prediction = self.prediction_head(fused).transpose(1, 2).contiguous()
        return prediction, coarse_features, fine_features, attention


class HCAN(nn.Module):
    def __init__(self, channels, pred_len, hidden_dim=512, coarse_classes=2,
                 fine_classes=4, regression_weight=1.0):
        super().__init__()
        if pred_len <= 0 or hidden_dim <= 0:
            raise ValueError('pred_len and hcan_hidden_dim must both be positive.')
        if regression_weight < 0:
            raise ValueError(f'hcan_alpha must be non-negative, got {regression_weight}.')
        self.channels = int(channels)
        self.pred_len = int(pred_len)
        self.regression_weight = float(regression_weight)
        self.mapper = HierarchicalBinMapper(
            channels, coarse_classes=coarse_classes, fine_classes=fine_classes
        )
        self.haa = HierarchyAwareAttention(pred_len, hidden_dim)
        self.coarse_relative_head = nn.Linear(hidden_dim, pred_len * coarse_classes)
        self.coarse_evidence_head = nn.Linear(hidden_dim, pred_len * coarse_classes)
        self.fine_relative_head = nn.Linear(hidden_dim, pred_len * fine_classes)
        self.fine_evidence_head = nn.Linear(hidden_dim, pred_len * fine_classes)
        self.coarse_uac = UncertaintyAwareClassifier(coarse_classes)
        self.fine_uac = UncertaintyAwareClassifier(fine_classes)
        self.coarse_classes = int(coarse_classes)
        self.fine_classes = int(fine_classes)
        self._last_outputs = None

    def forward(self, features):
        prediction, coarse_features, fine_features, attention = self.haa(features)
        batch_size = features.shape[0]
        coarse_relative = self.coarse_relative_head(coarse_features).reshape(
            batch_size, self.channels, self.pred_len, self.coarse_classes
        ).permute(0, 2, 1, 3).contiguous()
        coarse_logits = self.coarse_evidence_head(coarse_features).reshape(
            batch_size, self.channels, self.pred_len, self.coarse_classes
        ).permute(0, 2, 1, 3).contiguous()
        fine_relative = self.fine_relative_head(fine_features).reshape(
            batch_size, self.channels, self.pred_len, self.fine_classes
        ).permute(0, 2, 1, 3).contiguous()
        fine_logits = self.fine_evidence_head(fine_features).reshape(
            batch_size, self.channels, self.pred_len, self.fine_classes
        ).permute(0, 2, 1, 3).contiguous()
        self._last_outputs = {
            'coarse_relative': coarse_relative,
            'coarse_logits': coarse_logits,
            'fine_relative': fine_relative,
            'fine_logits': fine_logits,
            'mapped_fine_logits': self.mapper.fine_to_coarse(fine_logits),
            'attention': attention,
        }
        return prediction

    def training_losses(self, targets, annealing_coefficient=1.0):
        if self._last_outputs is None:
            raise RuntimeError('HCAN training_losses() requires a preceding forward pass.')
        coarse_labels, coarse_relative_target = self.mapper.map_targets(targets, 'coarse')
        fine_labels, fine_relative_target = self.mapper.map_targets(targets, 'fine')
        outputs = self._last_outputs
        coarse_uac, _, coarse_uncertainty = self.coarse_uac(
            outputs['coarse_logits'], coarse_labels, annealing_coefficient
        )
        fine_uac, _, fine_uncertainty = self.fine_uac(
            outputs['fine_logits'], fine_labels, annealing_coefficient
        )
        coarse_relative = outputs['coarse_relative'].gather(
            -1, coarse_labels.unsqueeze(-1)
        ).squeeze(-1)
        fine_relative = outputs['fine_relative'].gather(
            -1, fine_labels.unsqueeze(-1)
        ).squeeze(-1)
        coarse_regression = F.mse_loss(coarse_relative.float(), coarse_relative_target.float())
        fine_regression = F.mse_loss(fine_relative.float(), fine_relative_target.float())
        hierarchy = (
            fine_uac + self.regression_weight * fine_regression
            + coarse_uac + self.regression_weight * coarse_regression
        )
        consistency = hierarchical_consistency_loss(
            outputs['coarse_logits'], outputs['mapped_fine_logits']
        )
        return {
            'hierarchy': hierarchy,
            'consistency': consistency,
            'coarse_uac': coarse_uac,
            'fine_uac': fine_uac,
            'coarse_regression': coarse_regression,
            'fine_regression': fine_regression,
            'coarse_uncertainty': coarse_uncertainty.mean(),
            'fine_uncertainty': fine_uncertainty.mean(),
        }
