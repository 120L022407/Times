# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import math
import random
import weakref

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F


def _validate_forecast_pair(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f'prediction and target must have the same shape, got '
            f'{tuple(prediction.shape)} and {tuple(target.shape)}.'
        )
    if prediction.ndim != 3:
        raise ValueError(
            f'Forecasting losses expect [B, pred_len, C], got {tuple(prediction.shape)}.'
        )
    if prediction.shape[1] == 0:
        raise ValueError('pred_len must be greater than zero.')


def _forecast_fft(prediction, target):
    _validate_forecast_pair(prediction, target)

    # CUDA FFT only supports a subset of sequence lengths for low-precision inputs.
    if prediction.dtype in (t.float16, t.bfloat16):
        prediction = prediction.float()
    if target.dtype in (t.float16, t.bfloat16):
        target = target.float()

    prediction = prediction.transpose(1, 2)
    target = target.transpose(1, 2)
    prediction_fft = t.fft.fft(prediction, dim=-1, norm='ortho')
    target_fft = t.fft.fft(target, dim=-1, norm='ortho')
    return prediction_fft, target_fft


def _fourier_amplitude_loss(prediction_fft, target_fft):
    per_series_loss = (prediction_fft.abs() - target_fft.abs()).square().mean(dim=-1)
    return per_series_loss.mean()


def _fourier_correlation_loss(prediction_fft, target_fft, eps):
    numerator = (prediction_fft.conj() * target_fft).sum(dim=-1).real
    prediction_energy = prediction_fft.abs().square().sum(dim=-1)
    target_energy = target_fft.abs().square().sum(dim=-1)
    denominator = (prediction_energy * target_energy).clamp_min(eps * eps).sqrt()

    correlation = numerator / denominator
    correlation = correlation.clamp(min=-1.0, max=1.0)
    per_series_loss = 1.0 - correlation

    # Cosine distance between two identically zero series is defined as zero.
    both_zero = (prediction_energy == 0) & (target_energy == 0)
    per_series_loss = t.where(both_zero, t.zeros_like(per_series_loss), per_series_loss)
    return per_series_loss.mean()


class FourierAmplitudeLoss(nn.Module):
    """Fourier amplitude MSE for forecasts shaped [B, pred_len, C]."""

    def forward(self, prediction, target):
        prediction_fft, target_fft = _forecast_fft(prediction, target)
        return _fourier_amplitude_loss(prediction_fft, target_fft)


class FourierCorrelationLoss(nn.Module):
    """Normalized Fourier correlation distance computed per sample and channel."""

    def __init__(self, eps=1e-8):
        super().__init__()
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}.')
        self.eps = eps

    def forward(self, prediction, target):
        prediction_fft, target_fft = _forecast_fft(prediction, target)
        return _fourier_correlation_loss(prediction_fft, target_fft, self.eps)


class FourierAmplitudeCorrelationLoss(nn.Module):
    """Randomly select FCL or FAL using the paper's decreasing P(t) schedule."""

    def __init__(self, alpha=0.1, eps=1e-8):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError(f'alpha must satisfy 0 <= alpha < 1, got {alpha}.')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}.')
        self.alpha = alpha
        self.eps = eps
        self.current_step = 0
        self.total_steps = 1
        self.last_selected = None

    def set_progress(self, current_step, total_steps):
        if total_steps <= 0:
            raise ValueError(f'total_steps must be positive, got {total_steps}.')
        if not 0 <= current_step < total_steps:
            raise ValueError(
                f'current_step must satisfy 0 <= current_step < total_steps, got '
                f'{current_step} and {total_steps}.'
            )
        self.current_step = int(current_step)
        self.total_steps = int(total_steps)

    @property
    def probability_threshold(self):
        constant_steps = int(self.total_steps * self.alpha)
        decay_steps = self.total_steps - constant_steps
        if decay_steps <= 1:
            return 1.0
        if self.current_step >= decay_steps:
            return 0.0
        return 1.0 - self.current_step / (decay_steps - 1)

    def forward(self, prediction, target):
        prediction_fft, target_fft = _forecast_fft(prediction, target)
        if random.random() > self.probability_threshold:
            loss = _fourier_amplitude_loss(prediction_fft, target_fft)
            self.last_selected = 'fal'
        else:
            loss = _fourier_correlation_loss(prediction_fft, target_fft, self.eps)
            self.last_selected = 'fcl'

        # Official FACL rescales by sqrt(number of transformed spatial positions).
        return loss * math.sqrt(prediction.shape[1])


class _ForecastOutputParameterTracker:
    """Track the last trainable weight module connected to a forecast."""

    def __init__(self, model):
        self._model_ref = weakref.ref(model)
        self._candidates = {}
        self._execution_order = []
        self._handles = [model.register_forward_pre_hook(self._clear)]

        base_model = model.module if isinstance(model, nn.DataParallel) else model
        for name, module in base_model.named_modules():
            parameters = tuple(
                parameter
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad and parameter.ndim >= 2
            )
            if parameters:
                self._handles.append(module.register_forward_hook(self._capture(name)))

    def _clear(self, module, inputs):
        del module, inputs
        self._candidates.clear()
        self._execution_order.clear()

    def _capture(self, name):
        def hook(module, inputs, output):
            del inputs, output
            if not t.is_grad_enabled():
                return
            parameters = tuple(
                parameter
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad and parameter.ndim >= 2
            )
            if not parameters:
                return
            self._candidates.setdefault(name, []).extend(parameters)
            self._execution_order.append(name)

        return hook

    def parameter_groups(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError('The model bound to PS Loss no longer exists.')
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        provider = getattr(base_model, 'forecast_output_parameters', None)
        if provider is not None:
            parameters = tuple(parameter for parameter in provider() if parameter.requires_grad)
            if not parameters:
                raise ValueError('forecast_output_parameters() returned no trainable parameters.')
            yield 'forecast_output_parameters', parameters

        seen_names = set()
        for name in reversed(self._execution_order):
            if name in seen_names:
                continue
            seen_names.add(name)
            parameters = []
            seen_parameters = set()
            for parameter in self._candidates[name]:
                if id(parameter) not in seen_parameters:
                    seen_parameters.add(id(parameter))
                    parameters.append(parameter)
            yield name, tuple(parameters)


class PatchwiseStructuralLoss(nn.Module):
    """MSE plus Fourier-adaptive patch-wise structural loss with GDW."""

    def __init__(self, model, ps_lambda=3.0, delta=24, eps=1e-5):
        super().__init__()
        if model is None:
            raise ValueError('PS Loss requires a model to compute GDW with respect to output weights.')
        if ps_lambda < 0:
            raise ValueError(f'ps_lambda must be non-negative, got {ps_lambda}.')
        if delta < 1:
            raise ValueError(f'ps_delta must be at least 1, got {delta}.')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}.')
        self.ps_lambda = float(ps_lambda)
        self.delta = int(delta)
        self.eps = float(eps)
        self._tracker = _ForecastOutputParameterTracker(model)
        self.last_dominant_frequency = None
        self.last_period = None
        self.last_patch_length = None
        self.last_stride = None
        self.last_output_parameter_group = None
        self.last_weights = None
        self.last_components = None

    def adaptive_patch_parameters(self, target):
        if target.ndim != 3:
            raise ValueError(f'PS Loss expects [B, pred_len, C], got {tuple(target.shape)}.')
        sequence_length = target.shape[1]
        if sequence_length < 2:
            raise ValueError(f'PS Loss requires pred_len >= 2, got {sequence_length}.')

        fft_input = target.float() if target.dtype in (t.float16, t.bfloat16) else target
        amplitudes = t.fft.rfft(fft_input, dim=1).abs().mean(dim=(0, 2))
        amplitudes = amplitudes.clone()
        amplitudes[0] = -t.inf
        dominant_frequency = int(amplitudes.argmax().item())
        if dominant_frequency < 1:
            dominant_frequency = 1

        period = sequence_length // dominant_frequency
        patch_length = min(max(period // 2, 1), self.delta)
        stride = max(patch_length // 2, 1)
        return dominant_frequency, period, patch_length, stride

    @staticmethod
    def create_patches(values, patch_length, stride):
        values = values.transpose(1, 2)
        return values.unfold(dimension=-1, size=patch_length, step=stride)

    def fourier_adaptive_patching(self, prediction, target):
        _validate_forecast_pair(prediction, target)
        frequency, period, patch_length, stride = self.adaptive_patch_parameters(target)
        self.last_dominant_frequency = frequency
        self.last_period = period
        self.last_patch_length = patch_length
        self.last_stride = stride
        return (
            self.create_patches(prediction, patch_length, stride),
            self.create_patches(target, patch_length, stride),
        )

    def structural_components(self, prediction_patches, target_patches):
        prediction_mean = prediction_patches.mean(dim=-1, keepdim=True)
        target_mean = target_patches.mean(dim=-1, keepdim=True)
        prediction_centered = prediction_patches - prediction_mean
        target_centered = target_patches - target_mean
        prediction_var = prediction_centered.square().mean(dim=-1, keepdim=True)
        target_var = target_centered.square().mean(dim=-1, keepdim=True)
        covariance = (prediction_centered * target_centered).mean(dim=-1, keepdim=True)

        denominator = (prediction_var * target_var).clamp_min(self.eps * self.eps).sqrt()
        correlation = (covariance / denominator).clamp(min=-1.0, max=1.0)
        prediction_constant = prediction_var <= self.eps
        target_constant = target_var <= self.eps
        both_constant = prediction_constant & target_constant
        either_constant = prediction_constant | target_constant
        correlation = t.where(both_constant, t.ones_like(correlation), correlation)
        correlation = t.where(either_constant & ~both_constant, t.zeros_like(correlation), correlation)
        correlation_loss = (1.0 - correlation).mean()

        target_distribution = F.softmax(target_patches, dim=-1)
        prediction_log_distribution = F.log_softmax(prediction_patches, dim=-1)
        variance_loss = F.kl_div(
            prediction_log_distribution,
            target_distribution,
            reduction='none',
        ).sum(dim=-1).mean()
        mean_loss = (prediction_mean - target_mean).abs().mean()
        return correlation_loss, variance_loss, mean_loss

    def _mean_similarity_scale(self, prediction, target):
        prediction = prediction.transpose(1, 2)
        target = target.transpose(1, 2)
        prediction_centered = prediction - prediction.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        prediction_var = prediction_centered.square().mean(dim=-1, keepdim=True)
        target_var = target_centered.square().mean(dim=-1, keepdim=True)
        covariance = (prediction_centered * target_centered).mean(dim=-1, keepdim=True)
        denominator = (prediction_var * target_var).clamp_min(self.eps * self.eps).sqrt()
        correlation = (covariance / denominator).clamp(min=-1.0, max=1.0)

        prediction_constant = prediction_var <= self.eps
        target_constant = target_var <= self.eps
        both_constant = prediction_constant & target_constant
        either_constant = prediction_constant | target_constant
        correlation = t.where(both_constant, t.ones_like(correlation), correlation)
        correlation = t.where(either_constant & ~both_constant, t.zeros_like(correlation), correlation)

        c = 0.5 * (1.0 + correlation)
        prediction_std = prediction_var.clamp_min(0.0).sqrt()
        target_std = target_var.clamp_min(0.0).sqrt()
        v = (2.0 * prediction_std * target_std + self.eps) / (
            prediction_var + target_var + self.eps
        )
        return (c * v.clamp(min=0.0, max=1.0)).mean().detach()

    @staticmethod
    def _gradient_norm(loss, parameters):
        gradients = t.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        connected_gradients = [gradient for gradient in gradients if gradient is not None]
        if not connected_gradients:
            return None
        squared_norm = sum(gradient.detach().float().square().sum() for gradient in connected_gradients)
        return squared_norm.sqrt()

    def gradient_based_weights(self, prediction, target, components):
        if not prediction.requires_grad:
            raise RuntimeError('PS Loss requires predictions connected to trainable model parameters.')

        for group_name, parameters in self._tracker.parameter_groups():
            gradient_norms = tuple(self._gradient_norm(loss, parameters) for loss in components)
            if any(norm is None for norm in gradient_norms):
                continue
            average_norm = sum(gradient_norms) / len(gradient_norms)
            alpha = average_norm / gradient_norms[0].clamp_min(self.eps)
            beta = average_norm / gradient_norms[1].clamp_min(self.eps)
            gamma = average_norm / gradient_norms[2].clamp_min(self.eps)
            gamma = gamma * self._mean_similarity_scale(prediction, target)
            weights = alpha.detach(), beta.detach(), gamma.detach()
            self.last_output_parameter_group = group_name
            self.last_weights = weights
            return weights

        raise RuntimeError(
            'PS Loss could not find output-layer weights connected to the forecast. '
            'Expose them through model.forecast_output_parameters() for this model.'
        )

    def forward(self, prediction, target):
        _validate_forecast_pair(prediction, target)
        loss_prediction = prediction.float() if prediction.dtype in (t.float16, t.bfloat16) else prediction
        loss_target = target.float() if target.dtype in (t.float16, t.bfloat16) else target
        mse_loss = F.mse_loss(loss_prediction, loss_target)
        prediction_patches, target_patches = self.fourier_adaptive_patching(loss_prediction, loss_target)
        components = self.structural_components(prediction_patches, target_patches)
        weights = self.gradient_based_weights(loss_prediction, loss_target, components)
        ps_loss = sum(weight * component for weight, component in zip(weights, components))
        self.last_components = tuple(component.detach() for component in components)
        return mse_loss + self.ps_lambda * ps_loss


def build_forecasting_loss(
    name,
    facl_alpha=0.1,
    facl_eps=1e-8,
    model=None,
    ps_lambda=3.0,
    ps_delta=24,
):
    loss_name = name.lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    if loss_name == 'fal':
        return FourierAmplitudeLoss()
    if loss_name == 'fcl':
        return FourierCorrelationLoss(eps=facl_eps)
    if loss_name == 'facl':
        return FourierAmplitudeCorrelationLoss(alpha=facl_alpha, eps=facl_eps)
    if loss_name == 'ps':
        return PatchwiseStructuralLoss(model=model, ps_lambda=ps_lambda, delta=ps_delta)
    raise ValueError(
        f'Unsupported long-term forecasting loss {name!r}. '
        'Choose one of: mse, fal, fcl, facl, ps.'
    )


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
