import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


if 'reformer_pytorch' not in sys.modules:
    reformer_module = types.ModuleType('reformer_pytorch')

    class LSHSelfAttention:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('LSHSelfAttention is not used in PS Loss tests.')

    reformer_module.LSHSelfAttention = LSHSelfAttention
    sys.modules['reformer_pytorch'] = reformer_module

if 'einops' not in sys.modules:
    einops_module = types.ModuleType('einops')
    einops_module.rearrange = lambda tensor, *args, **kwargs: tensor
    einops_module.repeat = lambda tensor, *args, **kwargs: tensor
    sys.modules['einops'] = einops_module


from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from models.TimeMixer import Model as TimeMixer
from models.TimesNet import Model as TimesNet
from models.iTransformer import Model as ITransformer
from utils.losses import PatchwiseStructuralLoss, build_forecasting_loss


class ToyForecaster(nn.Module):
    def __init__(self, length=96, channels=1):
        super().__init__()
        self.output = nn.Linear(length, length, bias=False)
        self.channels = channels

    def forward(self, values):
        return self.output(values.transpose(1, 2)).transpose(1, 2)


class ProtocolForecaster(nn.Module):
    def __init__(self, length=96):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(length))

    def forecast_output_parameters(self):
        return (self.weight,)

    def forward(self, values):
        return torch.matmul(values.transpose(1, 2), self.weight.T).transpose(1, 2)


def _periodic_target(frequency=4, channels=1):
    steps = torch.arange(96, dtype=torch.float32)
    signal = torch.sin(2 * torch.pi * frequency * steps / 96)
    return signal.view(1, 96, 1).repeat(1, 1, channels)


def _model_args(**overrides):
    values = {
        'task_name': 'long_term_forecast',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 3,
        'dec_in': 3,
        'c_out': 3,
        'd_model': 8,
        'n_heads': 2,
        'e_layers': 1,
        'd_layers': 1,
        'd_ff': 16,
        'moving_avg': 25,
        'factor': 1,
        'dropout': 0.0,
        'embed': 'timeF',
        'freq': 'h',
        'activation': 'gelu',
        'top_k': 2,
        'num_kernels': 2,
        'patch_len': 16,
        'stride': 8,
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'use_norm': 1,
        'down_sampling_layers': 0,
        'down_sampling_window': 2,
        'down_sampling_method': 'avg',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fap_uses_ground_truth_dominant_frequency_and_paper_stride():
    model = ToyForecaster()
    criterion = PatchwiseStructuralLoss(model, delta=24)

    frequency, period, patch_length, stride = criterion.adaptive_patch_parameters(
        _periodic_target(frequency=4)
    )

    assert frequency == 4
    assert period == 24
    assert patch_length == 12
    assert stride == 6


def test_fap_applies_delta_and_handles_frequency_boundary():
    model = ToyForecaster()
    criterion = PatchwiseStructuralLoss(model, delta=5)

    _, _, patch_length, stride = criterion.adaptive_patch_parameters(_periodic_target(frequency=4))
    assert patch_length == 5
    assert stride == 2

    high_frequency = torch.where(
        torch.arange(96) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).view(1, 96, 1)
    frequency, period, patch_length, stride = criterion.adaptive_patch_parameters(high_frequency)
    assert frequency == 48
    assert period == 2
    assert patch_length == 1
    assert stride == 1


def test_constant_target_uses_safe_non_dc_fallback():
    model = ToyForecaster()
    criterion = PatchwiseStructuralLoss(model, delta=24)

    frequency, period, patch_length, stride = criterion.adaptive_patch_parameters(
        torch.ones(1, 96, 1)
    )

    assert (frequency, period, patch_length, stride) == (1, 96, 24, 12)


def test_prediction_and_target_share_identical_patch_partition():
    model = ToyForecaster(channels=3)
    criterion = PatchwiseStructuralLoss(model, delta=24)
    target = _periodic_target(frequency=4, channels=3)
    prediction = target + 1.0

    prediction_patches, target_patches = criterion.fourier_adaptive_patching(prediction, target)

    assert prediction_patches.shape == target_patches.shape == (1, 3, 15, 12)
    assert torch.allclose(prediction_patches - target_patches, torch.ones_like(target_patches))


def test_structural_components_match_pearson_kl_and_patch_mean_formulas():
    model = ToyForecaster(length=4)
    criterion = PatchwiseStructuralLoss(model)
    target = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    prediction = torch.tensor([[[[1.0, 3.0, 2.0, 0.0]]]], requires_grad=True)

    corr_loss, variance_loss, mean_loss = criterion.structural_components(prediction, target)

    target_centered = target - target.mean(dim=-1, keepdim=True)
    prediction_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    covariance = (target_centered * prediction_centered).mean(dim=-1)
    denominator = target_centered.square().mean(dim=-1).sqrt() * prediction_centered.square().mean(dim=-1).sqrt()
    expected_corr = (1.0 - covariance / denominator).mean()
    expected_variance = F.kl_div(
        F.log_softmax(prediction, dim=-1),
        F.softmax(target, dim=-1),
        reduction='none',
    ).sum(dim=-1).mean()
    expected_mean = (prediction.mean(dim=-1) - target.mean(dim=-1)).abs().mean()

    assert corr_loss.item() == pytest.approx(expected_corr.item(), rel=1e-6)
    assert variance_loss.item() == pytest.approx(expected_variance.item(), rel=1e-6)
    assert mean_loss.item() == pytest.approx(expected_mean.item(), rel=1e-6)


def test_variance_kl_is_finite_for_extreme_logits():
    model = ToyForecaster(length=4)
    criterion = PatchwiseStructuralLoss(model)
    target = torch.tensor([[[[1000.0, -1000.0, 1000.0, -1000.0]]]])
    prediction = torch.tensor([[[[-1000.0, 1000.0, -1000.0, 1000.0]]]], requires_grad=True)

    components = criterion.structural_components(prediction, target)
    sum(components).backward()

    assert all(torch.isfinite(component) for component in components)
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('shape', [(1, 96, 1), (2, 96, 5)])
def test_ps_loss_is_finite_and_supports_backward(shape):
    model = ToyForecaster(channels=shape[-1])
    criterion = PatchwiseStructuralLoss(model, ps_lambda=3.0, delta=24)
    inputs = torch.randn(shape)
    target = torch.randn(shape)

    loss = criterion(model(inputs), target)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
    assert criterion.last_output_parameter_group == 'output'
    assert all(torch.isfinite(weight) and not weight.requires_grad for weight in criterion.last_weights)


def test_identical_prediction_and_target_have_near_zero_total_loss():
    model = ToyForecaster(channels=3)
    criterion = PatchwiseStructuralLoss(model, ps_lambda=3.0, delta=24)
    inputs = torch.randn(1, 96, 3)
    prediction = model(inputs)

    loss = criterion(prediction, prediction.detach().clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_constant_sequences_are_finite_and_backward_safe():
    model = ToyForecaster()
    with torch.no_grad():
        model.output.weight.zero_()
    criterion = PatchwiseStructuralLoss(model, ps_lambda=3.0, delta=24)
    prediction = model(torch.ones(1, 96, 1))
    target = torch.ones_like(prediction)

    loss = criterion(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(model.output.weight.grad).all()
    assert all(torch.isfinite(component) for component in criterion.last_components)


def test_gdw_matches_paper_average_of_gradient_norms():
    model = ToyForecaster(channels=2)
    criterion = PatchwiseStructuralLoss(model, ps_lambda=3.0, delta=24)
    inputs = torch.randn(1, 96, 2)
    target = _periodic_target(frequency=4, channels=2)
    prediction = model(inputs)
    prediction_patches, target_patches = criterion.fourier_adaptive_patching(prediction, target)
    components = criterion.structural_components(prediction_patches, target_patches)

    gradient_norms = []
    for component in components:
        gradient = torch.autograd.grad(component, model.output.weight, retain_graph=True)[0]
        gradient_norms.append(gradient.norm().detach())
    average_norm = sum(gradient_norms) / 3.0
    mean_scale = criterion._mean_similarity_scale(prediction, target)
    expected = (
        average_norm / gradient_norms[0].clamp_min(criterion.eps),
        average_norm / gradient_norms[1].clamp_min(criterion.eps),
        average_norm / gradient_norms[2].clamp_min(criterion.eps) * mean_scale,
    )

    actual = criterion.gradient_based_weights(prediction, target, components)

    for actual_weight, expected_weight in zip(actual, expected):
        assert actual_weight.item() == pytest.approx(expected_weight.item(), rel=1e-6)


def test_explicit_output_parameter_protocol_supports_future_models():
    model = ProtocolForecaster()
    criterion = PatchwiseStructuralLoss(model, ps_lambda=1.0, delta=24)
    inputs = torch.randn(1, 96, 1)
    target = _periodic_target(frequency=4)

    loss = criterion(model(inputs), target)
    loss.backward()

    assert criterion.last_output_parameter_group == 'forecast_output_parameters'
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()


@pytest.mark.parametrize(
    'model_class,expected_parameter_group',
    [
        (PatchTST, 'head.linear'),
        (DLinear, 'Linear_Trend'),
        (ITransformer, 'projection'),
        (TimesNet, 'projection'),
        (TimeMixer, 'projection_layer'),
    ],
)
def test_supported_backbones_resolve_output_weights_without_model_branches(
    model_class,
    expected_parameter_group,
):
    model_args = _model_args(down_sampling_layers=1) if model_class is TimeMixer else _model_args()
    model = model_class(model_args)
    criterion = PatchwiseStructuralLoss(model, ps_lambda=1.0, delta=24)
    batch_x = torch.randn(1, 96, 3)
    batch_x_mark = torch.randn(1, 96, 4)
    batch_y = torch.zeros(1, 144, 3)
    batch_y_mark = torch.randn(1, 144, 4)
    target = _periodic_target(frequency=4, channels=3)

    prediction = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    loss = criterion(prediction, target)
    loss.backward()

    assert prediction.shape == (1, 96, 3)
    assert criterion.last_output_parameter_group == expected_parameter_group
    assert torch.isfinite(loss)


def test_mse_builder_remains_exact_pytorch_mse():
    prediction = torch.randn(1, 96, 2)
    target = torch.randn(1, 96, 2)

    actual = build_forecasting_loss('mse')(prediction, target)
    expected = nn.MSELoss()(prediction, target)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    'kwargs,error_pattern',
    [
        ({'model': None}, 'requires a model'),
        ({'model': ToyForecaster(), 'ps_lambda': -1.0}, 'ps_lambda'),
        ({'model': ToyForecaster(), 'delta': 0}, 'ps_delta'),
    ],
)
def test_invalid_ps_parameters_fail_fast(kwargs, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        PatchwiseStructuralLoss(**kwargs)
