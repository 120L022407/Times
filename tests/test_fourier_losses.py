import math
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from utils.losses import (
    FourierAmplitudeCorrelationLoss,
    FourierAmplitudeLoss,
    FourierCorrelationLoss,
    build_forecasting_loss,
)


@pytest.mark.parametrize('shape', [(1, 96, 1), (2, 96, 7)])
@pytest.mark.parametrize(
    'criterion',
    [FourierAmplitudeLoss(), FourierCorrelationLoss(), FourierAmplitudeCorrelationLoss()],
)
def test_fourier_losses_are_finite_and_support_backward(shape, criterion):
    prediction = torch.randn(shape, requires_grad=True)
    target = torch.randn(shape)

    loss = criterion(prediction, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('shape', [(1, 96, 1), (2, 96, 5)])
def test_fal_and_fcl_are_zero_for_identical_forecasts(shape):
    target = torch.randn(shape)

    fal = FourierAmplitudeLoss()(target, target)
    fcl = FourierCorrelationLoss()(target, target)

    assert fal.item() == pytest.approx(0.0, abs=1e-7)
    assert fcl.item() == pytest.approx(0.0, abs=1e-6)


def test_fcl_handles_two_zero_series_without_nan():
    values = torch.zeros(1, 96, 1, requires_grad=True)

    loss = FourierCorrelationLoss()(values, values)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert torch.isfinite(values.grad).all()


def test_facl_schedule_start_transition_and_end():
    criterion = FourierAmplitudeCorrelationLoss(alpha=0.2)

    criterion.set_progress(current_step=0, total_steps=11)
    assert criterion.probability_threshold == pytest.approx(1.0)

    criterion.set_progress(current_step=4, total_steps=11)
    assert criterion.probability_threshold == pytest.approx(0.5)

    criterion.set_progress(current_step=8, total_steps=11)
    assert criterion.probability_threshold == pytest.approx(0.0)

    criterion.set_progress(current_step=10, total_steps=11)
    assert criterion.probability_threshold == pytest.approx(0.0)


def test_alpha_controls_the_final_fal_only_training_ratio():
    no_constant_tail = FourierAmplitudeCorrelationLoss(alpha=0.0)
    half_constant_tail = FourierAmplitudeCorrelationLoss(alpha=0.5)

    no_constant_tail.set_progress(current_step=5, total_steps=11)
    half_constant_tail.set_progress(current_step=5, total_steps=11)

    assert no_constant_tail.probability_threshold == pytest.approx(0.5)
    assert half_constant_tail.probability_threshold == pytest.approx(0.0)


def test_facl_random_selection_and_official_length_scaling():
    prediction = torch.randn(1, 96, 2)
    target = torch.randn(1, 96, 2)
    criterion = FourierAmplitudeCorrelationLoss(alpha=0.2)
    criterion.set_progress(current_step=4, total_steps=11)

    with patch('utils.losses.random.random', return_value=0.75):
        actual_fal = criterion(prediction, target)
    expected_fal = FourierAmplitudeLoss()(prediction, target) * math.sqrt(96)
    assert criterion.last_selected == 'fal'
    assert actual_fal.item() == pytest.approx(expected_fal.item(), rel=1e-6)

    with patch('utils.losses.random.random', return_value=0.25):
        actual_fcl = criterion(prediction, target)
    expected_fcl = FourierCorrelationLoss()(prediction, target) * math.sqrt(96)
    assert criterion.last_selected == 'fcl'
    assert actual_fcl.item() == pytest.approx(expected_fcl.item(), rel=1e-6)


def test_mse_builder_preserves_pytorch_baseline():
    prediction = torch.randn(2, 96, 3, requires_grad=True)
    target = torch.randn(2, 96, 3)

    criterion = build_forecasting_loss('MSE')
    actual = criterion(prediction, target)
    expected = nn.MSELoss()(prediction, target)

    assert isinstance(criterion, nn.MSELoss)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('alpha', [-0.1, 1.0])
def test_invalid_alpha_fails_fast(alpha):
    with pytest.raises(ValueError, match='alpha'):
        FourierAmplitudeCorrelationLoss(alpha=alpha)


def test_invalid_forecast_shape_fails_fast():
    with pytest.raises(ValueError, match=r'\[B, pred_len, C\]'):
        FourierAmplitudeLoss()(torch.randn(2, 96), torch.randn(2, 96))


def test_unknown_loss_name_fails_fast():
    with pytest.raises(ValueError, match='Unsupported'):
        build_forecasting_loss('unknown')
