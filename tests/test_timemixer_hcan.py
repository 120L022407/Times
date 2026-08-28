from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from layers.HCAN import (
    HCAN,
    HierarchicalBinMapper,
    UncertaintyAwareClassifier,
    hierarchical_consistency_loss,
)
from models.TimeMixer import Model as TimeMixer
from models.TimeMixer_HCAN import Model as TimeMixerHCAN
from utils.losses import build_forecasting_loss


def _args(**overrides):
    values = {
        'task_name': 'long_term_forecast',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'features': 'MS',
        'd_model': 8,
        'e_layers': 1,
        'd_ff': 16,
        'moving_avg': 25,
        'dropout': 0.0,
        'embed': 'timeF',
        'freq': 'h',
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'top_k': 5,
        'use_norm': 1,
        'down_sampling_layers': 1,
        'down_sampling_window': 2,
        'down_sampling_method': 'avg',
        'hcan_kc': 2,
        'hcan_kf': 4,
        'hcan_hidden_dim': 16,
        'hcan_alpha': 1.0,
        'hcan_beta': 1.0,
        'hcan_gamma': 1.0,
        'hcan_annealing_steps': 10,
        'loss': 'mse',
        'use_multi_gpu': False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _inputs(batch_size=1):
    x = torch.randn(batch_size, 96, 7)
    x_mark = torch.randn(batch_size, 96, 4)
    dec = torch.zeros(batch_size, 144, 7)
    dec_mark = torch.randn(batch_size, 144, 4)
    return x, x_mark, dec, dec_mark


def test_original_timemixer_is_unchanged_by_wrapper_hook():
    baseline = TimeMixer(_args()).eval()
    wrapped = TimeMixerHCAN(_args()).eval()
    wrapped.backbone.load_state_dict(baseline.state_dict())
    inputs = _inputs()

    with torch.no_grad():
        expected = baseline(*inputs)
        actual = wrapped.backbone(*inputs)

    assert torch.equal(actual, expected)


def test_group_mapping_and_relative_targets_follow_training_quantiles():
    mapper = HierarchicalBinMapper(1, coarse_classes=2, fine_classes=4)
    mapper.fit(torch.arange(8, dtype=torch.float32).view(1, 8, 1))
    values = torch.tensor([[[0.0], [2.0], [4.0], [7.0]]])

    coarse_labels, coarse_relative = mapper.map_targets(values, 'coarse')
    fine_labels, fine_relative = mapper.map_targets(values, 'fine')

    assert coarse_labels.tolist() == [[[0], [0], [1], [1]]]
    assert fine_labels.tolist() == [[[0], [1], [2], [3]]]
    assert torch.all((coarse_relative >= 0) & (coarse_relative <= 1))
    assert torch.all((fine_relative >= 0) & (fine_relative <= 1))
    assert fine_relative[0, 0, 0].item() == pytest.approx(0.0)
    assert fine_relative[0, -1, 0].item() == pytest.approx(1.0)


def test_uac_and_hcl_are_finite_and_backward_safe():
    logits = torch.tensor(
        [[[[1000.0, -1000.0], [-1000.0, 1000.0]]]], requires_grad=True
    )
    labels = torch.tensor([[[0, 1]]])
    loss, probabilities, uncertainty = UncertaintyAwareClassifier(2)(
        logits, labels, annealing_coefficient=1.0
    )
    mapped = torch.flip(logits, dims=(-1,))
    hcl = hierarchical_consistency_loss(logits, mapped)
    (loss + hcl).backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(hcl)
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(uncertainty).all()
    assert torch.isfinite(logits.grad).all()


def test_hcan_attention_and_heads_have_expected_shapes():
    module = HCAN(1, pred_len=96, hidden_dim=16, coarse_classes=2, fine_classes=4)
    module.mapper.fit(torch.randn(2, 96, 1))
    prediction = module(torch.randn(1, 1, 96))

    assert prediction.shape == (1, 96, 1)
    assert module._last_outputs['attention'].shape == (1, 1, 1)
    assert module._last_outputs['coarse_logits'].shape == (1, 96, 1, 2)
    assert module._last_outputs['fine_logits'].shape == (1, 96, 1, 4)


def test_timemixer_hcan_forward_auxiliary_loss_and_backward_batch_one_cpu():
    model = TimeMixerHCAN(_args())
    train_targets = torch.linspace(-2, 2, 384).reshape(4, 96, 1)
    assert model.begin_auxiliary_fit()
    model.update_auxiliary_fit(train_targets[:2])
    model.update_auxiliary_fit(train_targets[2:])
    model.finalize_auxiliary_fit()
    model.set_training_progress(0, 10)

    prediction = model(*_inputs(batch_size=1))
    target = torch.randn_like(prediction)
    model.set_auxiliary_targets(target)
    mse_loss = F.mse_loss(prediction, target)
    total_loss = model.compose_training_loss(mse_loss)
    expected_loss = (
        model.last_loss_components['hierarchy']
        + model.hcan_beta * model.last_loss_components['consistency']
        + model.hcan_gamma * mse_loss.detach()
    )
    total_loss.backward()

    assert prediction.shape == (1, 96, 1)
    assert torch.isfinite(total_loss)
    assert total_loss.detach().item() == pytest.approx(expected_loss.item(), rel=1e-6)
    assert model.hcan.haa.prediction_head.weight.grad is not None
    assert torch.isfinite(model.hcan.haa.prediction_head.weight.grad).all()
    assert all(torch.isfinite(value) for value in model.last_loss_components.values())


@pytest.mark.parametrize('loss_name', ['ps', 'facl'])
def test_timemixer_hcan_supports_optional_forecast_losses(loss_name):
    model = TimeMixerHCAN(_args(loss=loss_name))
    model.hcan.mapper.fit(torch.linspace(-2, 2, 192).reshape(2, 96, 1))
    model.set_training_progress(0, 10)
    criterion = build_forecasting_loss(loss_name, model=model)
    if hasattr(criterion, 'set_progress'):
        criterion.set_progress(0, 10)

    prediction = model(*_inputs(batch_size=1))
    target = torch.randn_like(prediction)
    model.set_auxiliary_targets(target)
    total_loss = model.compose_training_loss(criterion(prediction, target))
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert model.hcan.haa.prediction_head.weight.grad is not None
    assert torch.isfinite(model.hcan.haa.prediction_head.weight.grad).all()


def test_validation_forward_needs_no_future_classification_labels():
    model = TimeMixerHCAN(_args()).eval()
    with torch.no_grad():
        prediction = model(*_inputs())
    assert prediction.shape == (1, 96, 1)


def test_boundaries_are_fixed_after_training_fit_and_ignore_validation_values():
    mapper = HierarchicalBinMapper(1, coarse_classes=2, fine_classes=4)
    mapper.fit(torch.linspace(-1, 1, 96).view(1, 96, 1))
    coarse_before = mapper.coarse_boundaries.clone()
    fine_before = mapper.fine_boundaries.clone()

    mapper.map_targets(torch.full((1, 96, 1), 1000.0), 'fine')

    assert torch.equal(mapper.coarse_boundaries, coarse_before)
    assert torch.equal(mapper.fine_boundaries, fine_before)
    with pytest.raises(RuntimeError, match='already fitted'):
        mapper.fit(torch.full((1, 96, 1), 1000.0))


@pytest.mark.parametrize(
    'overrides,pattern',
    [
        ({'hcan_kc': 1}, 'hcan_kc'),
        ({'hcan_kc': 3, 'hcan_kf': 4}, 'divisible'),
        ({'channel_independence': 0}, 'channel_independence'),
        ({'loss': 'fal'}, 'mse, ps, or facl'),
    ],
)
def test_invalid_core_configuration_fails_fast(overrides, pattern):
    with pytest.raises(ValueError, match=pattern):
        TimeMixerHCAN(_args(**overrides))
