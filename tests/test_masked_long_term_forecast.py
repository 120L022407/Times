import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "exp.exp_basic" not in sys.modules:
    exp_basic_module = types.ModuleType("exp.exp_basic")

    class Exp_Basic:
        def __init__(self, args):
            self.args = args
            self.device = getattr(args, "device", torch.device("cpu"))

    exp_basic_module.Exp_Basic = Exp_Basic
    sys.modules["exp.exp_basic"] = exp_basic_module

sys.modules.setdefault("patoolib", types.ModuleType("patoolib"))

if "sktime" not in sys.modules:
    sktime_module = types.ModuleType("sktime")
    sktime_datasets_module = types.ModuleType("sktime.datasets")

    def _unused_loader(*args, **kwargs):
        raise RuntimeError("sktime dataset loading is not used in these tests.")

    sktime_datasets_module.load_from_tsfile_to_dataframe = _unused_loader
    sktime_module.datasets = sktime_datasets_module
    sys.modules["sktime"] = sktime_module
    sys.modules["sktime.datasets"] = sktime_datasets_module

if "datasets" not in sys.modules:
    datasets_module = types.ModuleType("datasets")

    def _unused_hf_dataset(*args, **kwargs):
        raise RuntimeError("huggingface datasets loading is not used in these tests.")

    datasets_module.load_dataset = _unused_hf_dataset
    sys.modules["datasets"] = datasets_module

if "huggingface_hub" not in sys.modules:
    huggingface_hub_module = types.ModuleType("huggingface_hub")

    def _unused_hf_download(*args, **kwargs):
        raise RuntimeError("huggingface hub downloading is not used in these tests.")

    huggingface_hub_module.hf_hub_download = _unused_hf_download
    sys.modules["huggingface_hub"] = huggingface_hub_module

if "utils.tools" not in sys.modules:
    tools_module = types.ModuleType("utils.tools")

    class EarlyStopping:
        def __init__(self, *args, **kwargs):
            self.early_stop = False

        def __call__(self, *args, **kwargs):
            return None

    def adjust_learning_rate(*args, **kwargs):
        return None

    def visual(*args, **kwargs):
        return None

    tools_module.EarlyStopping = EarlyStopping
    tools_module.adjust_learning_rate = adjust_learning_rate
    tools_module.visual = visual
    sys.modules["utils.tools"] = tools_module

if "utils.dtw_metric" not in sys.modules:
    dtw_module = types.ModuleType("utils.dtw_metric")

    def _unused_dtw(*args, **kwargs):
        raise RuntimeError("DTW is not used in these masked long-term forecast tests.")

    dtw_module.dtw = _unused_dtw
    dtw_module.accelerated_dtw = _unused_dtw
    sys.modules["utils.dtw_metric"] = dtw_module

if "utils.augmentation" not in sys.modules:
    augmentation_module = types.ModuleType("utils.augmentation")

    def _unused_augmentation(*args, **kwargs):
        raise RuntimeError("Augmentation is not used in these masked long-term forecast tests.")

    augmentation_module.run_augmentation = _unused_augmentation
    augmentation_module.run_augmentation_single = _unused_augmentation
    sys.modules["utils.augmentation"] = augmentation_module


import exp.exp_long_term_forecasting as exp_module
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric


class ConstantForecastModel(nn.Module):
    def __init__(self, output_channels, bias_init=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))
        self.output_channels = output_channels

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        batch_size, out_len, _ = dec_inp.shape
        return self.bias * torch.ones(
            (batch_size, out_len, self.output_channels),
            device=dec_inp.device,
            dtype=dec_inp.dtype,
        )


class RecordingMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_loss = None

    def forward(self, pred, true):
        loss = torch.mean((pred - true) ** 2)
        self.last_loss = float(loss.detach().cpu().item())
        return loss


class CheckpointingEarlyStopping:
    def __init__(self, *args, **kwargs):
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        del val_loss
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))


def _make_args(**overrides):
    base = {
        "pred_len": 4,
        "label_len": 2,
        "features": "M",
        "eval_mask_mode": "auto",
        "use_amp": False,
        "use_multi_gpu": False,
        "use_gpu": False,
        "device": torch.device("cpu"),
        "learning_rate": 0.0,
        "train_epochs": 1,
        "patience": 1,
        "checkpoints": "unused-checkpoints",
        "inverse": False,
        "use_dtw": False,
        "lradj": "type1",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_exp(args=None, model=None):
    args = args or _make_args()
    model = model or ConstantForecastModel(output_channels=1)
    exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    exp.args = args
    exp.device = args.device
    exp.model = model
    return exp


def _make_batch(future_values, observation_mask=None, seq_len=3):
    future_values = torch.as_tensor(future_values, dtype=torch.float32)
    if future_values.ndim == 2:
        future_values = future_values.unsqueeze(0)

    batch_size, pred_len, channels = future_values.shape
    batch_x = torch.zeros((batch_size, seq_len, channels), dtype=torch.float32)
    history = torch.zeros((batch_size, 2, channels), dtype=torch.float32)
    batch_y = torch.cat([history, future_values], dim=1)
    batch_x_mark = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)
    batch_y_mark = torch.zeros((batch_size, 2 + pred_len, 1), dtype=torch.float32)

    if observation_mask is None:
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    observation_mask = torch.as_tensor(observation_mask, dtype=torch.float32)
    if observation_mask.ndim == 2:
        observation_mask = observation_mask.unsqueeze(0)
    if observation_mask.ndim == 1:
        observation_mask = observation_mask.view(1, pred_len, 1)
    return batch_x, batch_y, batch_x_mark, batch_y_mark, observation_mask


def _make_test_data():
    return SimpleNamespace(scale=False, inverse_transform=lambda array: array)


def test_unpack_batch_accepts_legacy_4_tuple():
    exp = _make_exp()
    batch = (1, 2, 3, 4)

    unpacked = exp._unpack_batch(batch)

    assert unpacked == (1, 2, 3, 4, None)


def test_unpack_batch_accepts_new_5_tuple():
    exp = _make_exp()
    batch = (1, 2, 3, 4, 5)

    unpacked = exp._unpack_batch(batch)

    assert unpacked == (1, 2, 3, 4, 5)


def test_train_loss_uses_all_points_even_when_mask_exists(tmp_path, monkeypatch):
    args = _make_args(checkpoints=str(tmp_path / "checkpoints"))
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=0.0))
    criterion = RecordingMSELoss()
    train_batch = _make_batch(
        future_values=[[[0.0], [100.0], [0.0], [0.0]]],
        observation_mask=[[[1.0], [0.0], [1.0], [1.0]]],
    )

    exp._get_data = lambda flag: (_make_test_data(), [train_batch])
    exp._select_optimizer = lambda: torch.optim.SGD(exp.model.parameters(), lr=0.0)
    exp._select_criterion = lambda: criterion
    exp.vali = lambda *args, **kwargs: 0.0

    monkeypatch.setattr(exp_module, "EarlyStopping", CheckpointingEarlyStopping)
    monkeypatch.setattr(exp_module, "adjust_learning_rate", lambda *args, **kwargs: None)

    exp.train("masked-train-uses-all-points")

    assert criterion.last_loss == pytest.approx(2500.0, rel=1e-6, abs=1e-6)


def test_validation_ignores_large_errors_where_mask_is_zero():
    args = _make_args(eval_mask_mode="auto")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=0.0))
    batch = _make_batch(
        future_values=[[[0.0], [100.0], [0.0], [0.0]]],
        observation_mask=[[[1.0], [0.0], [1.0], [1.0]]],
    )

    loss = exp.vali(_make_test_data(), [batch], nn.MSELoss())

    assert loss == pytest.approx(0.0, abs=1e-6)


def test_validation_uses_global_squared_error_sum_divided_by_global_valid_count():
    args = _make_args(eval_mask_mode="observed")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=0.0))
    batch_one = _make_batch(
        future_values=[[[10.0], [0.0], [0.0], [0.0]]],
        observation_mask=[[[1.0], [0.0], [0.0], [0.0]]],
    )
    batch_two = _make_batch(
        future_values=[[[1.0], [1.0], [1.0], [0.0]]],
        observation_mask=[[[1.0], [1.0], [1.0], [0.0]]],
    )

    loss = exp.vali(_make_test_data(), [batch_one, batch_two], nn.MSELoss())

    assert loss == pytest.approx(25.75, rel=1e-6, abs=1e-6)


def test_all_one_mask_matches_original_metric():
    pred = np.array([[[1.0, 4.0], [2.0, 5.0]], [[3.0, 6.0], [4.0, 7.0]]], dtype=np.float32)
    true = np.array([[[1.5, 3.0], [3.0, 6.0]], [[2.0, 7.0], [5.0, 8.0]]], dtype=np.float32)
    all_one_mask = np.ones((2, 2, 1), dtype=np.float32)

    original_metrics = metric(pred, true)
    masked_metrics = metric(pred, true, mask=all_one_mask)

    np.testing.assert_allclose(masked_metrics, original_metrics, rtol=1e-6, atol=1e-6)


def test_ms_mode_only_evaluates_ot_and_saves_observed_mask(tmp_path, monkeypatch):
    args = _make_args(features="MS", eval_mask_mode="auto")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=5.0))
    batch = _make_batch(
        future_values=[[[100.0, 5.0], [100.0, 5.0], [100.0, 5.0], [100.0, 5.0]]],
        observation_mask=[[[1.0], [1.0], [1.0], [1.0]]],
    )

    exp._get_data = lambda flag: (_make_test_data(), [batch])
    monkeypatch.chdir(tmp_path)

    exp.test("ms-observed-only")

    result_dir = tmp_path / "results" / "ms-observed-only"
    metrics_real_only = json.loads((result_dir / "metrics_real_only.json").read_text())

    assert metrics_real_only["evaluation_scope"] == "observed-only"
    assert metrics_real_only["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics_real_only["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics_real_only["observed_time_count"] == 4
    assert metrics_real_only["evaluated_value_count"] == 4

    observed_mask = np.load(result_dir / "observed_mask.npy")
    assert observed_mask.shape == (1, 4, 1)


def test_m_mode_mask_broadcasts_to_all_variables_in_test_metrics(tmp_path, monkeypatch):
    args = _make_args(features="M", eval_mask_mode="auto")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=2, bias_init=0.0))
    batch = _make_batch(
        future_values=[[[1.0, 2.0], [999.0, 999.0], [3.0, 4.0], [999.0, 999.0]]],
        observation_mask=[[[1.0], [0.0], [1.0], [0.0]]],
    )

    exp._get_data = lambda flag: (_make_test_data(), [batch])
    monkeypatch.chdir(tmp_path)

    exp.test("m-broadcast-mask")

    result_dir = tmp_path / "results" / "m-broadcast-mask"
    metrics_real_only = json.loads((result_dir / "metrics_real_only.json").read_text())

    assert metrics_real_only["evaluation_scope"] == "observed-only"
    assert metrics_real_only["observed_time_count"] == 2
    assert metrics_real_only["evaluated_value_count"] == 4
    assert metrics_real_only["mae"] == pytest.approx(2.5, rel=1e-6, abs=1e-6)
    assert metrics_real_only["mse"] == pytest.approx(7.5, rel=1e-6, abs=1e-6)
    assert metrics_real_only["rmse"] == pytest.approx(math.sqrt(7.5), rel=1e-6, abs=1e-6)


def test_observed_mode_without_mask_fails_fast():
    args = _make_args(eval_mask_mode="observed")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=0.0))
    batch = _make_batch(future_values=[[[0.0], [0.0], [0.0], [0.0]]])

    with pytest.raises(ValueError, match="requires observation_mask"):
        exp.vali(_make_test_data(), [batch], nn.MSELoss())


def test_all_zero_mask_fails_fast():
    args = _make_args(eval_mask_mode="observed")
    exp = _make_exp(args=args, model=ConstantForecastModel(output_channels=1, bias_init=0.0))
    batch = _make_batch(
        future_values=[[[0.0], [0.0], [0.0], [0.0]]],
        observation_mask=[[[0.0], [0.0], [0.0], [0.0]]],
    )

    with pytest.raises(ValueError, match="zero valid evaluation points"):
        exp.vali(_make_test_data(), [batch], nn.MSELoss())
