import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from data_provider.data_factory import data_dict
from data_provider.data_loader import Dataset_ETT_hour
from data_provider.ett_pchip15 import (
    FUTURE_HOURS,
    HISTORY_HOURS,
    PRED_LEN_15MIN,
    SEQ_LEN_15MIN,
    TRAIN_END_HOUR,
    VAL_END_HOUR,
    Dataset_ETTh1_PCHIP15,
    build_etth1_pchip15_cache,
)


FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TOTAL_HOURS = 12 * 30 * 24 + 8 * 30 * 24


def _make_synthetic_hourly_frame():
    index = np.arange(TOTAL_HOURS, dtype=np.float64)
    dates = pd.date_range("2021-01-01 00:00:00", periods=TOTAL_HOURS, freq="h")
    data = {"date": dates}

    split_offset = np.zeros(TOTAL_HOURS, dtype=np.float64)
    split_offset[TRAIN_END_HOUR:VAL_END_HOUR] = 5_000.0
    split_offset[VAL_END_HOUR:] = 15_000.0

    for feature_index, feature_name in enumerate(FEATURE_COLUMNS, start=1):
        trend = feature_index * 0.25 * index
        curve = ((index % 24.0) ** 2) / (feature_index + 3.0)
        data[feature_name] = trend + curve + split_offset * feature_index
    return pd.DataFrame(data)


def _write_hourly_csv(root_dir, frame):
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    csv_path = root_path / "ETTh1.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def _build_cache(root_path):
    cache_dir = Path(root_path) / "pchip15_cache" / "ETTh1"
    metadata = build_etth1_pchip15_cache(
        root_path=str(root_path),
        data_path="ETTh1.csv",
        cache_dir=str(cache_dir),
    )
    return cache_dir, metadata


def _dataset_args(cache_dir):
    return SimpleNamespace(augmentation_ratio=0, pchip15_cache_dir=str(cache_dir))


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    root_path = tmp_path_factory.mktemp("etth1_pchip15_base")
    frame = _make_synthetic_hourly_frame()
    csv_path = _write_hourly_csv(root_path, frame)
    cache_dir, metadata = _build_cache(root_path)
    return {
        "root_path": root_path,
        "csv_path": csv_path,
        "frame": frame,
        "cache_dir": cache_dir,
        "metadata": metadata,
    }


@pytest.fixture(scope="module")
def modified_val_cache(tmp_path_factory):
    root_path = tmp_path_factory.mktemp("etth1_pchip15_modified")
    frame = _make_synthetic_hourly_frame()
    modified_from_hour = 9000
    frame.loc[modified_from_hour + 1:, FEATURE_COLUMNS] += 50_000.0
    csv_path = _write_hourly_csv(root_path, frame)
    cache_dir, metadata = _build_cache(root_path)
    return {
        "root_path": root_path,
        "csv_path": csv_path,
        "frame": frame,
        "cache_dir": cache_dir,
        "metadata": metadata,
        "modified_from_hour": modified_from_hour,
    }


def _make_pchip_dataset(bundle, flag, scale=True, timeenc=0, features="M", label_len=96):
    return Dataset_ETTh1_PCHIP15(
        args=_dataset_args(bundle["cache_dir"]),
        root_path=str(bundle["root_path"]),
        flag=flag,
        size=[SEQ_LEN_15MIN, label_len, PRED_LEN_15MIN],
        features=features,
        data_path="ETTh1.csv",
        target="OT",
        scale=scale,
        timeenc=timeenc,
    )


def test_pchip15_shapes_and_memmap(synthetic_cache):
    dataset = _make_pchip_dataset(synthetic_cache, flag="train", scale=True, timeenc=0, features="M", label_len=96)

    seq_x, seq_y, seq_x_mark, seq_y_mark, observation_mask = dataset[0]

    assert seq_x.shape == (SEQ_LEN_15MIN, len(FEATURE_COLUMNS))
    assert seq_y.shape == (96 + PRED_LEN_15MIN, len(FEATURE_COLUMNS))
    assert seq_x_mark.shape == (SEQ_LEN_15MIN, 5)
    assert seq_y_mark.shape == (96 + PRED_LEN_15MIN, 5)
    assert observation_mask.shape == (PRED_LEN_15MIN, 1)
    assert isinstance(dataset.cache_arrays["x"], np.memmap)
    assert isinstance(dataset.cache_arrays["future_y"], np.memmap)


def test_hourly_values_are_preserved_on_observation_points(synthetic_cache):
    dataset = _make_pchip_dataset(synthetic_cache, flag="train", scale=False, timeenc=0, features="M", label_len=96)
    frame = synthetic_cache["frame"]
    raw_values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    sample_index = 17
    seq_x, seq_y, _, _, observation_mask = dataset[sample_index]
    start_hour = int(dataset.sample_start_hours[sample_index])
    future_y = seq_y[dataset.label_len:]
    observed_positions = np.flatnonzero(observation_mask[:, 0] == 1.0)

    np.testing.assert_array_equal(observed_positions, np.arange(0, PRED_LEN_15MIN, 4))
    np.testing.assert_allclose(
        seq_x[::4],
        raw_values[start_hour - HISTORY_HOURS:start_hour],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        future_y[observed_positions],
        raw_values[start_hour:start_hour + FUTURE_HOURS],
        atol=1e-5,
        rtol=1e-5,
    )


def test_every_mask_contains_96_real_targets(synthetic_cache):
    for split_name in ("train", "val", "test"):
        dataset = _make_pchip_dataset(synthetic_cache, flag=split_name, scale=False, timeenc=1, features="M")
        probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for probe_index in probe_indices:
            _, _, _, _, observation_mask = dataset[probe_index]
            assert int(observation_mask.sum()) == FUTURE_HOURS


def test_future_changes_do_not_affect_causal_seq_x(synthetic_cache, modified_val_cache):
    base_dataset = _make_pchip_dataset(synthetic_cache, flag="val", scale=True, timeenc=0, features="M")
    modified_dataset = _make_pchip_dataset(modified_val_cache, flag="val", scale=True, timeenc=0, features="M")

    start_hour = modified_val_cache["modified_from_hour"]
    sample_index = start_hour - int(base_dataset.sample_start_hours[0])

    base_seq_x = base_dataset[sample_index][0]
    modified_seq_x = modified_dataset[sample_index][0]

    np.testing.assert_allclose(base_seq_x, modified_seq_x, atol=1e-5, rtol=1e-5)


def test_scaler_uses_only_train_hourly_observations(synthetic_cache):
    expected_scaler = StandardScaler()
    expected_scaler.fit(synthetic_cache["frame"][FEATURE_COLUMNS].to_numpy(dtype=np.float64)[:TRAIN_END_HOUR])

    np.testing.assert_allclose(
        np.asarray(synthetic_cache["metadata"]["scaler_mean"], dtype=np.float64),
        expected_scaler.mean_,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(synthetic_cache["metadata"]["scaler_scale"], dtype=np.float64),
        expected_scaler.scale_,
        atol=1e-10,
        rtol=1e-10,
    )


def test_target_ranges_do_not_cross_split_boundaries(synthetic_cache):
    expected_ranges = {
        "train": (HISTORY_HOURS, TRAIN_END_HOUR),
        "val": (TRAIN_END_HOUR, VAL_END_HOUR),
        "test": (VAL_END_HOUR, TOTAL_HOURS),
    }

    for split_name, (expected_first_start, split_end_hour) in expected_ranges.items():
        dataset = _make_pchip_dataset(synthetic_cache, flag=split_name, scale=False, timeenc=0, features="M")
        assert int(dataset.sample_start_hours[0]) == expected_first_start
        last_start = int(dataset.sample_start_hours[-1])
        assert last_start + ((PRED_LEN_15MIN - 1) / 4.0) < split_end_hour


def test_original_etth_hour_contract_and_registration_stay_unchanged(monkeypatch, synthetic_cache):
    original_drop = pd.DataFrame.drop

    def _compat_drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors="raise"):
        return original_drop(
            self,
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    monkeypatch.setattr(pd.DataFrame, "drop", _compat_drop)

    hourly_dataset = Dataset_ETT_hour(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=str(synthetic_cache["root_path"]),
        flag="train",
        size=[384, 96, 384],
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        timeenc=0,
        freq="h",
    )

    item = hourly_dataset[0]
    assert len(item) == 4
    seq_x, seq_y, seq_x_mark, seq_y_mark = item
    assert seq_x.shape == (384, 1)
    assert seq_y.shape == (96 + 384, 1)
    assert seq_x_mark.shape == (384, 4)
    assert seq_y_mark.shape == (96 + 384, 4)

    assert data_dict["ETTh1"] is Dataset_ETT_hour
    assert data_dict["ETTh2"] is Dataset_ETT_hour
    assert data_dict["ETTm1"] is not Dataset_ETTh1_PCHIP15
    assert data_dict["ETTh1_PCHIP15"] is Dataset_ETTh1_PCHIP15
