import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


SEQ_LEN_15MIN = 384
PRED_LEN_15MIN = 384
POINTS_PER_HOUR = 4
HISTORY_HOURS = SEQ_LEN_15MIN // POINTS_PER_HOUR
FUTURE_HOURS = PRED_LEN_15MIN // POINTS_PER_HOUR
RESOLUTION = "15min"
TRAIN_END_HOUR = 12 * 30 * 24
VAL_END_HOUR = TRAIN_END_HOUR + 4 * 30 * 24
TEST_END_HOUR = VAL_END_HOUR + 4 * 30 * 24
SPLIT_TARGET_RANGES = {
    "train": (0, TRAIN_END_HOUR),
    "val": (TRAIN_END_HOUR, VAL_END_HOUR),
    "test": (VAL_END_HOUR, TEST_END_HOUR),
}
TIMEENC0_SUFFIX = "timeenc0"
TIMEENC1_SUFFIX = "timeenc1"
METADATA_FILENAME = "metadata.json"


def default_etth1_pchip15_cache_dir(root_path, data_path="ETTh1.csv"):
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(root_path, "pchip15_cache", data_name)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_timeenc0_mark(index):
    stamp = pd.DataFrame({"date": pd.DatetimeIndex(index)})
    stamp["month"] = stamp["date"].dt.month
    stamp["day"] = stamp["date"].dt.day
    stamp["weekday"] = stamp["date"].dt.weekday
    stamp["hour"] = stamp["date"].dt.hour
    stamp["minute"] = stamp["date"].dt.minute // 15
    return stamp[["month", "day", "weekday", "hour", "minute"]].to_numpy(dtype=np.float32)


def _build_timeenc1_mark(index):
    encoded = time_features(pd.DatetimeIndex(index), freq=RESOLUTION)
    return encoded.transpose(1, 0).astype(np.float32)


def _compute_split_infos(total_hours):
    if total_hours < TEST_END_HOUR:
        raise ValueError(
            f"ETTh1 PCHIP15 expects at least {TEST_END_HOUR} hourly rows, got {total_hours}."
        )

    max_supported_start = total_hours - FUTURE_HOURS - 1
    split_infos = {}
    for split_name, (target_start, target_end) in SPLIT_TARGET_RANGES.items():
        start_hour_min = max(target_start, HISTORY_HOURS)
        start_hour_max = min(target_end - FUTURE_HOURS, max_supported_start)
        if start_hour_max < start_hour_min:
            sample_count = 0
        else:
            sample_count = start_hour_max - start_hour_min + 1
        split_infos[split_name] = {
            "target_start_hour": target_start,
            "target_end_hour": target_end,
            "sample_start_hour_min": start_hour_min,
            "sample_start_hour_max": start_hour_max,
            "sample_count": sample_count,
        }
    return split_infos


def _build_split_cache(
    cache_dir,
    split_name,
    split_info,
    hourly_index,
    raw_values,
    observed_timestamps_ns,
):
    from scipy.interpolate import PchipInterpolator

    sample_count = split_info["sample_count"]
    num_features = raw_values.shape[1]
    file_map = {
        "x": os.path.join(cache_dir, f"{split_name}_x.npy"),
        "future_y": os.path.join(cache_dir, f"{split_name}_future_y.npy"),
        "x_mark_timeenc0": os.path.join(cache_dir, f"{split_name}_x_mark_{TIMEENC0_SUFFIX}.npy"),
        "future_y_mark_timeenc0": os.path.join(
            cache_dir, f"{split_name}_future_y_mark_{TIMEENC0_SUFFIX}.npy"
        ),
        "x_mark_timeenc1": os.path.join(cache_dir, f"{split_name}_x_mark_{TIMEENC1_SUFFIX}.npy"),
        "future_y_mark_timeenc1": os.path.join(
            cache_dir, f"{split_name}_future_y_mark_{TIMEENC1_SUFFIX}.npy"
        ),
        "observation_mask": os.path.join(cache_dir, f"{split_name}_observation_mask.npy"),
    }

    arrays = {
        "x": np.lib.format.open_memmap(
            file_map["x"], mode="w+", dtype=np.float32, shape=(sample_count, SEQ_LEN_15MIN, num_features)
        ),
        "future_y": np.lib.format.open_memmap(
            file_map["future_y"],
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, PRED_LEN_15MIN, num_features),
        ),
        "x_mark_timeenc0": np.lib.format.open_memmap(
            file_map["x_mark_timeenc0"], mode="w+", dtype=np.float32, shape=(sample_count, SEQ_LEN_15MIN, 5)
        ),
        "future_y_mark_timeenc0": np.lib.format.open_memmap(
            file_map["future_y_mark_timeenc0"],
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, PRED_LEN_15MIN, 5),
        ),
        "x_mark_timeenc1": np.lib.format.open_memmap(
            file_map["x_mark_timeenc1"], mode="w+", dtype=np.float32, shape=(sample_count, SEQ_LEN_15MIN, 5)
        ),
        "future_y_mark_timeenc1": np.lib.format.open_memmap(
            file_map["future_y_mark_timeenc1"],
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, PRED_LEN_15MIN, 5),
        ),
        "observation_mask": np.lib.format.open_memmap(
            file_map["observation_mask"], mode="w+", dtype=np.float32, shape=(sample_count, PRED_LEN_15MIN, 1)
        ),
    }

    start_min = split_info["sample_start_hour_min"]
    start_max = split_info["sample_start_hour_max"]
    x_grid = np.arange(SEQ_LEN_15MIN, dtype=np.float64) / POINTS_PER_HOUR
    future_grid = np.arange(PRED_LEN_15MIN, dtype=np.float64) / POINTS_PER_HOUR

    for row_index, start_hour in enumerate(range(start_min, start_max + 1)):
        input_hours = np.arange(start_hour - HISTORY_HOURS, start_hour + 1, dtype=np.float64)
        future_hours = np.arange(start_hour, start_hour + FUTURE_HOURS + 1, dtype=np.float64)

        input_values = raw_values[start_hour - HISTORY_HOURS:start_hour + 1]
        future_values = raw_values[start_hour:start_hour + FUTURE_HOURS + 1]

        input_interp = PchipInterpolator(input_hours, input_values, axis=0)
        future_interp = PchipInterpolator(future_hours, future_values, axis=0)

        arrays["x"][row_index] = input_interp((start_hour - HISTORY_HOURS) + x_grid).astype(np.float32)
        arrays["future_y"][row_index] = future_interp(start_hour + future_grid).astype(np.float32)

        prediction_start_ts = hourly_index[start_hour]
        x_index = pd.date_range(
            start=prediction_start_ts - pd.Timedelta(hours=HISTORY_HOURS),
            periods=SEQ_LEN_15MIN,
            freq=RESOLUTION,
        )
        future_index = pd.date_range(start=prediction_start_ts, periods=PRED_LEN_15MIN, freq=RESOLUTION)

        arrays["x_mark_timeenc0"][row_index] = _build_timeenc0_mark(x_index)
        arrays["future_y_mark_timeenc0"][row_index] = _build_timeenc0_mark(future_index)
        arrays["x_mark_timeenc1"][row_index] = _build_timeenc1_mark(x_index)
        arrays["future_y_mark_timeenc1"][row_index] = _build_timeenc1_mark(future_index)

        mask = np.isin(
            future_index.to_numpy(dtype="datetime64[ns]"),
            observed_timestamps_ns,
        ).astype(np.float32).reshape(PRED_LEN_15MIN, 1)
        if int(mask.sum()) != FUTURE_HOURS:
            raise ValueError(
                f"{split_name} sample at hour {start_hour} expected {FUTURE_HOURS} real target points, got {int(mask.sum())}."
            )
        arrays["observation_mask"][row_index] = mask

    for array in arrays.values():
        array.flush()
    return file_map


def build_etth1_pchip15_cache(root_path, data_path="ETTh1.csv", cache_dir=None):
    data_file = os.path.join(root_path, data_path)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Hourly source file not found: {data_file}")

    cache_dir = cache_dir or default_etth1_pchip15_cache_dir(root_path, data_path)
    os.makedirs(cache_dir, exist_ok=True)

    raw_sha256 = _sha256_file(data_file)
    df_raw = pd.read_csv(data_file)
    if "date" not in df_raw.columns:
        raise ValueError("ETTh1 PCHIP15 expects a 'date' column in the hourly source file.")

    df_raw = df_raw.iloc[:TEST_END_HOUR].copy()
    feature_order = [column for column in df_raw.columns if column != "date"]
    if len(feature_order) != 7:
        raise ValueError(f"ETTh1 PCHIP15 expects 7 variables, got {len(feature_order)}: {feature_order}")

    hourly_index = pd.to_datetime(df_raw["date"])
    raw_values = df_raw[feature_order].to_numpy(dtype=np.float64)

    split_infos = _compute_split_infos(len(df_raw))
    scaler = StandardScaler()
    scaler.fit(raw_values[:TRAIN_END_HOUR])

    observed_timestamps_ns = hourly_index.to_numpy(dtype="datetime64[ns]")
    cache_files = {}
    for split_name in ("train", "val", "test"):
        cache_files[split_name] = _build_split_cache(
            cache_dir=cache_dir,
            split_name=split_name,
            split_info=split_infos[split_name],
            hourly_index=hourly_index,
            raw_values=raw_values,
            observed_timestamps_ns=observed_timestamps_ns,
        )

    metadata = {
        "raw_file": os.path.abspath(data_file),
        "raw_sha256": raw_sha256,
        "feature_order": feature_order,
        "pchip_method": "scipy.interpolate.PchipInterpolator",
        "resolution": RESOLUTION,
        "seq_len": SEQ_LEN_15MIN,
        "pred_len": PRED_LEN_15MIN,
        "history_hours": HISTORY_HOURS,
        "future_hours": FUTURE_HOURS,
        "split_sample_counts": {
            split_name: split_infos[split_name]["sample_count"] for split_name in ("train", "val", "test")
        },
        "split_infos": split_infos,
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "causal_input_only": True,
        "real_target_points_per_sample": FUTURE_HOURS,
        "cache_files": cache_files,
    }

    metadata_path = os.path.join(cache_dir, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return metadata


class Dataset_ETTh1_PCHIP15(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq=RESOLUTION,
        seasonal_patterns=None,
    ):
        del seasonal_patterns
        self.args = args
        self.seq_len = SEQ_LEN_15MIN
        self.pred_len = PRED_LEN_15MIN

        if size is None:
            self.label_len = 96
        else:
            seq_len, label_len, pred_len = size
            if seq_len != SEQ_LEN_15MIN or pred_len != PRED_LEN_15MIN:
                raise ValueError(
                    f"ETTh1_PCHIP15 is fixed to seq_len={SEQ_LEN_15MIN}, pred_len={PRED_LEN_15MIN}; "
                    f"got seq_len={seq_len}, pred_len={pred_len}."
                )
            self.label_len = label_len

        if self.label_len < 0 or self.label_len > self.seq_len:
            raise ValueError(f"label_len must be in [0, {self.seq_len}], got {self.label_len}.")
        if flag not in ("train", "val", "test"):
            raise ValueError(f"Unsupported flag for ETTh1_PCHIP15: {flag}")
        if timeenc not in (0, 1):
            raise ValueError(f"timeenc must be 0 or 1, got {timeenc}")

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = RESOLUTION
        self.root_path = root_path
        self.data_path = data_path
        self.cache_dir = getattr(args, "pchip15_cache_dir", None) or default_etth1_pchip15_cache_dir(
            root_path, data_path
        )

        self._load_cache()

    def _load_cache(self):
        metadata_path = os.path.join(self.cache_dir, METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Missing ETTh1_PCHIP15 cache metadata at {metadata_path}. "
                "Run scripts/data/build_etth1_pchip15.py first."
            )

        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        if self.metadata["seq_len"] != SEQ_LEN_15MIN or self.metadata["pred_len"] != PRED_LEN_15MIN:
            raise ValueError("ETTh1_PCHIP15 cache metadata does not match the fixed 15min protocol.")

        feature_order = self.metadata["feature_order"]
        if self.target not in feature_order:
            raise ValueError(f"Target '{self.target}' not found in cache feature order: {feature_order}")

        if self.features == "S":
            self.feature_indices = [feature_order.index(self.target)]
        elif self.features in ("M", "MS"):
            self.feature_indices = list(range(len(feature_order)))
        else:
            raise ValueError(f"Unsupported features setting for ETTh1_PCHIP15: {self.features}")

        self.feature_order = feature_order
        self.selected_feature_order = [feature_order[index] for index in self.feature_indices]
        scaler_mean = np.asarray(self.metadata["scaler_mean"], dtype=np.float32)
        scaler_scale = np.asarray(self.metadata["scaler_scale"], dtype=np.float32)
        self.scaler_mean = scaler_mean[self.feature_indices]
        self.scaler_scale = scaler_scale[self.feature_indices]
        self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

        file_map = self.metadata["cache_files"][self.flag]
        mark_suffix = TIMEENC0_SUFFIX if self.timeenc == 0 else TIMEENC1_SUFFIX
        self.cache_paths = {
            "x": file_map["x"],
            "future_y": file_map["future_y"],
            "x_mark": file_map[f"x_mark_{mark_suffix}"],
            "future_y_mark": file_map[f"future_y_mark_{mark_suffix}"],
            "observation_mask": file_map["observation_mask"],
        }
        self.cache_arrays = {
            "x": np.load(self.cache_paths["x"], mmap_mode="r"),
            "future_y": np.load(self.cache_paths["future_y"], mmap_mode="r"),
            "x_mark": np.load(self.cache_paths["x_mark"], mmap_mode="r"),
            "future_y_mark": np.load(self.cache_paths["future_y_mark"], mmap_mode="r"),
            "observation_mask": np.load(self.cache_paths["observation_mask"], mmap_mode="r"),
        }

        split_info = self.metadata["split_infos"][self.flag]
        sample_count = split_info["sample_count"]
        if self.cache_arrays["x"].shape[0] != sample_count:
            raise ValueError(
                f"{self.flag} cache sample count mismatch: metadata={sample_count}, file={self.cache_arrays['x'].shape[0]}"
            )
        self.sample_start_hours = np.arange(
            split_info["sample_start_hour_min"],
            split_info["sample_start_hour_max"] + 1,
            dtype=np.int32,
        )
        self.split_target_range = (
            split_info["target_start_hour"],
            split_info["target_end_hour"],
        )

    def __len__(self):
        return self.cache_arrays["x"].shape[0]

    def _select_features(self, batch):
        if len(self.feature_indices) == batch.shape[-1]:
            return batch
        return batch[..., self.feature_indices]

    def _transform(self, batch):
        return ((batch - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    def __getitem__(self, index):
        seq_x = np.asarray(self._select_features(self.cache_arrays["x"][index]), dtype=np.float32)
        future_y = np.asarray(self._select_features(self.cache_arrays["future_y"][index]), dtype=np.float32)
        seq_x_mark = np.asarray(self.cache_arrays["x_mark"][index], dtype=np.float32)
        future_y_mark = np.asarray(self.cache_arrays["future_y_mark"][index], dtype=np.float32)
        observation_mask = np.asarray(self.cache_arrays["observation_mask"][index], dtype=np.float32)

        if self.scale:
            seq_x = self._transform(seq_x)
            future_y = self._transform(future_y)

        if self.label_len == 0:
            seq_y = future_y
            seq_y_mark = future_y_mark
        else:
            seq_y = np.concatenate([seq_x[-self.label_len:], future_y], axis=0)
            seq_y_mark = np.concatenate([seq_x_mark[-self.label_len:], future_y_mark], axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, observation_mask

    def inverse_transform(self, data):
        return np.asarray(data, dtype=np.float32) * self.scaler_scale + self.scaler_mean
