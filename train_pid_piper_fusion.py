#!/usr/bin/env python3
"""Train PID-Piper compatible gate-NN models.

This script trains per-angle gate models to blend y_pid and the existing
PID-Piper ML/reference model output y_ml on attack data.

Fusion formula at inference:
    y_fused = y_pid + alpha * (y_ml - y_pid)
where alpha is predicted in [0, 1].

Input files:
- attack_*.csv from PID_Piper logging, or Data_Piper_Training_Wide.processed.csv
- optional clean long CSV for compatibility/reporting only

Outputs:
- Keras .h5 gate models, per angle
- Optional frugally-deep JSON exports
- training_summary.json with metrics and paths
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# UNC/mounted paths can fail with h5py file locking on Windows.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: tensorflow. Install it first, e.g. pip install tensorflow"
    ) from exc


# Keep features aligned with current C++ LSTM.cpp tensor_shape(18) input.
BASE_FEATURES_18 = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "pos_x",
    "pos_y",
    "pos_z",
    "gpsVel",
    "ahrsRP",
    "ahrsYaw",
    "posVarH",
    "posVarV",
    "velVarX",
    "velVarY",
    "navRoll",
    "navPitch",
]

# Extra gate context from attack table.
GATE_EXTRA_FEATURES = [
    "y_pid",
    "y_ml",
    "residual",
    "ml_pid_gap_abs",
]
GATE_FEATURES_22 = BASE_FEATURES_18 + GATE_EXTRA_FEATURES

REQUIRED_CLEAN_LONG = BASE_FEATURES_18 + ["angle_type", "y_true", "timestamp"]
REQUIRED_ATTACK = BASE_FEATURES_18 + [
    "angle_type",
    "timestamp",
    "y_pid",
    "y_ml",
]

ALPHA_TARGET_COLUMNS = ["alpha_star", "alpha_target", "gate_alpha_target"]
CONTINUOUS_TARGET_COLUMNS = ["y_target", "y_true", "y_ref"]
BINARY_TARGET_COLUMNS = ["attack_label", "recovery_mode"]

VALID_ANGLES = ["roll", "pitch", "yaw"]

# -----------------------------------------------------------------------------
# IDE RUN CONFIG (edit only this section for click-to-run debugging)
# -----------------------------------------------------------------------------
# Behavior:
# - If script is launched without CLI args (e.g., VS Code Run Python File),
#   it will use IDE_RUN_CONFIG below.
# - If CLI args are provided, CLI args take precedence.
USE_IDE_CONFIG_WHEN_NO_ARGS = True

IDE_RUN_CONFIG = {
    "clean_long": None,
    "attack": None,
    "output_dir": "training_artifacts",
    "angles": ["roll", "pitch", "yaw"],
    "epochs_gate": 8,
    "batch_size": 2048,
    "val_fraction": 0.2,
    "lr_gate": 5e-4,
    "seed": 42,
    "normalize_features": True,
    "z_clip": 8.0,
    "save_diagnostics": True,
    "export_fdeep": False,
    "convert_script": "frugally-deep/keras_export/convert_model.py",
    "runtime_bundle_dir": "runtime_gate_bundle",
    "gate_target_column": "auto",
    "gate_architecture": "lstm",
    "gate_output_activation": "hard_sigmoid",
    "sequence_length": 100,
    "min_ml_pid_gap": 1.0e-4,
    "enforce_target_reachability": True,
    "target_reachability_margin": 0.02,
    "alpha_target_cap": 1.0,
    "gate_loss": "fused_huber",
    "alpha_l2_penalty": 0.0,
    "fit_verbose": 1,
}
# -----------------------------------------------------------------------------


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def discover_latest_attack_csv(project_root: Path) -> Path | None:
    patterns = [
        "analysis_*/gate_training_attack_with_target.csv",
        "analysis_*/Data_Piper_Training_Wide.processed.csv",
        "analysis_*/Data_Piper_Training_Wide.csv",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.is_file():
                candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


def build_args_for_ide() -> argparse.Namespace:
    return argparse.Namespace(**IDE_RUN_CONFIG)


def str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def fit_standard_scaler(x: np.ndarray, feature_names: List[str], eps: float = 1e-8) -> Dict[str, object]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < eps, 1.0, std)
    return {
        "feature_names": feature_names,
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


def apply_standard_scaler(
    x: np.ndarray,
    scaler: Dict[str, object],
    z_clip: float | None,
) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    z = (x - mean) / std
    if z_clip is not None and z_clip > 0:
        z = np.clip(z, -z_clip, z_clip)
    return z.astype(np.float32)


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def summarize_feature_matrix(x: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        stats[name] = summarize_array(x[:, idx])
    return stats


def infer_angle_unit_from_values(values: np.ndarray) -> str:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return "unknown"
    q99 = float(np.percentile(finite, 99))
    if q99 > 720.0:
        return "centideg"
    if q99 > (2.0 * math.pi + 0.5):
        return "deg"
    return "rad"


def angle_values_to_rad(values: np.ndarray, wrap_yaw: bool) -> Tuple[np.ndarray, str]:
    unit = infer_angle_unit_from_values(values)
    out = values.astype(np.float64, copy=True)
    if unit == "deg":
        out = np.deg2rad(out)
    elif unit == "centideg":
        out = out * (math.pi / 18000.0)
    if wrap_yaw:
        out = (out + math.pi) % (2.0 * math.pi) - math.pi
    return out.astype(np.float32), unit


def normalize_target_semantics(
    clean_long: pd.DataFrame | None,
    attack: pd.DataFrame,
    extra_attack_target_cols: List[str],
) -> Dict[str, object]:
    report: Dict[str, object] = {"clean": {}, "attack": {}}

    if clean_long is not None:
        # Clean target y_true should be radians. Yaw should be wrapped to [-pi, pi].
        for angle in VALID_ANGLES:
            mask = clean_long["angle_type"] == angle
            if not bool(mask.any()):
                continue
            raw = pd.to_numeric(clean_long.loc[mask, "y_true"], errors="coerce").to_numpy(dtype=np.float64)
            converted, unit = angle_values_to_rad(raw, wrap_yaw=(angle == "yaw"))
            clean_long.loc[mask, "y_true"] = converted
            report["clean"][angle] = {
                "source_unit": unit,
                "rows": int(mask.sum()),
            }

    # Attack targets should also be in radians for stable fusion training.
    attack_target_cols = [
        c for c in ["y_pid", "y_ml", "y_selected", "residual"] + extra_attack_target_cols
        if c in attack.columns
    ]
    for col in attack_target_cols:
        report["attack"][col] = {}
        for angle in VALID_ANGLES:
            mask = attack["angle_type"] == angle
            if not bool(mask.any()):
                continue
            raw = pd.to_numeric(attack.loc[mask, col], errors="coerce").to_numpy(dtype=np.float64)
            # residual is a magnitude; do not wrap.
            wrap = (col in {"y_pid", "y_ml", "y_selected"}) and (angle == "yaw")
            converted, unit = angle_values_to_rad(raw, wrap_yaw=wrap)
            attack.loc[mask, col] = converted
            report["attack"][col][angle] = {
                "source_unit": unit,
                "rows": int(mask.sum()),
            }

    return report


@dataclass
class AxisArtifacts:
    angle: str
    gate_h5: str
    gate_scaler_json: str
    diagnostics_json: str | None
    gate_json: str | None
    gate_train_rows: int
    gate_val_rows: int
    low_gap_rows: int
    unreachable_rows: int
    reachable_fraction: float | None
    gate_target_column: str
    gate_target_kind: str
    gate_architecture: str
    sequence_length: int
    mae_alpha_vs_target: float
    rmse_alpha_vs_target: float
    mae_fused_vs_target: float | None
    rmse_fused_vs_target: float | None
    mae_pid_vs_target: float | None
    rmse_pid_vs_target: float | None
    mae_ml_vs_target: float | None
    rmse_ml_vs_target: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_and_prepare(
    clean_long_path: Path | None,
    attack_path: Path,
    gate_target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    clean_long = pd.read_csv(clean_long_path) if clean_long_path is not None else None
    attack = pd.read_csv(attack_path)

    if clean_long is not None:
        clean_long.columns = [c.strip() for c in clean_long.columns]
    attack.columns = [c.strip() for c in attack.columns]

    if clean_long is not None:
        ensure_columns(clean_long, REQUIRED_CLEAN_LONG, "clean.long")
    ensure_columns(attack, REQUIRED_ATTACK, "attack.processed")
    preparation_report: Dict[str, object] = {
        "input_rows": {
            "clean_long": int(len(clean_long)) if clean_long is not None else None,
            "attack": int(len(attack)),
        },
        "derived_columns": {},
        "notes": [
            "Attack may be either raw attack_*.csv from PID_Piper mode1 logging or a *.processed.csv from preprocess_two_csv.py.",
            "The residual feature is recomputed after target-unit normalization to match runtime abs(y_ml - y_pid).",
            "Gate-only training does not train or emit reference models; y_ml is the existing PID-Piper ML/reference output captured in attack data.",
        ],
    }

    if clean_long is not None:
        clean_long["angle_type"] = clean_long["angle_type"].astype(str).str.strip().str.lower()
    attack["angle_type"] = attack["angle_type"].astype(str).str.strip().str.lower()

    if clean_long is not None:
        clean_long = clean_long[clean_long["angle_type"].isin(VALID_ANGLES)].copy()
    attack = attack[attack["angle_type"].isin(VALID_ANGLES)].copy()

    # Enforce numeric types used by training.
    if clean_long is not None:
        for col in BASE_FEATURES_18 + ["y_true", "timestamp"]:
            clean_long[col] = pd.to_numeric(clean_long[col], errors="coerce")

    for col in BASE_FEATURES_18 + ["timestamp", "y_pid", "y_ml"]:
        attack[col] = pd.to_numeric(attack[col], errors="coerce")
    optional_numeric_cols = [
        "residual",
        "y_selected",
        "y_fused",
        "alpha",
        "attack_label",
        "recovery_mode",
        "strategy_mode",
    ] + ALPHA_TARGET_COLUMNS + CONTINUOUS_TARGET_COLUMNS
    if gate_target_column != "auto":
        optional_numeric_cols.append(gate_target_column)
    for col in sorted(set(optional_numeric_cols)):
        if col in attack.columns:
            attack[col] = pd.to_numeric(attack[col], errors="coerce")

    if "residual" not in attack.columns:
        attack["residual"] = (attack["y_ml"] - attack["y_pid"]).abs()
        preparation_report["derived_columns"]["residual"] = "abs(y_ml - y_pid)"
    if "y_selected" not in attack.columns:
        attack["y_selected"] = attack["y_ml"]
        preparation_report["derived_columns"]["y_selected"] = "y_ml fallback for raw ML-only attack logs"

    if clean_long is not None:
        clean_long = clean_long.dropna(subset=BASE_FEATURES_18 + ["y_true", "timestamp"]).copy()
    attack = attack.dropna(
        subset=BASE_FEATURES_18
        + ["timestamp", "y_pid", "y_ml", "residual", "y_selected"]
    ).copy()

    if clean_long is not None:
        clean_long = clean_long.sort_values(["angle_type", "timestamp"]).reset_index(drop=True)
    sort_cols = ["angle_type"]
    for col in ["mission_id", "pair_id"]:
        if col in attack.columns:
            sort_cols.append(col)
    sort_cols.append("t_norm" if "t_norm" in attack.columns else "timestamp")
    attack = attack.sort_values(sort_cols).reset_index(drop=True)

    extra_attack_targets = [c for c in CONTINUOUS_TARGET_COLUMNS if c in attack.columns]
    semantic_report = normalize_target_semantics(
        clean_long,
        attack,
        sorted(set(extra_attack_targets)),
    )

    attack["residual"] = (attack["y_ml"] - attack["y_pid"]).abs()
    attack["ml_pid_gap_abs"] = attack["residual"]
    attack = attack.dropna(subset=GATE_FEATURES_22 + ["timestamp", "y_selected"]).copy()

    preparation_report["output_rows"] = {
        "clean_long": int(len(clean_long)) if clean_long is not None else None,
        "attack": int(len(attack)),
    }
    preparation_report["semantic_normalization"] = semantic_report

    print("[INFO] Target semantic normalization report:")
    print(json.dumps(semantic_report, ensure_ascii=True))

    return attack, preparation_report


def time_split(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    n = len(df)
    if n < 20:
        raise ValueError(f"Not enough rows for split: {n}")

    split_idx = int((1.0 - val_fraction) * n)
    split_idx = min(max(split_idx, 1), n - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_gate_model(
    input_dim: int,
    learning_rate: float,
    gate_architecture: str,
    sequence_length: int,
    gate_loss: str,
    alpha_l2_penalty: float,
    gate_output_activation: str,
) -> keras.Model:
    if alpha_l2_penalty < 0.0:
        raise ValueError("alpha_l2_penalty must be >= 0")
    if gate_output_activation not in {"sigmoid", "hard_sigmoid"}:
        raise ValueError(f"Unsupported gate output activation: {gate_output_activation}")

    if gate_architecture == "lstm":
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(sequence_length, input_dim)),
                keras.layers.LSTM(64, return_sequences=False),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation=gate_output_activation),
            ]
        )
    elif gate_architecture == "mlp":
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation=gate_output_activation),
            ]
        )
    else:
        raise ValueError(f"Unsupported gate architecture: {gate_architecture}")

    def alpha_mae_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if gate_loss == "fused_huber":
            alpha_true = y_true[:, 0:1]
        else:
            alpha_true = y_true
        return tf.reduce_mean(tf.abs(alpha_true - y_pred))

    if gate_loss == "alpha_huber":
        loss = keras.losses.Huber()
    elif gate_loss == "fused_huber":
        def fused_huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_target = y_true[:, 1:2]
            y_pid = y_true[:, 2:3]
            y_ml = y_true[:, 3:4]
            y_fused = y_pid + y_pred * (y_ml - y_pid)
            err = y_target - y_fused
            abs_err = tf.abs(err)
            delta = tf.constant(1.0, dtype=err.dtype)
            loss_value = tf.where(
                abs_err <= delta,
                0.5 * tf.square(err),
                delta * (abs_err - 0.5 * delta),
            )
            if alpha_l2_penalty > 0.0:
                penalty = tf.cast(alpha_l2_penalty, loss_value.dtype) * tf.square(y_pred)
                loss_value = loss_value + penalty
            return tf.squeeze(loss_value, axis=-1)

        loss = fused_huber_loss
    else:
        raise ValueError(f"Unsupported gate loss: {gate_loss}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[alpha_mae_metric],
    )
    return model


def make_fused_loss_targets(endpoint_df: pd.DataFrame) -> np.ndarray:
    required = ["alpha_star", "gate_continuous_target", "y_pid", "y_ml"]
    missing = [c for c in required if c not in endpoint_df.columns]
    if missing:
        raise ValueError(f"fused_huber loss requires continuous target columns: {missing}")
    return endpoint_df[required].to_numpy(dtype=np.float32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def split_gate_dataframe(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "mission_id" not in df.columns:
        return time_split(df, val_fraction)

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    group_cols = ["mission_id"]
    if "pair_id" in df.columns:
        group_cols.append("pair_id")
    for _key, group in df.groupby(group_cols, sort=False):
        train_part, val_part = time_split(group.reset_index(drop=True), val_fraction)
        train_parts.append(train_part)
        val_parts.append(val_part)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


def make_lstm_windows(
    df: pd.DataFrame,
    gate_cols: List[str],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2 for LSTM gate training")

    windows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    endpoint_frames: List[pd.DataFrame] = []
    group_cols = ["mission_id"] if "mission_id" in df.columns else ["__sequence_group"]
    if "pair_id" in df.columns and "pair_id" not in group_cols:
        group_cols.append("pair_id")

    work = df.copy()
    if "__sequence_group" in group_cols:
        work["__sequence_group"] = "single"

    sort_cols = ["t_norm"] if "t_norm" in work.columns else ["timestamp"]
    for _key, group in work.groupby(group_cols, sort=False):
        group = group.sort_values(sort_cols).reset_index(drop=True)
        x = group[gate_cols].to_numpy(dtype=np.float32)
        y = group["alpha_star"].to_numpy(dtype=np.float32)
        if len(group) < sequence_length:
            continue
        view = np.lib.stride_tricks.sliding_window_view(x, window_shape=sequence_length, axis=0)
        windows.append(np.moveaxis(view, -1, 1))
        targets.append(y[sequence_length - 1:])
        endpoint_frames.append(group.iloc[sequence_length - 1:].drop(columns=["__sequence_group"], errors="ignore"))

    if not windows:
        raise ValueError(
            f"No LSTM windows produced; need at least {sequence_length} rows per mission/axis split"
        )
    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    return np.concatenate(windows, axis=0).astype(np.float32), np.concatenate(targets).astype(np.float32), endpoints


class LstmWindowSequence(keras.utils.Sequence):
    """Batch generator for LSTM windows without materializing all windows."""

    def __init__(
        self,
        df: pd.DataFrame,
        gate_cols: List[str],
        sequence_length: int,
        batch_size: int,
        gate_loss: str,
        shuffle: bool,
        seed: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2 for LSTM gate training")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.gate_cols = gate_cols
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.gate_loss = gate_loss
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.groups: List[Dict[str, np.ndarray]] = []
        self.group_starts: np.ndarray
        self.window_counts: List[int] = []
        endpoint_frames: List[pd.DataFrame] = []

        group_cols = ["mission_id"] if "mission_id" in df.columns else ["__sequence_group"]
        if "pair_id" in df.columns and "pair_id" not in group_cols:
            group_cols.append("pair_id")

        work = df.copy()
        if "__sequence_group" in group_cols:
            work["__sequence_group"] = "single"

        sort_cols = ["t_norm"] if "t_norm" in work.columns else ["timestamp"]
        for _key, group in work.groupby(group_cols, sort=False):
            group = group.sort_values(sort_cols).reset_index(drop=True)
            if len(group) < self.sequence_length:
                continue

            x = group[gate_cols].to_numpy(dtype=np.float32, copy=True)
            if gate_loss == "fused_huber":
                y_fit = make_fused_loss_targets(group)
            elif gate_loss == "alpha_huber":
                y_fit = group["alpha_star"].to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
            else:
                raise ValueError(f"Unsupported gate loss: {gate_loss}")

            window_count = len(group) - self.sequence_length + 1
            self.groups.append({"x": x, "y_fit": y_fit})
            self.window_counts.append(window_count)
            endpoint_frames.append(
                group.iloc[self.sequence_length - 1:].drop(columns=["__sequence_group"], errors="ignore")
            )

        if not self.groups:
            raise ValueError(
                f"No LSTM windows produced; need at least {sequence_length} rows per mission/axis split"
            )

        self.group_starts = np.cumsum([0] + self.window_counts[:-1], dtype=np.int64)
        self.total_windows = int(sum(self.window_counts))
        self.order = np.arange(self.total_windows, dtype=np.int64)
        self.endpoints = pd.concat(endpoint_frames, ignore_index=True)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(self.total_windows / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        end = min(start + self.batch_size, self.total_windows)
        batch_indices = self.order[start:end]

        x_batch = np.empty(
            (len(batch_indices), self.sequence_length, len(self.gate_cols)),
            dtype=np.float32,
        )
        first_y = self.groups[0]["y_fit"]
        y_batch = np.empty((len(batch_indices), first_y.shape[1]), dtype=np.float32)

        group_ids = np.searchsorted(self.group_starts, batch_indices, side="right") - 1
        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            batch_positions = np.flatnonzero(mask)
            local_starts = batch_indices[mask] - self.group_starts[group_id]
            group = self.groups[int(group_id)]
            x = group["x"]
            y_fit = group["y_fit"]
            for out_pos, local_start in zip(batch_positions, local_starts):
                local_start_int = int(local_start)
                endpoint = local_start_int + self.sequence_length - 1
                x_batch[out_pos] = x[local_start_int:local_start_int + self.sequence_length]
                y_batch[out_pos] = y_fit[endpoint]

        return x_batch, y_batch


def select_gate_target_column(attack_axis: pd.DataFrame, requested: str) -> Tuple[str, str]:
    if requested != "auto":
        if requested not in attack_axis.columns:
            raise ValueError(f"Requested gate target column not found: {requested}")
        if requested in CONTINUOUS_TARGET_COLUMNS:
            return requested, "continuous"
        if requested in BINARY_TARGET_COLUMNS:
            return requested, "binary"
        return requested, "alpha"

    for col in ALPHA_TARGET_COLUMNS:
        if col in attack_axis.columns and attack_axis[col].notna().any():
            return col, "alpha"
    for col in CONTINUOUS_TARGET_COLUMNS:
        if col in attack_axis.columns and attack_axis[col].notna().any():
            return col, "continuous"
    for col in BINARY_TARGET_COLUMNS:
        if col in attack_axis.columns and attack_axis[col].notna().any():
            return col, "binary"

    raise ValueError(
        "No gate target column available. Provide attack_label/recovery_mode, "
        "an alpha target column, or a continuous y_target/y_true/y_ref column."
    )


def construct_gate_target(
    attack_axis: pd.DataFrame,
    requested: str,
    min_ml_pid_gap: float,
    alpha_target_cap: float,
) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, object]]:
    target_col, target_kind = select_gate_target_column(attack_axis, requested)
    target_values = attack_axis[target_col].to_numpy(dtype=np.float32)

    low_gap_rows = 0
    continuous_target: np.ndarray | None = None

    alpha_cap = float(alpha_target_cap)
    if alpha_cap <= 0.0 or alpha_cap > 1.0:
        raise ValueError("alpha_target_cap must be in (0, 1]")

    if target_kind == "continuous":
        y_pid = attack_axis["y_pid"].to_numpy(dtype=np.float32)
        y_ml = attack_axis["y_ml"].to_numpy(dtype=np.float32)
        gap = y_ml - y_pid
        gap_abs = np.abs(gap)
        min_gap = max(float(min_ml_pid_gap), 1.0e-12)
        stable_gap = gap_abs >= min_gap

        alpha_star = np.zeros_like(y_pid, dtype=np.float32)
        alpha_star[stable_gap] = np.clip(
            (target_values[stable_gap] - y_pid[stable_gap]) / gap[stable_gap],
            0.0,
            1.0,
        ).astype(np.float32)
        if alpha_cap < 1.0:
            alpha_star = np.clip(alpha_star, 0.0, alpha_cap).astype(np.float32)
        low_gap_rows = int((~stable_gap).sum())
        continuous_target = target_values
    else:
        if target_kind == "alpha":
            finite = target_values[np.isfinite(target_values)]
            if finite.size and (float(np.min(finite)) < -1.0e-6 or float(np.max(finite)) > 1.0 + 1.0e-6):
                raise ValueError(
                    f"Gate alpha target column {target_col} must be in [0, 1]; "
                    "use y_target/y_true/y_ref for continuous angle targets."
                )
        alpha_star = np.clip(target_values, 0.0, alpha_cap).astype(np.float32)

    target_report = {
        "requested": requested,
        "column": target_col,
        "kind": target_kind,
        "low_gap_rows": low_gap_rows,
        "alpha_target_cap": alpha_cap,
        "alpha_formula": (
            "clip((target - y_pid) / (y_ml - y_pid), 0, 1); low-gap rows use alpha=0"
            if target_kind == "continuous"
            else "clip(target_column, 0, alpha_target_cap)"
        ),
    }
    return alpha_star, continuous_target, target_report


def export_fdeep_json(model_h5: Path, output_json: Path, convert_script: Path) -> None:
    # Newer Keras objects can break test-vector generation inside legacy converter;
    # keep conversion deterministic by disabling embedded tests.
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_h5),
        str(output_json),
        "--no-tests",
    ]
    subprocess.run(cmd, check=True)


def cpp_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite scaler value: {value}")
    return f"{float(value):.9g}f"


def cpp_array(name: str, values: List[float]) -> str:
    chunks = [cpp_float(v) for v in values]
    lines = [f"static const std::array<float, kGateFeatureCount> {name} = {{"]
    for i in range(0, len(chunks), 6):
        lines.append("\t" + ", ".join(chunks[i:i + 6]) + ("," if i + 6 < len(chunks) else ""))
    lines.append("};")
    return "\n".join(lines)


def write_runtime_bundle(
    artifacts: List[AxisArtifacts],
    out_dir: Path,
    runtime_bundle_dir: str,
    gate_architecture: str,
    gate_output_activation: str,
    sequence_length: int,
) -> Dict[str, object]:
    bundle_dir = out_dir / runtime_bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    scaler_snippets: List[str] = [
        "// Generated by train_pid_piper_fusion.py.",
        "// Paste these arrays into simulator/libraries/PID_Piper/PID_Piper.cpp",
        "// together with the matching exported gate JSON files.",
        f"// Feature order: {', '.join(GATE_FEATURES_22)}",
        "",
    ]
    runtime_models: Dict[str, str] = {}
    scaler_files: Dict[str, str] = {}

    for artifact in artifacts:
        angle_title = artifact.angle.capitalize()
        gate_scaler_path = Path(artifact.gate_scaler_json)
        scaler_doc = json.loads(gate_scaler_path.read_text(encoding="utf-8"))
        feature_names = scaler_doc.get("feature_names", [])
        if feature_names != GATE_FEATURES_22:
            raise ValueError(
                f"{artifact.angle} gate scaler feature order mismatch: {feature_names}"
            )
        scaler = scaler_doc["scaler"]
        scaler_snippets.append(cpp_array(f"k{angle_title}GateMean", scaler["mean"]))
        scaler_snippets.append("")
        scaler_snippets.append(cpp_array(f"k{angle_title}GateStd", scaler["std"]))
        scaler_snippets.append("")

        scaler_files[artifact.angle] = str(gate_scaler_path)

        if artifact.gate_json:
            source_json = Path(artifact.gate_json)
            if source_json.is_file():
                runtime_name = f"{artifact.angle}-gate-nn.json"
                runtime_path = bundle_dir / runtime_name
                shutil.copy2(source_json, runtime_path)
                runtime_models[artifact.angle] = str(runtime_path)

    scaler_snippet_path = bundle_dir / "pid_piper_gate_scalers.cpp_snippet"
    scaler_snippet_path.write_text("\n".join(scaler_snippets).rstrip() + "\n", encoding="utf-8")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "feature_order": GATE_FEATURES_22,
        "gate_architecture": gate_architecture,
        "gate_output_activation": gate_output_activation,
        "sequence_length": sequence_length if gate_architecture == "lstm" else 1,
        "deployment_requires_cpp_sequence_buffer": gate_architecture == "lstm",
        "scaler_snippet": str(scaler_snippet_path),
        "gate_scaler_json": scaler_files,
        "runtime_gate_models": runtime_models,
        "runtime_model_names_expected_by_cpp": {
            angle: f"{angle}-gate-nn.json" for angle in VALID_ANGLES if angle in scaler_files
        },
        "note": "Do not deploy LSTM gate JSONs until PID_Piper.cpp has matching sequence-buffer inference support.",
    }
    manifest_path = bundle_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def train_for_angle(
    angle: str,
    attack: pd.DataFrame,
    out_dir: Path,
    epochs_gate: int,
    batch_size: int,
    val_fraction: float,
    lr_gate: float,
    normalize_features: bool,
    z_clip: float,
    save_diagnostics: bool,
    export_fdeep: bool,
    convert_script: Path,
    gate_target_column: str,
    gate_architecture: str,
    gate_output_activation: str,
    sequence_length: int,
    min_ml_pid_gap: float,
    enforce_target_reachability: bool,
    target_reachability_margin: float,
    alpha_target_cap: float,
    gate_loss: str,
    alpha_l2_penalty: float,
    fit_verbose: int,
) -> AxisArtifacts:
    axis_t0 = time.time()
    print(f"[INFO] {angle}: preparing rows", flush=True)
    attack_axis = attack[attack["angle_type"] == angle].copy()

    target_col, _target_kind = select_gate_target_column(attack_axis, gate_target_column)
    attack_axis = attack_axis.dropna(subset=[target_col]).copy()

    if len(attack_axis) < 200:
        raise ValueError(f"Angle {angle}: not enough attack rows ({len(attack_axis)})")

    alpha_star, continuous_target, target_report = construct_gate_target(
        attack_axis,
        gate_target_column,
        min_ml_pid_gap,
        alpha_target_cap,
    )

    gate_cols = GATE_FEATURES_22

    attack_for_split = attack_axis.copy()
    attack_for_split["alpha_star"] = alpha_star
    if continuous_target is not None:
        attack_for_split["gate_continuous_target"] = continuous_target
    unreachable_rows = 0
    reachable_fraction: float | None = None
    if target_report["kind"] == "continuous":
        stable_mask = attack_for_split["ml_pid_gap_abs"] >= max(float(min_ml_pid_gap), 1.0e-12)
        attack_for_split = attack_for_split[stable_mask].copy()
        y_target = attack_for_split["gate_continuous_target"].to_numpy(dtype=np.float32)
        y_pid = attack_for_split["y_pid"].to_numpy(dtype=np.float32)
        y_ml = attack_for_split["y_ml"].to_numpy(dtype=np.float32)
        lower = np.minimum(y_pid, y_ml) - float(target_reachability_margin)
        upper = np.maximum(y_pid, y_ml) + float(target_reachability_margin)
        reachable_mask = (y_target >= lower) & (y_target <= upper)
        unreachable_rows = int((~reachable_mask).sum())
        reachable_fraction = float(np.mean(reachable_mask)) if reachable_mask.size else None
        if enforce_target_reachability:
            attack_for_split = attack_for_split[reachable_mask].copy()

    if len(attack_for_split) < 200:
        raise ValueError(
            f"Angle {angle}: not enough usable attack rows after target filtering ({len(attack_for_split)})"
        )
    print(
        f"[INFO] {angle}: usable_rows={len(attack_for_split)}, "
        f"reachable_fraction={reachable_fraction}, alpha_cap={alpha_target_cap}",
        flush=True,
    )

    gate_train_df, gate_val_df = split_gate_dataframe(attack_for_split, val_fraction)

    gate_scaler = fit_standard_scaler(gate_train_df[gate_cols].to_numpy(dtype=np.float32), gate_cols)
    train_for_model = gate_train_df.copy()
    val_for_model = gate_val_df.copy()
    # Keep gate feature columns in float dtype so normalized float values can be
    # assigned without pandas incompatible-dtype warnings.
    for col in gate_cols:
        train_for_model[col] = train_for_model[col].astype(np.float32, copy=False)
        val_for_model[col] = val_for_model[col].astype(np.float32, copy=False)
    if normalize_features:
        train_for_model.loc[:, gate_cols] = apply_standard_scaler(
            train_for_model[gate_cols].to_numpy(dtype=np.float32),
            gate_scaler,
            z_clip=z_clip,
        )
        val_for_model.loc[:, gate_cols] = apply_standard_scaler(
            val_for_model[gate_cols].to_numpy(dtype=np.float32),
            gate_scaler,
            z_clip=z_clip,
        )

    if gate_architecture == "lstm":
        print(f"[INFO] {angle}: preparing streamed LSTM windows seq={sequence_length}", flush=True)
        train_sequence = LstmWindowSequence(
            train_for_model,
            gate_cols,
            sequence_length,
            batch_size=batch_size,
            gate_loss=gate_loss,
            shuffle=True,
            seed=12345,
        )
        val_sequence = LstmWindowSequence(
            val_for_model,
            gate_cols,
            sequence_length,
            batch_size=batch_size,
            gate_loss=gate_loss,
            shuffle=False,
            seed=12345,
        )
        train_endpoints = train_sequence.endpoints
        gate_val_df = val_sequence.endpoints
        y_gate_train = train_endpoints["alpha_star"].to_numpy(dtype=np.float32)
        y_gate_val = gate_val_df["alpha_star"].to_numpy(dtype=np.float32)
        train_window_count = train_sequence.total_windows
        val_window_count = val_sequence.total_windows
        print(
            f"[INFO] {angle}: train_windows={train_window_count}, val_windows={val_window_count} "
            "(streamed; windows are not materialized in RAM)",
            flush=True,
        )
    else:
        x_gate_train = train_for_model[gate_cols].to_numpy(dtype=np.float32)
        y_gate_train = train_for_model["alpha_star"].to_numpy(dtype=np.float32)
        x_gate_val = val_for_model[gate_cols].to_numpy(dtype=np.float32)
        y_gate_val = val_for_model["alpha_star"].to_numpy(dtype=np.float32)
        train_endpoints = train_for_model
        gate_val_df = val_for_model
        train_window_count = int(len(y_gate_train))
        val_window_count = int(len(y_gate_val))

    if gate_architecture != "lstm":
        if gate_loss == "fused_huber":
            y_gate_train_fit = make_fused_loss_targets(train_endpoints)
            y_gate_val_fit = make_fused_loss_targets(gate_val_df)
        elif gate_loss == "alpha_huber":
            y_gate_train_fit = y_gate_train.reshape(-1, 1)
            y_gate_val_fit = y_gate_val.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported gate loss: {gate_loss}")

    gate_model = build_gate_model(
        input_dim=len(gate_cols),
        learning_rate=lr_gate,
        gate_architecture=gate_architecture,
        sequence_length=sequence_length,
        gate_loss=gate_loss,
        alpha_l2_penalty=alpha_l2_penalty,
        gate_output_activation=gate_output_activation,
    )
    print(
        f"[INFO] {angle}: training epochs={epochs_gate}, batch_size={batch_size}, "
        f"loss={gate_loss}, alpha_l2={alpha_l2_penalty}, output={gate_output_activation}",
        flush=True,
    )
    if gate_architecture == "lstm":
        gate_model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs_gate,
            verbose=int(fit_verbose),
            workers=0,
            use_multiprocessing=False,
            max_queue_size=2,
        )
    else:
        gate_model.fit(
            x_gate_train,
            y_gate_train_fit,
            validation_data=(x_gate_val, y_gate_val_fit),
            epochs=epochs_gate,
            batch_size=batch_size,
            verbose=int(fit_verbose),
        )

    print(f"[INFO] {angle}: evaluating", flush=True)
    if gate_architecture == "lstm":
        alpha_pred_val = gate_model.predict(
            val_sequence,
            verbose=0,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=2,
        ).reshape(-1)
    else:
        alpha_pred_val = gate_model.predict(x_gate_val, verbose=0).reshape(-1)

    y_pid_val = gate_val_df["y_pid"].to_numpy(dtype=np.float32)
    y_ml_val = gate_val_df["y_ml"].to_numpy(dtype=np.float32)
    y_selected_val = gate_val_df["y_selected"].to_numpy(dtype=np.float32)
    alpha_target_val = gate_val_df["alpha_star"].to_numpy(dtype=np.float32)
    y_fused_val = y_pid_val + alpha_pred_val * (y_ml_val - y_pid_val)
    continuous_target_val = (
        gate_val_df["gate_continuous_target"].to_numpy(dtype=np.float32)
        if "gate_continuous_target" in gate_val_df.columns
        else None
    )

    angle_out = out_dir / angle
    angle_out.mkdir(parents=True, exist_ok=True)

    gate_h5 = angle_out / f"{angle}_gate.h5"
    gate_scaler_json = angle_out / f"{angle}_gate_scaler.json"
    diagnostics_json: Path | None = None

    gate_model.save(gate_h5, include_optimizer=False)
    print(f"[OK] {angle}: finished in {time.time() - axis_t0:.1f}s", flush=True)

    gate_scaler_json.write_text(
        json.dumps(
            {
                "feature_names": gate_cols,
                "normalize_features": normalize_features,
                "z_clip": z_clip,
                "gate_architecture": gate_architecture,
                "gate_output_activation": gate_output_activation,
                "sequence_length": sequence_length,
                "gate_loss": gate_loss,
                "alpha_l2_penalty": alpha_l2_penalty,
                "scaler": gate_scaler,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    if save_diagnostics:
        diagnostics_json = angle_out / f"{angle}_diagnostics.json"
        attack_raw = attack_axis[BASE_FEATURES_18].to_numpy(dtype=np.float32)

        diagnostics = {
            "angle": angle,
            "gate_architecture": gate_architecture,
            "gate_output_activation": gate_output_activation,
            "sequence_length": sequence_length,
            "gate_loss": gate_loss,
            "alpha_l2_penalty": alpha_l2_penalty,
            "normalize_features": normalize_features,
            "z_clip": z_clip,
            "feature_stats": {
                "attack_all_raw": summarize_feature_matrix(attack_raw, BASE_FEATURES_18),
            },
            "targets": {
                "y_pid": summarize_array(attack_axis["y_pid"].to_numpy(dtype=np.float32)),
                "y_ml": summarize_array(attack_axis["y_ml"].to_numpy(dtype=np.float32)),
                "alpha_star": summarize_array(alpha_star),
                "alpha_pred_val": summarize_array(alpha_pred_val),
                "y_fused_val": summarize_array(y_fused_val),
                "y_selected_val": summarize_array(y_selected_val),
            },
            "target_construction": target_report | {
                "min_ml_pid_gap": min_ml_pid_gap,
                "alpha_star_clipped_fraction": float(np.mean((alpha_star <= 0.0) | (alpha_star >= 1.0))),
                "usable_rows_after_filtering": int(len(attack_for_split)),
                "enforce_target_reachability": bool(enforce_target_reachability),
                "target_reachability_margin": float(target_reachability_margin),
                "alpha_target_cap": float(alpha_target_cap),
                "alpha_l2_penalty": float(alpha_l2_penalty),
                "unreachable_rows_after_low_gap": int(unreachable_rows),
                "reachable_fraction_after_low_gap": reachable_fraction,
            },
            "metrics": {
                "mae_alpha_vs_target": mae(alpha_target_val, alpha_pred_val),
                "rmse_alpha_vs_target": rmse(alpha_target_val, alpha_pred_val),
            },
        }
        if continuous_target_val is not None:
            diagnostics["targets"]["continuous_target_val"] = summarize_array(continuous_target_val)
            diagnostics["metrics"].update(
                {
                    "mae_fused_vs_target": mae(continuous_target_val, y_fused_val),
                    "rmse_fused_vs_target": rmse(continuous_target_val, y_fused_val),
                    "mae_pid_vs_target": mae(continuous_target_val, y_pid_val),
                    "rmse_pid_vs_target": rmse(continuous_target_val, y_pid_val),
                    "mae_ml_vs_target": mae(continuous_target_val, y_ml_val),
                    "rmse_ml_vs_target": rmse(continuous_target_val, y_ml_val),
                    "mae_selected_vs_target": mae(continuous_target_val, y_selected_val),
                    "rmse_selected_vs_target": rmse(continuous_target_val, y_selected_val),
                }
            )
        diagnostics_json.write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    gate_json: Path | None = None

    if export_fdeep:
        gate_json = angle_out / f"{angle}_gate_fdeep.json"
        export_fdeep_json(gate_h5, gate_json, convert_script)

    mae_fused_vs_target = None
    rmse_fused_vs_target = None
    mae_pid_vs_target = None
    rmse_pid_vs_target = None
    mae_ml_vs_target = None
    rmse_ml_vs_target = None
    if continuous_target_val is not None:
        mae_fused_vs_target = mae(continuous_target_val, y_fused_val)
        rmse_fused_vs_target = rmse(continuous_target_val, y_fused_val)
        mae_pid_vs_target = mae(continuous_target_val, y_pid_val)
        rmse_pid_vs_target = rmse(continuous_target_val, y_pid_val)
        mae_ml_vs_target = mae(continuous_target_val, y_ml_val)
        rmse_ml_vs_target = rmse(continuous_target_val, y_ml_val)

    return AxisArtifacts(
        angle=angle,
        gate_h5=str(gate_h5),
        gate_scaler_json=str(gate_scaler_json),
        diagnostics_json=str(diagnostics_json) if diagnostics_json else None,
        gate_json=str(gate_json) if gate_json else None,
        gate_train_rows=int(train_window_count),
        gate_val_rows=int(val_window_count),
        low_gap_rows=int(target_report["low_gap_rows"]),
        unreachable_rows=int(unreachable_rows),
        reachable_fraction=reachable_fraction,
        gate_target_column=str(target_report["column"]),
        gate_target_kind=str(target_report["kind"]),
        gate_architecture=gate_architecture,
        sequence_length=sequence_length if gate_architecture == "lstm" else 1,
        mae_alpha_vs_target=mae(alpha_target_val, alpha_pred_val),
        rmse_alpha_vs_target=rmse(alpha_target_val, alpha_pred_val),
        mae_fused_vs_target=mae_fused_vs_target,
        rmse_fused_vs_target=rmse_fused_vs_target,
        mae_pid_vs_target=mae_pid_vs_target,
        rmse_pid_vs_target=rmse_pid_vs_target,
        mae_ml_vs_target=mae_ml_vs_target,
        rmse_ml_vs_target=rmse_ml_vs_target,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train PID-Piper gate-NN models")
    parser.add_argument(
        "--clean-long",
        default=IDE_RUN_CONFIG["clean_long"],
        help="Optional path to clean long CSV. Kept for pipeline compatibility; gate-only training does not train reference models.",
    )
    parser.add_argument(
        "--attack",
        default=IDE_RUN_CONFIG["attack"],
        help="Path to attack processed CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=IDE_RUN_CONFIG["output_dir"],
        help="Directory to store trained models and summary",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        default=IDE_RUN_CONFIG["angles"],
        choices=VALID_ANGLES,
        help="Angles to train",
    )
    parser.add_argument(
        "--epochs-ref",
        type=int,
        default=None,
        help="Ignored compatibility option; reference models are no longer trained.",
    )
    parser.add_argument("--epochs-gate", type=int, default=IDE_RUN_CONFIG["epochs_gate"])
    parser.add_argument("--batch-size", type=int, default=IDE_RUN_CONFIG["batch_size"])
    parser.add_argument("--val-fraction", type=float, default=IDE_RUN_CONFIG["val_fraction"])
    parser.add_argument(
        "--lr-ref",
        type=float,
        default=None,
        help="Ignored compatibility option; reference models are no longer trained.",
    )
    parser.add_argument("--lr-gate", type=float, default=IDE_RUN_CONFIG["lr_gate"])
    parser.add_argument("--seed", type=int, default=IDE_RUN_CONFIG["seed"])
    parser.add_argument(
        "--normalize-features",
        type=str_to_bool,
        default=IDE_RUN_CONFIG["normalize_features"],
        help="Whether to apply per-angle standardization before training/inference [true/false]",
    )
    parser.add_argument(
        "--z-clip",
        type=float,
        default=IDE_RUN_CONFIG["z_clip"],
        help="Clip z-score after normalization (<=0 disables clipping)",
    )
    parser.add_argument(
        "--save-diagnostics",
        type=str_to_bool,
        default=IDE_RUN_CONFIG["save_diagnostics"],
        help="Write per-angle diagnostics json [true/false]",
    )
    parser.add_argument(
        "--export-fdeep",
        action="store_true",
        help="Export Keras models to frugally-deep JSON using convert_model.py",
    )
    parser.add_argument(
        "--convert-script",
        default=IDE_RUN_CONFIG["convert_script"],
        help="Path to frugally-deep convert_model.py",
    )
    parser.add_argument(
        "--runtime-bundle-dir",
        default=IDE_RUN_CONFIG["runtime_bundle_dir"],
        help="Subdirectory under output-dir for deployment manifest, runtime model copies, and C++ scaler snippet",
    )
    parser.add_argument(
        "--gate-target-column",
        default=IDE_RUN_CONFIG["gate_target_column"],
        help=(
            "Gate target source. Use auto, attack_label, recovery_mode, an alpha target column, "
            "or y_target/y_true/y_ref for continuous alpha construction."
        ),
    )
    parser.add_argument(
        "--gate-architecture",
        choices=["lstm", "mlp"],
        default=IDE_RUN_CONFIG["gate_architecture"],
        help="Gate model architecture. lstm uses sequence windows; mlp uses single-frame input.",
    )
    parser.add_argument(
        "--gate-output-activation",
        choices=["sigmoid", "hard_sigmoid"],
        default=IDE_RUN_CONFIG["gate_output_activation"],
        help="Final gate activation. hard_sigmoid can saturate exactly at 0/1 for conservative PID fallback.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=IDE_RUN_CONFIG["sequence_length"],
        help="Number of frames per LSTM gate window.",
    )
    parser.add_argument(
        "--min-ml-pid-gap",
        type=float,
        default=IDE_RUN_CONFIG["min_ml_pid_gap"],
        help="Minimum abs(y_ml - y_pid) used for direct alpha target division",
    )
    parser.add_argument(
        "--enforce-target-reachability",
        type=str_to_bool,
        default=IDE_RUN_CONFIG["enforce_target_reachability"],
        help="When gate target is continuous, keep only rows where y_target lies between y_pid and y_ml [true/false].",
    )
    parser.add_argument(
        "--target-reachability-margin",
        type=float,
        default=IDE_RUN_CONFIG["target_reachability_margin"],
        help="Reachability margin added to [min(y_pid,y_ml), max(y_pid,y_ml)] for continuous target filtering.",
    )
    parser.add_argument(
        "--alpha-target-cap",
        type=float,
        default=IDE_RUN_CONFIG["alpha_target_cap"],
        help="Cap alpha supervision target into [0, alpha_target_cap] to reduce aggressive ML blending.",
    )
    parser.add_argument(
        "--gate-loss",
        choices=["alpha_huber", "fused_huber"],
        default=IDE_RUN_CONFIG["gate_loss"],
        help="alpha_huber fits alpha_star; fused_huber directly minimizes y_fused vs y_target.",
    )
    parser.add_argument(
        "--alpha-l2-penalty",
        type=float,
        default=IDE_RUN_CONFIG["alpha_l2_penalty"],
        help="Extra fused_huber penalty on alpha^2. Use >0 to prefer PID unless ML blending clearly helps.",
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        default=IDE_RUN_CONFIG["fit_verbose"],
        choices=[0, 1, 2],
        help="Keras fit verbosity. Use 1 for epoch progress.",
    )

    parser.set_defaults(export_fdeep=IDE_RUN_CONFIG["export_fdeep"])

    no_cli_args = len(sys.argv) == 1
    if USE_IDE_CONFIG_WHEN_NO_ARGS and no_cli_args:
        args = build_args_for_ide()
        print("[INFO] No CLI args detected; using IDE_RUN_CONFIG.")
    else:
        args = parser.parse_args()

    set_seed(args.seed)

    clean_long_path = resolve_path(args.clean_long, project_root) if args.clean_long else None
    auto_selected_attack = False
    if args.attack:
        attack_path = resolve_path(args.attack, project_root)
    else:
        attack_path = discover_latest_attack_csv(project_root)
        auto_selected_attack = attack_path is not None
        if attack_path is not None:
            print(f"[INFO] Auto-selected attack CSV: {attack_path}")

    if attack_path is None:
        raise FileNotFoundError(
            "attack file not found. Provide --attack or run pipeline first to generate gate training csv."
        )

    if no_cli_args and auto_selected_attack and not Path(args.output_dir).is_absolute():
        out_dir = (attack_path.parent / args.output_dir).resolve()
        print(f"[INFO] Auto-selected output dir: {out_dir}")
    else:
        out_dir = resolve_path(args.output_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean_long_path is not None and not clean_long_path.is_file():
        raise FileNotFoundError(f"clean-long file not found: {clean_long_path}")
    if not attack_path.is_file():
        raise FileNotFoundError(f"attack file not found: {attack_path}")

    convert_script = resolve_path(args.convert_script, project_root)
    if args.export_fdeep and not convert_script.is_file():
        raise FileNotFoundError(f"convert_model.py not found: {convert_script}")

    attack, semantic_report = read_and_prepare(
        clean_long_path,
        attack_path,
        args.gate_target_column,
    )

    artifacts: List[AxisArtifacts] = []
    for angle in args.angles:
        axis_artifacts = train_for_angle(
            angle=angle,
            attack=attack,
            out_dir=out_dir,
            epochs_gate=args.epochs_gate,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            lr_gate=args.lr_gate,
            normalize_features=args.normalize_features,
            z_clip=args.z_clip,
            save_diagnostics=args.save_diagnostics,
            export_fdeep=args.export_fdeep,
            convert_script=convert_script,
            gate_target_column=args.gate_target_column,
            gate_architecture=args.gate_architecture,
            gate_output_activation=args.gate_output_activation,
            sequence_length=args.sequence_length,
            min_ml_pid_gap=args.min_ml_pid_gap,
            enforce_target_reachability=args.enforce_target_reachability,
            target_reachability_margin=args.target_reachability_margin,
            alpha_target_cap=args.alpha_target_cap,
            gate_loss=args.gate_loss,
            alpha_l2_penalty=args.alpha_l2_penalty,
            fit_verbose=args.fit_verbose,
        )
        artifacts.append(axis_artifacts)

    runtime_bundle = write_runtime_bundle(
        artifacts=artifacts,
        out_dir=out_dir,
        runtime_bundle_dir=args.runtime_bundle_dir,
        gate_architecture=args.gate_architecture,
        gate_output_activation=args.gate_output_activation,
        sequence_length=args.sequence_length,
    )

    summary = {
        "settings": {
            "model_type": "gate_only",
            "clean_long": str(clean_long_path) if clean_long_path is not None else None,
            "attack": str(attack_path),
            "output_dir": str(out_dir),
            "angles": args.angles,
            "epochs_gate": args.epochs_gate,
            "batch_size": args.batch_size,
            "val_fraction": args.val_fraction,
            "lr_gate": args.lr_gate,
            "seed": args.seed,
            "normalize_features": args.normalize_features,
            "z_clip": args.z_clip,
            "save_diagnostics": args.save_diagnostics,
            "export_fdeep": args.export_fdeep,
            "convert_script": str(convert_script),
            "runtime_bundle_dir": args.runtime_bundle_dir,
            "gate_target_column": args.gate_target_column,
            "gate_architecture": args.gate_architecture,
            "gate_output_activation": args.gate_output_activation,
            "sequence_length": args.sequence_length,
            "min_ml_pid_gap": args.min_ml_pid_gap,
            "enforce_target_reachability": args.enforce_target_reachability,
            "target_reachability_margin": args.target_reachability_margin,
            "alpha_target_cap": args.alpha_target_cap,
            "gate_loss": args.gate_loss,
            "alpha_l2_penalty": args.alpha_l2_penalty,
            "fit_verbose": args.fit_verbose,
            "base_features_18": BASE_FEATURES_18,
            "gate_extra_features": GATE_EXTRA_FEATURES,
            "gate_features_22": GATE_FEATURES_22,
            "fusion_formula": "y_fused = y_pid + alpha * (y_ml - y_pid)",
            "data_preparation": semantic_report,
        },
        "artifacts": [a.__dict__ for a in artifacts],
        "runtime_bundle": runtime_bundle,
    }

    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("[OK] Training completed")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Runtime bundle: {runtime_bundle['bundle_dir']}")
    for a in artifacts:
        print(
            f"[OK] {a.angle}: gate target={a.gate_target_column}/{a.gate_target_kind}, "
            f"arch={a.gate_architecture}, alpha MAE={a.mae_alpha_vs_target:.6f}"
        )


if __name__ == "__main__":
    # Keep TensorFlow logs quieter in terminal output.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
