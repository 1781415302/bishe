#!/usr/bin/env python3
"""Prepare paired clean/attack runs for continuous gate-NN training.

The input manifest lists repeated clean/attack runs of the same mission:
    mission_id,clean_csv,attack_csv

For each pair this script aligns attack rows to clean ground-truth targets by
relative normalized time within each angle axis, then writes one attack-domain
training table with continuous y_target and alpha_star.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

VALID_ANGLES = ["roll", "pitch", "yaw"]
GT_COLUMNS = {"roll": "gt_roll", "pitch": "gt_pitch", "yaw": "gt_yaw"}

REQUIRED_MANIFEST = ["mission_id", "clean_csv", "attack_csv"]
REQUIRED_CLEAN = ["timestamp", "gt_roll", "gt_pitch", "gt_yaw"]
REQUIRED_ATTACK = ["timestamp", "angle_type", "y_pid", "y_ml"] + BASE_FEATURES_18


def resolve_input_path(value: str, manifest_dir: Path) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def infer_pos_unit(values: np.ndarray) -> str:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return "unknown"
    med = float(np.median(finite))
    return "m" if med <= 500.0 else "cm"


def maybe_extract_clean_from_bin(
    clean_path: Path,
    output_dir: Path,
    pair_id: str,
    python_exe: str,
    bin_to_wide_script: Optional[str],
    freq: float,
    semantic_mode: str,
    gt_source: Optional[str],
) -> Path:
    if clean_path.suffix.lower() != ".bin":
        return clean_path
    if not bin_to_wide_script:
        raise ValueError(
            f"clean path is BIN for {pair_id}, but --bin-to-wide-script is not provided"
        )
    script_path = Path(bin_to_wide_script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"bin_to_wide script not found: {script_path}")
    clean_cache_dir = output_dir / "_clean_wide_cache"
    clean_cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = clean_cache_dir / f"{pair_id}_clean_wide.csv"
    cmd = [
        python_exe,
        str(script_path),
        str(clean_path),
        "-o",
        str(out_csv),
        "--freq",
        str(freq),
        "--semantic-mode",
        semantic_mode,
    ]
    if gt_source:
        cmd.extend(["--gt-source", gt_source])
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bin_to_wide failed for {clean_path} with code {proc.returncode}"
        )
    if not out_csv.is_file():
        raise FileNotFoundError(f"bin_to_wide output missing: {out_csv}")
    return out_csv


def ensure_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


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


def add_normalized_time(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_numeric(out[timestamp_col], errors="coerce")
    t_min = float(ts.min())
    t_max = float(ts.max())
    denom = t_max - t_min
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("timestamp range must be positive for normalized-time alignment")
    out["t_norm"] = ((ts - t_min) / denom).astype(np.float64)
    return out


def build_clean_long(clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ensure_columns(clean, REQUIRED_CLEAN, "clean csv")
    clean = clean.copy()
    clean.columns = [c.strip() for c in clean.columns]
    clean["timestamp"] = pd.to_numeric(clean["timestamp"], errors="coerce")
    for col in GT_COLUMNS.values():
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=REQUIRED_CLEAN).sort_values("timestamp").reset_index(drop=True)
    clean = add_normalized_time(clean)

    reports: Dict[str, object] = {}
    frames: List[pd.DataFrame] = []
    for angle, gt_col in GT_COLUMNS.items():
        raw = clean[gt_col].to_numpy(dtype=np.float64)
        converted, unit = angle_values_to_rad(raw, wrap_yaw=(angle == "yaw"))
        axis = clean[["timestamp", "t_norm"]].copy()
        axis["angle_type"] = angle
        axis["clean_t_norm"] = axis["t_norm"]
        axis["y_target"] = converted
        frames.append(axis)
        reports[angle] = {"source_unit": unit, "rows": int(len(axis))}

    clean_long = pd.concat(frames, ignore_index=True)
    clean_long = clean_long.sort_values(["angle_type", "clean_t_norm"]).reset_index(drop=True)
    return clean_long, reports


def trim_clean_by_attack_start(clean: pd.DataFrame, attack: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {
        "applied": False,
        "reason": "missing_pos_z",
        "rows_before": int(len(clean)),
        "rows_after": int(len(clean)),
    }
    if "pos_z" not in clean.columns or "pos_z" not in attack.columns:
        return clean, report

    clean_z = pd.to_numeric(clean["pos_z"], errors="coerce").to_numpy(dtype=np.float64)
    attack_z = pd.to_numeric(attack["pos_z"], errors="coerce").to_numpy(dtype=np.float64)
    clean_finite = np.isfinite(clean_z)
    attack_finite = np.isfinite(attack_z)
    if not bool(clean_finite.any()) or not bool(attack_finite.any()):
        report["reason"] = "non_finite_pos_z"
        return clean, report

    attack_start = float(np.median(attack_z[attack_finite][: min(300, int(attack_finite.sum()))]))
    clean_unit = infer_pos_unit(clean_z)
    attack_unit = infer_pos_unit(attack_z)
    if clean_unit == "m" and attack_unit == "cm":
        attack_start_adj = attack_start / 100.0
        tol = 0.5
    elif clean_unit == "cm" and attack_unit == "m":
        attack_start_adj = attack_start * 100.0
        tol = 50.0
    else:
        attack_start_adj = attack_start
        tol = 0.5 if clean_unit == "m" else 50.0

    clean_series = pd.to_numeric(clean["pos_z"], errors="coerce")
    crossing = clean_series[clean_series >= (attack_start_adj - tol)]
    if not crossing.empty:
        anchor_idx = int(crossing.index[0])
        method = "first_crossing"
    else:
        diffs = np.abs(clean_z - attack_start_adj)
        diffs[~np.isfinite(diffs)] = np.inf
        anchor_idx = int(np.argmin(diffs))
        method = "nearest"

    trimmed = clean.iloc[anchor_idx:].reset_index(drop=True)
    report.update(
        {
            "applied": True,
            "reason": "ok",
            "method": method,
            "anchor_idx": anchor_idx,
            "attack_start_pos_z_raw": attack_start,
            "attack_start_pos_z_in_clean_unit": float(attack_start_adj),
            "clean_pos_unit": clean_unit,
            "attack_pos_unit": attack_unit,
            "rows_after": int(len(trimmed)),
        }
    )
    return trimmed, report


def trim_clean_to_flight_phase(clean: pd.DataFrame, start_report: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {
        "applied": False,
        "reason": "missing_pos_z_or_timestamp",
        "rows_before": int(len(clean)),
        "rows_after": int(len(clean)),
    }
    if "pos_z" not in clean.columns or "timestamp" not in clean.columns:
        return clean, report

    z = pd.to_numeric(clean["pos_z"], errors="coerce").to_numpy(dtype=np.float64)
    ts = pd.to_numeric(clean["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(ts)
    if not bool(finite.any()):
        report["reason"] = "non_finite_pos_z_or_timestamp"
        return clean, report

    clean_unit = str(start_report.get("clean_pos_unit") or infer_pos_unit(z))
    start_alt = start_report.get("attack_start_pos_z_in_clean_unit")
    if start_alt is None or not math.isfinite(float(start_alt)):
        finite_z = z[finite]
        start_alt = float(np.percentile(finite_z, 90))
    else:
        start_alt = float(start_alt)

    tol = 0.5 if clean_unit == "m" else 50.0
    threshold = start_alt - tol
    in_flight = finite & (z >= threshold)
    valid_idx = np.flatnonzero(in_flight)
    if valid_idx.size == 0:
        report["reason"] = "no_rows_above_flight_threshold"
        report.update(
            {
                "clean_pos_unit": clean_unit,
                "flight_altitude_threshold": float(threshold),
                "flight_start_pos_z": float(start_alt),
            }
        )
        return clean, report

    start_idx = int(valid_idx[0])
    end_idx = int(valid_idx[-1])
    if end_idx <= start_idx:
        report["reason"] = "non_positive_flight_duration"
        return clean, report

    trimmed = clean.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    t_start = float(ts[start_idx])
    t_end = float(ts[end_idx])
    duration = t_end - t_start
    if not math.isfinite(duration) or duration <= 0.0:
        report["reason"] = "non_positive_flight_duration"
        return clean, report

    report.update(
        {
            "applied": True,
            "reason": "ok",
            "method": "altitude_above_attack_start_threshold",
            "clean_pos_unit": clean_unit,
            "flight_start_pos_z": float(start_alt),
            "flight_altitude_threshold": float(threshold),
            "flight_start_idx": start_idx,
            "flight_end_idx": end_idx,
            "flight_start_timestamp": t_start,
            "flight_end_timestamp": t_end,
            "flight_duration": float(duration),
            "rows_after": int(len(trimmed)),
            "dropped_before_flight": int(start_idx),
            "dropped_after_flight": int(len(clean) - end_idx - 1),
        }
    )
    return trimmed, report


def trim_attack_by_clean_flight_duration(
    attack: pd.DataFrame,
    clean_flight_report: Dict[str, object],
    max_clean_flight_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {
        "applied": False,
        "reason": "disabled",
        "rows_before": int(len(attack)),
        "rows_after": int(len(attack)),
        "max_clean_flight_ratio": float(max_clean_flight_ratio),
    }
    if max_clean_flight_ratio <= 0.0:
        return attack, report

    duration = clean_flight_report.get("flight_duration")
    if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0.0:
        report["reason"] = "missing_clean_flight_duration"
        return attack, report
    if "timestamp" not in attack.columns:
        report["reason"] = "missing_attack_timestamp"
        return attack, report

    ts = pd.to_numeric(attack["timestamp"], errors="coerce")
    finite = ts[np.isfinite(ts)]
    if finite.empty:
        report["reason"] = "non_finite_attack_timestamp"
        return attack, report

    attack_start = float(finite.min())
    attack_end_before = float(finite.max())
    allowed_duration = float(duration) * float(max_clean_flight_ratio)
    cutoff = attack_start + allowed_duration
    keep_mask = ts <= cutoff
    trimmed = attack[keep_mask].reset_index(drop=True)
    attack_end_after = float(pd.to_numeric(trimmed["timestamp"], errors="coerce").max()) if not trimmed.empty else None

    report.update(
        {
            "applied": True,
            "reason": "ok",
            "clean_flight_duration": float(duration),
            "allowed_attack_duration": float(allowed_duration),
            "attack_start_timestamp": attack_start,
            "attack_end_timestamp_before": attack_end_before,
            "attack_end_timestamp_after": attack_end_after,
            "attack_duration_before": float(attack_end_before - attack_start),
            "attack_duration_after": (
                float(attack_end_after - attack_start) if attack_end_after is not None else None
            ),
            "cutoff_timestamp": float(cutoff),
            "rows_after": int(len(trimmed)),
            "dropped_tail_rows": int(len(attack) - len(trimmed)),
        }
    )
    return trimmed, report


def prepare_attack(attack: pd.DataFrame) -> pd.DataFrame:
    attack = attack.copy()
    attack.columns = [c.strip() for c in attack.columns]
    ensure_columns(attack, REQUIRED_ATTACK, "attack csv")
    attack["angle_type"] = attack["angle_type"].astype(str).str.strip().str.lower()
    attack = attack[attack["angle_type"].isin(VALID_ANGLES)].copy()

    numeric_cols = BASE_FEATURES_18 + [
        "timestamp",
        "y_pid",
        "y_ml",
        "residual",
        "y_selected",
        "y_fused",
        "alpha",
        "attack_label",
        "recovery_mode",
        "strategy_mode",
    ]
    for col in numeric_cols:
        if col in attack.columns:
            attack[col] = pd.to_numeric(attack[col], errors="coerce")

    if "y_selected" not in attack.columns:
        attack["y_selected"] = attack["y_ml"]
    if "residual" not in attack.columns:
        attack["residual"] = (attack["y_ml"] - attack["y_pid"]).abs()

    attack = attack.dropna(subset=REQUIRED_ATTACK + ["y_selected", "residual"]).copy()
    attack = attack.sort_values(["timestamp", "angle_type"]).reset_index(drop=True)
    attack = add_normalized_time(attack)

    for col in ["y_pid", "y_ml", "y_selected", "y_fused"]:
        if col not in attack.columns:
            continue
        for angle in VALID_ANGLES:
            mask = attack["angle_type"] == angle
            if not bool(mask.any()):
                continue
            raw = attack.loc[mask, col].to_numpy(dtype=np.float64)
            converted, _unit = angle_values_to_rad(raw, wrap_yaw=(angle == "yaw"))
            attack.loc[mask, col] = converted

    attack["residual"] = (attack["y_ml"] - attack["y_pid"]).abs()
    attack["ml_pid_gap_abs"] = attack["residual"]
    return attack


def align_pair(
    mission_id: str,
    pair_id: str,
    clean_path: Path,
    attack_path: Path,
    output_dir: Path,
    max_align_error_norm: float,
    min_ml_pid_gap: float,
    python_exe: str,
    bin_to_wide_script: Optional[str],
    freq: float,
    semantic_mode: str,
    gt_source: Optional[str],
    attack_clean_flight_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    clean_materialized_path = maybe_extract_clean_from_bin(
        clean_path=clean_path,
        output_dir=output_dir,
        pair_id=pair_id,
        python_exe=python_exe,
        bin_to_wide_script=bin_to_wide_script,
        freq=freq,
        semantic_mode=semantic_mode,
        gt_source=gt_source,
    )
    clean = pd.read_csv(clean_materialized_path)
    attack = pd.read_csv(attack_path)
    clean, trim_report = trim_clean_by_attack_start(clean, attack)
    clean, clean_flight_report = trim_clean_to_flight_phase(clean, trim_report)
    attack, attack_tail_report = trim_attack_by_clean_flight_duration(
        attack,
        clean_flight_report,
        attack_clean_flight_ratio,
    )

    clean_long, clean_report = build_clean_long(clean)
    attack_prepared = prepare_attack(attack)
    attack_prepared["mission_id"] = mission_id
    attack_prepared["pair_id"] = pair_id

    aligned_frames: List[pd.DataFrame] = []
    axis_reports: Dict[str, object] = {}
    for angle in VALID_ANGLES:
        attack_axis = attack_prepared[attack_prepared["angle_type"] == angle].copy()
        clean_axis = clean_long[clean_long["angle_type"] == angle].copy()
        if attack_axis.empty or clean_axis.empty:
            axis_reports[angle] = {
                "attack_rows": int(len(attack_axis)),
                "clean_rows": int(len(clean_axis)),
                "kept_rows": 0,
                "dropped_by_alignment": int(len(attack_axis)),
            }
            continue

        attack_axis = attack_axis.sort_values("t_norm").reset_index(drop=True)
        clean_axis = clean_axis.sort_values("clean_t_norm").reset_index(drop=True)
        aligned = pd.merge_asof(
            attack_axis,
            clean_axis[["clean_t_norm", "y_target"]],
            left_on="t_norm",
            right_on="clean_t_norm",
            direction="nearest",
        )
        aligned["align_error_norm"] = (aligned["t_norm"] - aligned["clean_t_norm"]).abs()
        before_align = len(aligned)
        if max_align_error_norm > 0:
            aligned = aligned[aligned["align_error_norm"] <= max_align_error_norm].copy()
        after_align = len(aligned)

        gap = aligned["y_ml"] - aligned["y_pid"]
        stable_gap = gap.abs() >= max(float(min_ml_pid_gap), 1.0e-12)
        aligned = aligned[stable_gap].copy()
        gap = aligned["y_ml"] - aligned["y_pid"]
        aligned["alpha_star"] = ((aligned["y_target"] - aligned["y_pid"]) / gap).clip(0.0, 1.0)

        axis_reports[angle] = {
            "attack_rows": int(len(attack_axis)),
            "clean_rows": int(len(clean_axis)),
            "kept_rows": int(len(aligned)),
            "dropped_by_alignment": int(before_align - after_align),
            "dropped_by_low_gap": int((~stable_gap).sum()),
            "align_error_norm_mean": float(aligned["align_error_norm"].mean()) if not aligned.empty else None,
            "align_error_norm_max": float(aligned["align_error_norm"].max()) if not aligned.empty else None,
        }
        aligned_frames.append(aligned)

    if aligned_frames:
        paired = pd.concat(aligned_frames, ignore_index=True)
    else:
        paired = pd.DataFrame()

    report = {
        "mission_id": mission_id,
        "pair_id": pair_id,
        "clean_csv": str(clean_path),
        "clean_materialized_csv": str(clean_materialized_path),
        "attack_csv": str(attack_path),
        "clean_trim_by_attack_start": trim_report,
        "clean_flight_phase": clean_flight_report,
        "attack_tail_trim_by_clean_flight": attack_tail_report,
        "clean_target_units": clean_report,
        "axes": axis_reports,
        "output_rows": int(len(paired)),
    }
    return paired, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired clean/attack data for continuous LSTM gate training")
    parser.add_argument("--pairs-manifest", required=True, help="CSV manifest with mission_id,clean_csv,attack_csv")
    parser.add_argument("--output-dir", default="analysis_one_click", help="Directory for prepared outputs")
    parser.add_argument("--output-csv", default="gate_training_attack_with_target.csv")
    parser.add_argument("--report-name", default="pairing_report.json")
    parser.add_argument("--max-align-error-norm", type=float, default=0.02)
    parser.add_argument("--min-ml-pid-gap", type=float, default=1.0e-4)
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used for optional BIN extraction")
    parser.add_argument("--bin-to-wide-script", default=None, help="Path to bin_to_wide.py when manifest clean files are BIN")
    parser.add_argument("--freq", type=float, default=50.0, help="Target frequency for BIN extraction")
    parser.add_argument("--semantic-mode", choices=["attack-compatible", "legacy"], default="attack-compatible")
    parser.add_argument("--gt-source", default=None, help="Optional gt source priority list for bin_to_wide.py")
    parser.add_argument(
        "--attack-clean-flight-ratio",
        type=float,
        default=1.05,
        help="Trim attack rows after clean flight-phase duration * this ratio. Use <=0 to disable.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.pairs_manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    manifest.columns = [c.strip() for c in manifest.columns]
    ensure_columns(manifest, REQUIRED_MANIFEST, "pairs manifest")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    reports: List[Dict[str, object]] = []
    for idx, row in manifest.iterrows():
        mission_id = str(row["mission_id"]).strip()
        pair_id = f"{idx + 1:03d}_{mission_id}"
        clean_path = resolve_input_path(row["clean_csv"], manifest_dir)
        attack_path = resolve_input_path(row["attack_csv"], manifest_dir)
        if not clean_path.is_file():
            raise FileNotFoundError(f"clean_csv not found for {mission_id}: {clean_path}")
        if not attack_path.is_file():
            raise FileNotFoundError(f"attack_csv not found for {mission_id}: {attack_path}")

        paired, report = align_pair(
            mission_id=mission_id,
            pair_id=pair_id,
            clean_path=clean_path,
            attack_path=attack_path,
            output_dir=out_dir,
            max_align_error_norm=args.max_align_error_norm,
            min_ml_pid_gap=args.min_ml_pid_gap,
            python_exe=args.python_exe,
            bin_to_wide_script=args.bin_to_wide_script,
            freq=args.freq,
            semantic_mode=args.semantic_mode,
            gt_source=args.gt_source,
            attack_clean_flight_ratio=args.attack_clean_flight_ratio,
        )
        reports.append(report)
        if not paired.empty:
            all_frames.append(paired)

    if not all_frames:
        raise RuntimeError("No paired rows produced from manifest")

    combined = pd.concat(all_frames, ignore_index=True)
    preferred = [
        "mission_id",
        "pair_id",
        "timestamp",
        "t_norm",
        "clean_t_norm",
        "align_error_norm",
        "angle_type",
    ] + BASE_FEATURES_18 + [
        "y_pid",
        "y_ml",
        "residual",
        "ml_pid_gap_abs",
        "y_target",
        "alpha_star",
        "attack_label",
        "recovery_mode",
        "strategy_mode",
        "alpha",
        "y_fused",
        "y_selected",
    ]
    ordered = [col for col in preferred if col in combined.columns]
    ordered.extend(col for col in combined.columns if col not in ordered)
    combined = combined[ordered].sort_values(["mission_id", "pair_id", "angle_type", "t_norm"]).reset_index(drop=True)

    output_csv = out_dir / args.output_csv
    combined.to_csv(output_csv, index=False)

    report = {
        "pairs_manifest": str(manifest_path),
        "output_csv": str(output_csv),
        "settings": {
            "alignment_basis": "normalized_time",
            "max_align_error_norm": args.max_align_error_norm,
            "min_ml_pid_gap": args.min_ml_pid_gap,
            "python_exe": args.python_exe,
            "bin_to_wide_script": args.bin_to_wide_script,
            "freq": args.freq,
            "semantic_mode": args.semantic_mode,
            "gt_source": args.gt_source,
            "attack_clean_flight_ratio": args.attack_clean_flight_ratio,
        },
        "pairs": reports,
        "output_rows": int(len(combined)),
        "rows_by_mission_axis": {
            f"{mission}_{angle}": int(n)
            for (mission, angle), n in combined.groupby(["mission_id", "angle_type"]).size().items()
        },
        "rows_by_pair_axis": {
            f"{mission}_{pair}_{angle}": int(n)
            for (mission, pair, angle), n in combined.groupby(["mission_id", "pair_id", "angle_type"]).size().items()
        },
    }
    report_path = out_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[OK] paired training csv: {output_csv} ({len(combined)} rows)")
    print(f"[OK] pairing report: {report_path}")


if __name__ == "__main__":
    main()
