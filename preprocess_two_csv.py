#!/usr/bin/env python3
"""Preprocess two simulator CSV tables in the current directory.

The script looks for:
1) clean+GT table: includes gt_roll/gt_pitch/gt_yaw
2) attack+ML table: includes angle_type/y_pid/y_ml/residual/attack_label/recovery_mode

Outputs:
- <clean_file>.processed.csv
- <attack_file>.processed.csv
- <clean_file>.long.csv (optional, generated from gt_* columns)
- preprocessing_report.json

Default behavior is tuned for time-series training:
- sort by time before filling
- forward-fill only (no backward fill)
- fill attack table per angle_type to avoid cross-angle leakage
- keep duplicate rows unless --drop-exact-duplicates is explicitly set
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pandas. Install it with: pip install pandas"
    ) from exc


CLEAN_REQUIRED = {"gt_roll", "gt_pitch", "gt_yaw"}
ATTACK_REQUIRED = {
    "angle_type",
    "y_pid",
    "y_ml",
    "residual",
    "attack_label",
    "recovery_mode",
}

CLEAN_NUMERIC_COLUMNS = [
    "timestamp",
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
    "gt_roll",
    "gt_pitch",
    "gt_yaw",
]

ATTACK_NUMERIC_COLUMNS = [
    "timestamp",
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
    "navYaw",
    "y_pid",
    "y_ml",
    "residual",
    "y_selected",
    "attack_label",
    "recovery_mode",
]

CLEAN_CATEGORICAL_DEFAULTS = {
    "scenario": "default",
    "label": "normal",
    "firmware": "unknown",
    "params_hash": "unknown",
}

VALID_ANGLE_TYPES = {"roll", "pitch", "yaw"}

CLEAN_MIN_ALT_M_DEFAULT = 48.0


def median_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return None
    return float(values.abs().median())


def infer_clean_position_unit(pos_z_series: pd.Series) -> str:
    med_abs = median_abs(pos_z_series)
    if med_abs is None:
        return "unknown"
    # bin_to_wide.py typically emits meters for LOCAL_POSITION_NED style logs.
    return "m" if med_abs <= 500.0 else "cm"


def convert_angle_to_centideg(
    series: pd.Series, companion: pd.Series | None = None
) -> Tuple[pd.Series, str, float]:
    values = pd.to_numeric(series, errors="coerce")
    med_abs = median_abs(values)
    if med_abs is None:
        return values, "unknown", 1.0

    if med_abs > 720.0:
        return values, "centideg_or_other", 1.0

    if companion is not None:
        comp_med_abs = median_abs(companion)
        if comp_med_abs is not None:
            if comp_med_abs > 720.0:
                return values, "centideg_or_other", 1.0
            if comp_med_abs > (2.0 * math.pi + 0.5):
                factor = 100.0
                return values * factor, "deg", factor

    if med_abs <= (2.0 * math.pi + 0.5):
        factor = 18000.0 / math.pi
        return values * factor, "rad", factor

    if med_abs <= 720.0:
        factor = 100.0
        return values * factor, "deg", factor

    return values, "centideg_or_other", 1.0


def harmonize_clean_units(df: pd.DataFrame, pos_unit: str) -> Dict[str, object]:
    report: Dict[str, object] = {
        "position_unit_detected": pos_unit,
        "applied": {},
    }

    applied: Dict[str, object] = {}

    if pos_unit == "m":
        for col in ["pos_x", "pos_y", "pos_z"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 100.0
        applied["position"] = {"from": "m", "to": "cm", "factor": 100.0}
    elif pos_unit == "cm":
        applied["position"] = {"from": "cm", "to": "cm", "factor": 1.0}
    else:
        applied["position"] = {"from": "unknown", "to": "unknown", "factor": 1.0}

    if "gpsVel" in df.columns:
        gps_med = median_abs(df["gpsVel"])
        if gps_med is not None and gps_med <= 80.0:
            df["gpsVel"] = pd.to_numeric(df["gpsVel"], errors="coerce") * 100.0
            applied["gpsVel"] = {"from": "m/s", "to": "cm/s", "factor": 100.0}
        elif gps_med is not None:
            applied["gpsVel"] = {"from": "cm/s_or_other", "to": "cm/s_or_other", "factor": 1.0}
        else:
            applied["gpsVel"] = {"from": "unknown", "to": "unknown", "factor": 1.0}

    angle_conversion: Dict[str, object] = {}
    if "navRoll" in df.columns and "navPitch" in df.columns:
        roll_converted, roll_detected, roll_factor = convert_angle_to_centideg(
            df["navRoll"], companion=df["navPitch"]
        )
        pitch_converted, pitch_detected, pitch_factor = convert_angle_to_centideg(
            df["navPitch"], companion=df["navRoll"]
        )
        df["navRoll"] = roll_converted
        df["navPitch"] = pitch_converted
        angle_conversion["navRoll"] = {
            "from": roll_detected,
            "to": "centideg" if roll_factor != 1.0 else roll_detected,
            "factor": roll_factor,
        }
        angle_conversion["navPitch"] = {
            "from": pitch_detected,
            "to": "centideg" if pitch_factor != 1.0 else pitch_detected,
            "factor": pitch_factor,
        }
    else:
        for col in ["navRoll", "navPitch"]:
            if col in df.columns:
                converted, detected, factor = convert_angle_to_centideg(df[col])
                df[col] = converted
                angle_conversion[col] = {
                    "from": detected,
                    "to": "centideg" if factor != 1.0 else detected,
                    "factor": factor,
                }

    if angle_conversion:
        applied["angles"] = angle_conversion

    gt_cols = [col for col in ["gt_roll", "gt_pitch", "gt_yaw"] if col in df.columns]
    if gt_cols:
        applied["gt_angles"] = {
            "columns": gt_cols,
            "kept_as": "radians",
        }

    report["applied"] = applied
    return report


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return [h.strip() for h in next(reader)]


def classify_csv(csv_path: Path) -> str:
    header = set(read_header(csv_path))
    if CLEAN_REQUIRED.issubset(header):
        return "clean"
    if ATTACK_REQUIRED.issubset(header):
        return "attack"
    return "unknown"


def pick_one(candidates: List[Path], label: str) -> Path:
    if not candidates:
        raise RuntimeError(f"Could not find {label} CSV in input directory")

    # Prefer original source CSVs over files generated by this script.
    raw_candidates = [
        p for p in candidates if ".processed" not in p.stem and ".long" not in p.stem
    ]
    if raw_candidates:
        candidates = raw_candidates

    # Pick the largest candidate to avoid accidentally selecting test snippets.
    return max(candidates, key=lambda p: p.stat().st_size)


def locate_input_tables(input_dir: Path) -> Tuple[Path, Path, Dict[str, List[Path]]]:
    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if len(csv_files) < 2:
        raise RuntimeError(
            f"Expected at least 2 CSV files in {input_dir}, found {len(csv_files)}"
        )

    grouped: Dict[str, List[Path]] = {"clean": [], "attack": [], "unknown": []}
    for csv_file in csv_files:
        grouped[classify_csv(csv_file)].append(csv_file)

    clean_file = pick_one(grouped["clean"], "clean+GT")
    attack_file = pick_one(grouped["attack"], "attack+ML")
    return clean_file, attack_file, grouped


def convert_numeric(df: pd.DataFrame, columns: List[str]) -> List[str]:
    existing = [c for c in columns if c in df.columns]
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return existing


def missing_counts(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in columns if col in df.columns}


def sort_by_timestamp(df: pd.DataFrame, extra_cols: List[str] | None = None) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    cols = ["timestamp"]
    if extra_cols:
        cols.extend([c for c in extra_cols if c in df.columns])
    return df.sort_values(cols).reset_index(drop=True)


def fill_numeric(
    df: pd.DataFrame,
    columns: List[str],
    fill_strategy: str,
    group_cols: List[str] | None = None,
) -> None:
    if not columns or fill_strategy == "none":
        return

    if group_cols:
        existing_group_cols = [c for c in group_cols if c in df.columns]
    else:
        existing_group_cols = []

    if existing_group_cols:
        if fill_strategy == "ffill":
            df[columns] = df.groupby(existing_group_cols, dropna=False)[columns].transform(
                lambda g: g.ffill()
            )
        elif fill_strategy == "ffill_bfill":
            df[columns] = df.groupby(existing_group_cols, dropna=False)[columns].transform(
                lambda g: g.ffill().bfill()
            )
    else:
        if fill_strategy == "ffill":
            df[columns] = df[columns].ffill()
        elif fill_strategy == "ffill_bfill":
            df[columns] = df[columns].ffill().bfill()


def preprocess_clean(
    clean_path: Path,
    fill_strategy: str,
    drop_exact_duplicates: bool,
    drop_remaining_nan: bool,
    clean_min_alt_m: float,
    harmonize_units: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(clean_path)
    df.columns = [c.strip() for c in df.columns]
    df = sort_by_timestamp(df)

    report: Dict[str, object] = {"input_rows": int(len(df))}

    duplicate_rows = int(df.duplicated().sum())
    report["duplicate_rows_detected"] = duplicate_rows
    dropped_duplicates = 0
    if drop_exact_duplicates and duplicate_rows:
        before_drop = len(df)
        df = df.drop_duplicates().copy()
        dropped_duplicates = int(before_drop - len(df))
    report["duplicate_rows_dropped"] = dropped_duplicates

    numeric_cols = convert_numeric(df, CLEAN_NUMERIC_COLUMNS)

    if "pos_z" in df.columns:
        pos_unit = infer_clean_position_unit(df["pos_z"])
        threshold_raw = clean_min_alt_m if pos_unit != "cm" else clean_min_alt_m * 100.0
        before_alt_filter = len(df)
        pos_z_numeric = pd.to_numeric(df["pos_z"], errors="coerce")
        # NED-style logs may represent altitude as negative-up/down-signed Z.
        # Use absolute magnitude so "height > X" semantics remain stable.
        filtered_df = df[pos_z_numeric.abs() > threshold_raw].copy()
        dropped_rows = int(before_alt_filter - len(filtered_df))
        if before_alt_filter > 0 and len(filtered_df) == 0:
            report["altitude_filter"] = {
                "enabled": False,
                "clean_min_alt_m": clean_min_alt_m,
                "pos_z_unit_detected": pos_unit,
                "rule": "abs(pos_z) > threshold",
                "raw_threshold_used": threshold_raw,
                "dropped_rows": dropped_rows,
                "fallback": "disabled_filter_to_avoid_empty_output",
            }
        else:
            df = filtered_df
            report["altitude_filter"] = {
                "enabled": True,
                "clean_min_alt_m": clean_min_alt_m,
                "pos_z_unit_detected": pos_unit,
                "rule": "abs(pos_z) > threshold",
                "raw_threshold_used": threshold_raw,
                "dropped_rows": dropped_rows,
            }
    else:
        pos_unit = "unknown"
        report["altitude_filter"] = {
            "enabled": False,
            "reason": "pos_z column missing",
            "clean_min_alt_m": clean_min_alt_m,
        }

    if harmonize_units:
        report["unit_harmonization"] = harmonize_clean_units(df, pos_unit)
    else:
        report["unit_harmonization"] = {
            "enabled": False,
            "position_unit_detected": pos_unit,
        }

    report["missing_before_fill"] = missing_counts(df, numeric_cols)

    fill_numeric(df, numeric_cols, fill_strategy)
    report["missing_after_fill"] = missing_counts(df, numeric_cols)

    for col, default_value in CLEAN_CATEGORICAL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_value).astype(str).str.strip()
            df.loc[df[col] == "", col] = default_value

    if drop_remaining_nan and numeric_cols:
        before_drop = len(df)
        df = df.dropna(subset=numeric_cols).copy()
        report["dropped_rows_remaining_numeric_nan"] = int(before_drop - len(df))
    else:
        report["dropped_rows_remaining_numeric_nan"] = 0

    key_cols = [c for c in ["timestamp", "gt_roll", "gt_pitch", "gt_yaw"] if c in df.columns]
    before_drop = len(df)
    if key_cols:
        df = df.dropna(subset=key_cols).copy()
    report["dropped_rows_on_key_columns"] = int(before_drop - len(df))

    df = sort_by_timestamp(df)

    report["output_rows"] = int(len(df))
    return df, report


def preprocess_attack(
    attack_path: Path,
    fill_strategy: str,
    drop_exact_duplicates: bool,
    drop_remaining_nan: bool,
    binary_label_policy: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(attack_path)
    df.columns = [c.strip() for c in df.columns]

    report: Dict[str, object] = {"input_rows": int(len(df))}

    if "angle_type" in df.columns:
        df["angle_type"] = df["angle_type"].astype(str).str.strip().str.lower()
        invalid_count = int((~df["angle_type"].isin(VALID_ANGLE_TYPES)).sum())
        report["invalid_angle_type_rows"] = invalid_count
        df = df[df["angle_type"].isin(VALID_ANGLE_TYPES)].copy()

    if {"timestamp", "angle_type"}.issubset(df.columns):
        angle_order = {"roll": 0, "pitch": 1, "yaw": 2}
        df["__angle_order"] = df["angle_type"].map(angle_order)
        df = sort_by_timestamp(df, ["__angle_order"])
        df = df.drop(columns=["__angle_order"])
    else:
        df = sort_by_timestamp(df)

    duplicate_rows = int(df.duplicated().sum())
    report["duplicate_rows_detected"] = duplicate_rows
    dropped_duplicates = 0
    if drop_exact_duplicates and duplicate_rows:
        before_drop = len(df)
        df = df.drop_duplicates().copy()
        dropped_duplicates = int(before_drop - len(df))
    report["duplicate_rows_dropped"] = dropped_duplicates

    numeric_cols = convert_numeric(df, ATTACK_NUMERIC_COLUMNS)
    report["missing_before_fill"] = missing_counts(df, numeric_cols)

    fill_groups = ["angle_type"] if "angle_type" in df.columns else None
    fill_numeric(df, numeric_cols, fill_strategy, group_cols=fill_groups)
    report["missing_after_fill"] = missing_counts(df, numeric_cols)

    if drop_remaining_nan and numeric_cols:
        before_drop = len(df)
        df = df.dropna(subset=numeric_cols).copy()
        report["dropped_rows_remaining_numeric_nan"] = int(before_drop - len(df))
    else:
        report["dropped_rows_remaining_numeric_nan"] = 0

    for binary_col in ["attack_label", "recovery_mode"]:
        if binary_col in df.columns:
            raw_series = pd.to_numeric(df[binary_col], errors="coerce")
            invalid_mask = (~raw_series.isin([0, 1])) & (~raw_series.isna())
            invalid_count = int(invalid_mask.sum())
            report[f"{binary_col}_invalid_rows"] = invalid_count

            if binary_label_policy == "clip":
                df[binary_col] = raw_series.fillna(0).astype(int).clip(lower=0, upper=1)
            elif binary_label_policy == "drop-invalid":
                before_drop = len(df)
                keep_mask = raw_series.isin([0, 1])
                df = df[keep_mask].copy()
                df[binary_col] = raw_series[keep_mask].astype(int)
                report[f"{binary_col}_dropped_invalid_rows"] = int(before_drop - len(df))
            else:  # keep
                df[binary_col] = raw_series

    if binary_label_policy == "keep":
        for binary_col in ["attack_label", "recovery_mode"]:
            if binary_col in df.columns:
                unique_vals = sorted([str(v) for v in df[binary_col].dropna().unique().tolist()])
                report[f"{binary_col}_unique_values"] = unique_vals

    key_cols = [
        c
        for c in ["timestamp", "angle_type", "y_pid", "y_ml", "residual", "y_selected"]
        if c in df.columns
    ]
    before_drop = len(df)
    if key_cols:
        df = df.dropna(subset=key_cols).copy()
    report["dropped_rows_on_key_columns"] = int(before_drop - len(df))

    if {"timestamp", "angle_type"}.issubset(df.columns):
        angle_order = {"roll": 0, "pitch": 1, "yaw": 2}
        df["__angle_order"] = df["angle_type"].map(angle_order)
        df = sort_by_timestamp(df, ["__angle_order"])
        df = df.drop(columns=["__angle_order"])
    else:
        df = sort_by_timestamp(df)

    if {"attack_label", "recovery_mode"}.issubset(df.columns):
        combo = df.groupby(["attack_label", "recovery_mode"]).size().sort_index()
        report["attack_recovery_counts"] = {
            f"attack_{int(a)}_recovery_{int(r)}": int(n)
            for (a, r), n in combo.items()
        }

    report["output_rows"] = int(len(df))
    return df, report


def build_clean_long(clean_df: pd.DataFrame) -> pd.DataFrame | None:
    gt_cols = ["gt_roll", "gt_pitch", "gt_yaw"]
    if not all(col in clean_df.columns for col in gt_cols):
        return None

    id_cols = [col for col in clean_df.columns if col not in gt_cols]
    long_df = clean_df.melt(
        id_vars=id_cols,
        value_vars=gt_cols,
        var_name="angle_type",
        value_name="y_true",
    )
    long_df["angle_type"] = long_df["angle_type"].str.replace("gt_", "", regex=False)
    if {"timestamp", "angle_type"}.issubset(long_df.columns):
        angle_order = {"roll": 0, "pitch": 1, "yaw": 2}
        long_df["__angle_order"] = long_df["angle_type"].map(angle_order)
        long_df = long_df.sort_values(["timestamp", "__angle_order"]).drop(columns=["__angle_order"])
        long_df = long_df.reset_index(drop=True)

    preferred_order = [
        "timestamp",
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
        "angle_type",
        "y_true",
        "scenario",
        "label",
        "firmware",
        "params_hash",
    ]
    ordered_cols = [col for col in preferred_order if col in long_df.columns]
    ordered_cols.extend(col for col in long_df.columns if col not in ordered_cols)
    return long_df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess clean+GT and attack+ML CSV files")
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing the two input CSV files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same as input-dir)",
    )
    parser.add_argument(
        "--report-name",
        default="preprocessing_report.json",
        help="Output report file name (default: preprocessing_report.json)",
    )
    parser.add_argument(
        "--fill-strategy",
        choices=["ffill", "ffill_bfill", "none"],
        default="ffill",
        help="Numeric missing-value strategy (default: ffill)",
    )
    parser.add_argument(
        "--drop-exact-duplicates",
        action="store_true",
        help="Drop fully identical rows (default: keep duplicates for time-series fidelity)",
    )
    parser.add_argument(
        "--keep-remaining-nan",
        action="store_true",
        help="Keep rows that still have numeric NaN after fill (default: drop them)",
    )
    parser.add_argument(
        "--binary-label-policy",
        choices=["keep", "clip", "drop-invalid"],
        default="keep",
        help="How to handle non-binary attack/recovery labels (default: keep)",
    )
    parser.add_argument(
        "--clean-min-alt-m",
        type=float,
        default=CLEAN_MIN_ALT_M_DEFAULT,
        help="Keep only clean rows with altitude above this value in meters (default: 48)",
    )
    parser.add_argument(
        "--disable-clean-unit-harmonization",
        action="store_true",
        help="Disable clean unit harmonization to attack-domain style units",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_file, attack_file, grouped = locate_input_tables(input_dir)

    drop_remaining_nan = not args.keep_remaining_nan

    clean_df, clean_report = preprocess_clean(
        clean_file,
        fill_strategy=args.fill_strategy,
        drop_exact_duplicates=args.drop_exact_duplicates,
        drop_remaining_nan=drop_remaining_nan,
        clean_min_alt_m=args.clean_min_alt_m,
        harmonize_units=not args.disable_clean_unit_harmonization,
    )
    attack_df, attack_report = preprocess_attack(
        attack_file,
        fill_strategy=args.fill_strategy,
        drop_exact_duplicates=args.drop_exact_duplicates,
        drop_remaining_nan=drop_remaining_nan,
        binary_label_policy=args.binary_label_policy,
    )
    clean_long_df = build_clean_long(clean_df)

    clean_out = output_dir / f"{clean_file.stem}.processed.csv"
    attack_out = output_dir / f"{attack_file.stem}.processed.csv"
    clean_df.to_csv(clean_out, index=False)
    attack_df.to_csv(attack_out, index=False)

    clean_long_out = None
    if clean_long_df is not None:
        clean_long_out = output_dir / f"{clean_file.stem}.long.csv"
        clean_long_df.to_csv(clean_long_out, index=False)

    report = {
        "input_dir": str(input_dir),
        "selected_files": {
            "clean": str(clean_file),
            "attack": str(attack_file),
        },
        "detected_files": {
            key: [str(p) for p in paths]
            for key, paths in grouped.items()
        },
        "clean_report": clean_report,
        "attack_report": attack_report,
        "outputs": {
            "clean_processed": str(clean_out),
            "attack_processed": str(attack_out),
            "clean_long": str(clean_long_out) if clean_long_out else None,
        },
        "settings": {
            "fill_strategy": args.fill_strategy,
            "drop_exact_duplicates": args.drop_exact_duplicates,
            "drop_remaining_nan": drop_remaining_nan,
            "binary_label_policy": args.binary_label_policy,
            "clean_min_alt_m": args.clean_min_alt_m,
            "clean_unit_harmonization": not args.disable_clean_unit_harmonization,
        },
    }

    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[OK] clean file: {clean_file.name}")
    print(f"[OK] attack file: {attack_file.name}")
    print(f"[OK] output clean: {clean_out.name} ({len(clean_df)} rows)")
    print(f"[OK] output attack: {attack_out.name} ({len(attack_df)} rows)")
    if clean_long_out is not None:
        print(f"[OK] output clean long: {clean_long_out.name} ({len(clean_long_df)} rows)")
    print(f"[OK] report: {report_path.name}")


if __name__ == "__main__":
    main()
