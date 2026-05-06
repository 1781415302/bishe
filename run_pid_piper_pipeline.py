#!/usr/bin/env python3
"""One-click pipeline for PID-Piper gate-NN experiments.

Pipeline stages:
1) Optionally prepare clean data from an already-extracted clean wide/long CSV,
   or extract clean wide data from a DataFlash BIN with bin_to_wide.py.
2) Prepare attack data from the local PID-Piper attack_*.csv / wide CSV output.
3) Preprocess clean + attack tables only when a clean wide table is available.
4) Train gate-NN models with train_pid_piper_fusion.py.

Examples:
  python run_pid_piper_pipeline.py
  python run_pid_piper_pipeline.py --quick
  python run_pid_piper_pipeline.py --training-suite single
  python run_pid_piper_pipeline.py --training-suite comparison --experiment-routes fused_huber_sigmoid,fused_huber_hardsigmoid
  python run_pid_piper_pipeline.py --pairs-manifest gate_pair_manifest.csv --export-fdeep
  python run_pid_piper_pipeline.py --work-dir analysis_exp1 --attack-csv logs/copter/attack_*.csv
  python run_pid_piper_pipeline.py --work-dir analysis_exp1 --clean-wide clean_sample_check/Data_Piper_WIDE_clean.csv --attack-csv logs/copter/attack_*.csv
  python run_pid_piper_pipeline.py --work-dir analysis_exp1 --bin-file 00000005.BIN --clean-wide ""
  python run_pid_piper_pipeline.py --skip-training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# -----------------------------------------------------------------------------
# IDE RUN CONFIG (edit only this section for click-to-run usage)
# -----------------------------------------------------------------------------
USE_IDE_CONFIG_WHEN_NO_ARGS = True
PREFERRED_GPU_PYTHON = Path(r"C:\Users\17814\anaconda3\envs\ai_tf210_gpu\python.exe")
DEFAULT_IDE_PYTHON = str(PREFERRED_GPU_PYTHON) if PREFERRED_GPU_PYTHON.is_file() else sys.executable

IDE_RUN_CONFIG = {
    "python": DEFAULT_IDE_PYTHON,
    "clean_wide": None,
    "clean_long": None,
    "bin_file": None,
    "pairs_manifest": None,
    "attack_csv": None,
    "work_dir": "analysis_pair_lstm_{time}",
    "auto_discover_pairs": True,
    "mission_glob": "mission*",
    "attack_pattern": "*.csv",
    "auto_manifest_name": "_auto_pairs_manifest.csv",
    "bin_to_wide_script": r"c:\home\jiran\ardupilot\Tools\bin_to_wide.py",
    "preprocess_script": "preprocess_two_csv.py",
    "pair_prepare_script": "prepare_gate_training_pairs.py",
    "train_script": "train_pid_piper_fusion.py",
    "freq": 50.0,
    "semantic_mode": "attack-compatible",
    "gt_source": None,
    "report_name": "preprocessing_report.json",
    "pair_report_name": "pairing_report.json",
    "pair_output_csv": "gate_training_attack_with_target.csv",
    "train_output_subdir": "training_artifacts",
    # single: train once with the scalar settings below.
    # comparison: preprocess once, then run TRAINING_ROUTE_PRESETS and write
    # experiment_comparison.csv/json under work_dir.
    "training_suite": "comparison",
    "experiment_routes": "fused_huber_sigmoid,fused_huber_sigmoid_l2_001,fused_huber_hardsigmoid",
    "comparison_output_csv": "experiment_comparison.csv",
    "comparison_output_json": "experiment_comparison.json",
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
    "max_align_error_norm": 0.02,
    "attack_clean_flight_ratio": 1.05,
    "min_ml_pid_gap": 1.0e-4,
    "enforce_target_reachability": True,
    "target_reachability_margin": 0.02,
    "alpha_target_cap": 1.0,
    "gate_loss": "fused_huber",
    "alpha_l2_penalty": 0.0,
    "fit_verbose": 1,
    "fill_strategy": "ffill",
    "binary_label_policy": "keep",
    "clean_min_alt_m": 48.0,
    "drop_exact_duplicates": False,
    "keep_remaining_nan": False,
    "disable_clean_unit_harmonization": False,
    "quick": False,
    "skip_training": False,
    "reset_work_dir": False,
}
# -----------------------------------------------------------------------------


TRAINING_ROUTE_PRESETS: Dict[str, Dict[str, Any]] = {
    "alpha_star_lstm_40ep": {
        "description": "LSTM gate trained against alpha_star with alpha_huber, close to the original continuous-alpha baseline.",
        "train_output_subdir": "training_artifacts_alpha_star_lstm_40ep",
        "epochs_gate": 40,
        "batch_size": 512,
        "lr_gate": 1.0e-3,
        "gate_loss": "alpha_huber",
        "gate_output_activation": "sigmoid",
        "alpha_target_cap": 1.0,
        "alpha_l2_penalty": 0.0,
        "target_reachability_margin": 0.0,
    },
    "alpha_huber_cap06": {
        "description": "LSTM gate trained against capped alpha_star; tests whether limiting ML blending helps.",
        "train_output_subdir": "training_artifacts_alpha_huber_cap06",
        "epochs_gate": 8,
        "batch_size": 2048,
        "lr_gate": 5.0e-4,
        "gate_loss": "alpha_huber",
        "gate_output_activation": "sigmoid",
        "alpha_target_cap": 0.6,
        "alpha_l2_penalty": 0.0,
        "target_reachability_margin": 0.02,
    },
    "fused_huber_sigmoid": {
        "description": "Directly optimizes fused output y_pid + alpha*(y_ml-y_pid), with ordinary sigmoid output.",
        "train_output_subdir": "training_artifacts_fused_huber_sigmoid",
        "epochs_gate": 8,
        "batch_size": 2048,
        "lr_gate": 5.0e-4,
        "gate_loss": "fused_huber",
        "gate_output_activation": "sigmoid",
        "alpha_target_cap": 1.0,
        "alpha_l2_penalty": 0.0,
        "target_reachability_margin": 0.02,
    },
    "fused_huber_sigmoid_l2_001": {
        "description": "Fused-output objective plus alpha^2 penalty; tests conservative ML usage.",
        "train_output_subdir": "training_artifacts_fused_huber_sigmoid_l2_001",
        "epochs_gate": 8,
        "batch_size": 2048,
        "lr_gate": 5.0e-4,
        "gate_loss": "fused_huber",
        "gate_output_activation": "sigmoid",
        "alpha_target_cap": 1.0,
        "alpha_l2_penalty": 0.01,
        "target_reachability_margin": 0.02,
    },
    "fused_huber_hardsigmoid": {
        "description": "Current recommended route: fused-output objective with hard_sigmoid for exact PID fallback.",
        "train_output_subdir": "training_artifacts_fused_huber_hardsigmoid",
        "epochs_gate": 8,
        "batch_size": 2048,
        "lr_gate": 5.0e-4,
        "gate_loss": "fused_huber",
        "gate_output_activation": "hard_sigmoid",
        "alpha_target_cap": 1.0,
        "alpha_l2_penalty": 0.0,
        "target_reachability_margin": 0.02,
    },
}


def str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_work_dir(value: str, project_root: Path) -> Path:
    raw = str(value).replace("{time}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    return resolve_path(raw, project_root)


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def resolve_optional_input(value: str | None, project_root: Path, label: str) -> Path | None:
    if value is None or str(value).strip() == "":
        return None

    raw_value = str(value)
    if has_glob_magic(raw_value):
        pattern = Path(raw_value)
        if not pattern.is_absolute():
            pattern = project_root / pattern
        parent = pattern.parent
        matches = sorted(
            [p for p in parent.glob(pattern.name) if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(f"{label} pattern matched no files: {pattern}")
        selected = matches[0].resolve()
        print(f"[INFO] Selected {label}: {selected}")
        return selected

    path = resolve_path(raw_value, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def copy_if_needed(src: Path, dst: Path, label: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() and src.samefile(dst):
            print(f"[OK] {label} already in work-dir: {dst}")
            return
    except OSError:
        pass
    shutil.copy2(src, dst)
    print(f"[OK] Copied {label} -> {dst}")


def run_cmd(cmd: List[str], env: dict | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def parse_route_names(value: str) -> List[str]:
    raw = str(value).strip()
    if raw == "" or raw.lower() == "all":
        return list(TRAINING_ROUTE_PRESETS.keys())
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in TRAINING_ROUTE_PRESETS]
    if unknown:
        valid = ", ".join(TRAINING_ROUTE_PRESETS)
        raise ValueError(f"Unknown experiment route(s): {unknown}. Valid routes: {valid}")
    return names


def apply_route_config(args: argparse.Namespace, route_name: str | None, route: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "route_name": route_name or "single",
        "description": "Single training run from CLI/IDE scalar settings.",
        "train_output_subdir": args.train_output_subdir,
        "epochs_gate": args.epochs_gate,
        "batch_size": args.batch_size,
        "lr_gate": args.lr_gate,
        "gate_loss": args.gate_loss,
        "gate_output_activation": args.gate_output_activation,
        "alpha_target_cap": args.alpha_target_cap,
        "alpha_l2_penalty": args.alpha_l2_penalty,
        "target_reachability_margin": args.target_reachability_margin,
    }
    if route is not None:
        cfg.update(route)
        cfg["route_name"] = route_name
    if args.quick:
        cfg["epochs_gate"] = min(int(cfg["epochs_gate"]), 6)
    return cfg


def build_train_cmd(
    *,
    python_exe: Path,
    train_script: Path,
    attack_for_training: Path,
    train_out: Path,
    args: argparse.Namespace,
    gate_target_column: str,
    route_cfg: Dict[str, Any],
    clean_long: Path,
    convert_script: Path,
) -> List[str]:
    cmd = [
        str(python_exe),
        str(train_script),
        "--attack",
        str(attack_for_training),
        "--output-dir",
        str(train_out),
        "--epochs-gate",
        str(route_cfg["epochs_gate"]),
        "--batch-size",
        str(route_cfg["batch_size"]),
        "--val-fraction",
        str(args.val_fraction),
        "--lr-gate",
        str(route_cfg["lr_gate"]),
        "--seed",
        str(args.seed),
        "--normalize-features",
        "true" if args.normalize_features else "false",
        "--z-clip",
        str(args.z_clip),
        "--save-diagnostics",
        "true" if args.save_diagnostics else "false",
        "--runtime-bundle-dir",
        args.runtime_bundle_dir,
        "--gate-target-column",
        gate_target_column,
        "--gate-architecture",
        args.gate_architecture,
        "--gate-output-activation",
        str(route_cfg["gate_output_activation"]),
        "--sequence-length",
        str(args.sequence_length),
        "--min-ml-pid-gap",
        str(args.min_ml_pid_gap),
        "--enforce-target-reachability",
        "true" if args.enforce_target_reachability else "false",
        "--target-reachability-margin",
        str(route_cfg["target_reachability_margin"]),
        "--alpha-target-cap",
        str(route_cfg["alpha_target_cap"]),
        "--gate-loss",
        str(route_cfg["gate_loss"]),
        "--alpha-l2-penalty",
        str(route_cfg["alpha_l2_penalty"]),
        "--fit-verbose",
        str(args.fit_verbose),
    ]
    if clean_long.exists():
        cmd.extend(["--clean-long", str(clean_long)])
    if args.export_fdeep:
        cmd.extend([
            "--export-fdeep",
            "--convert-script",
            str(convert_script),
        ])
    return cmd


def summarize_training_result(route_cfg: Dict[str, Any], summary_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for artifact in data.get("artifacts", []):
        fused = artifact.get("mae_fused_vs_target")
        pid = artifact.get("mae_pid_vs_target")
        ml = artifact.get("mae_ml_vs_target")
        improvement = None
        if fused is not None and pid not in {None, 0}:
            improvement = (float(pid) - float(fused)) / float(pid) * 100.0
        rows.append(
            {
                "route": route_cfg["route_name"],
                "description": route_cfg["description"],
                "angle": artifact.get("angle"),
                "mae_fused_vs_target": fused,
                "mae_pid_vs_target": pid,
                "mae_ml_vs_target": ml,
                "improvement_vs_pid_pct": improvement,
                "mae_alpha_vs_target": artifact.get("mae_alpha_vs_target"),
                "gate_train_rows": artifact.get("gate_train_rows"),
                "gate_val_rows": artifact.get("gate_val_rows"),
                "summary_path": str(summary_path),
                "train_output_subdir": route_cfg["train_output_subdir"],
                "epochs_gate": route_cfg["epochs_gate"],
                "batch_size": route_cfg["batch_size"],
                "lr_gate": route_cfg["lr_gate"],
                "gate_loss": route_cfg["gate_loss"],
                "gate_output_activation": route_cfg["gate_output_activation"],
                "alpha_target_cap": route_cfg["alpha_target_cap"],
                "alpha_l2_penalty": route_cfg["alpha_l2_penalty"],
                "target_reachability_margin": route_cfg["target_reachability_margin"],
            }
        )
    return rows


def write_comparison_outputs(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route",
        "angle",
        "mae_fused_vs_target",
        "mae_pid_vs_target",
        "mae_ml_vs_target",
        "improvement_vs_pid_pct",
        "mae_alpha_vs_target",
        "gate_train_rows",
        "gate_val_rows",
        "epochs_gate",
        "batch_size",
        "lr_gate",
        "gate_loss",
        "gate_output_activation",
        "alpha_target_cap",
        "alpha_l2_penalty",
        "target_reachability_margin",
        "train_output_subdir",
        "summary_path",
        "description",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")


def pick_latest(paths: List[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def discover_mission_pairs(
    project_root: Path,
    mission_glob: str,
    attack_pattern: str,
) -> List[dict]:
    pairs: List[dict] = []
    mission_dirs = sorted([p for p in project_root.glob(mission_glob) if p.is_dir()])

    for mission_dir in mission_dirs:
        bins = [p for p in mission_dir.glob("*.BIN") if p.is_file()]
        bins.extend([p for p in mission_dir.glob("*.bin") if p.is_file()])
        clean_csv = pick_latest(bins)

        attacks = sorted([p for p in mission_dir.glob(attack_pattern) if p.is_file()])

        if clean_csv is None:
            csv_clean_candidates = []
            for p in mission_dir.glob("*.csv"):
                if not p.is_file():
                    continue
                if p.name.lower().startswith("attack_"):
                    continue
                if p in attacks:
                    continue
                csv_clean_candidates.append(p)
            clean_csv = pick_latest(csv_clean_candidates)

        if clean_csv is None:
            continue

        attacks = [p for p in attacks if p.resolve() != clean_csv.resolve()]
        if not attacks:
            continue

        for attack_csv in attacks:
            pairs.append(
                {
                    "mission_id": mission_dir.name,
                    "clean_csv": str(clean_csv.resolve()),
                    "attack_csv": str(attack_csv.resolve()),
                }
            )

    return pairs


def write_auto_manifest(rows: List[dict], work_dir: Path, file_name: str) -> Path:
    manifest_path = work_dir / file_name
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mission_id", "clean_csv", "attack_csv"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def build_args_for_ide() -> argparse.Namespace:
    return argparse.Namespace(**IDE_RUN_CONFIG)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click extraction + preprocessing + training")

    parser.add_argument("--python", default=IDE_RUN_CONFIG["python"], help="Python executable to run child scripts")

    parser.add_argument(
        "--clean-wide",
        default=IDE_RUN_CONFIG["clean_wide"],
        help="Optional path to already-extracted clean wide CSV with gt_roll/gt_pitch/gt_yaw. Use empty string to disable.",
    )
    parser.add_argument(
        "--clean-long",
        default=IDE_RUN_CONFIG["clean_long"],
        help="Optional path to already-preprocessed clean long CSV. Takes precedence over --clean-wide.",
    )
    parser.add_argument(
        "--bin-file",
        default=IDE_RUN_CONFIG["bin_file"],
        help="Optional clean simulator BIN log. Used only when --clean-long and --clean-wide are empty.",
    )
    parser.add_argument(
        "--pairs-manifest",
        default=IDE_RUN_CONFIG["pairs_manifest"],
        help="CSV manifest with mission_id,clean_csv,attack_csv for paired continuous gate training.",
    )
    parser.add_argument(
        "--attack-csv",
        default=IDE_RUN_CONFIG["attack_csv"],
        help="Path or glob pattern for PID-Piper attack wide CSV. Ignored when --pairs-manifest is provided.",
    )
    parser.add_argument(
        "--work-dir",
        default=IDE_RUN_CONFIG["work_dir"],
        help="Output working directory for this run",
    )
    parser.add_argument("--auto-discover-pairs", type=str_to_bool, default=IDE_RUN_CONFIG["auto_discover_pairs"])
    parser.add_argument("--mission-glob", default=IDE_RUN_CONFIG["mission_glob"])
    parser.add_argument(
        "--attack-pattern",
        default=IDE_RUN_CONFIG["attack_pattern"],
        help="CSV glob inside each mission folder. Default '*.csv' uses every CSV as an attack log.",
    )
    parser.add_argument("--auto-manifest-name", default=IDE_RUN_CONFIG["auto_manifest_name"])

    parser.add_argument(
        "--bin-to-wide-script",
        default=IDE_RUN_CONFIG["bin_to_wide_script"],
        help="Path to clean extraction script bin_to_wide.py",
    )
    parser.add_argument(
        "--preprocess-script",
        default=IDE_RUN_CONFIG["preprocess_script"],
        help="Path to preprocess_two_csv.py",
    )
    parser.add_argument(
        "--pair-prepare-script",
        default=IDE_RUN_CONFIG["pair_prepare_script"],
        help="Path to prepare_gate_training_pairs.py",
    )
    parser.add_argument(
        "--train-script",
        default=IDE_RUN_CONFIG["train_script"],
        help="Path to train_pid_piper_fusion.py",
    )

    parser.add_argument("--freq", type=float, default=IDE_RUN_CONFIG["freq"], help="Target frequency for clean extraction")
    parser.add_argument(
        "--semantic-mode",
        choices=["attack-compatible", "legacy"],
        default=IDE_RUN_CONFIG["semantic_mode"],
        help="Semantic mode used by bin_to_wide.py",
    )
    parser.add_argument(
        "--gt-source",
        default=IDE_RUN_CONFIG["gt_source"],
        help="Optional gt source priority list for bin_to_wide.py (comma-separated)",
    )

    parser.add_argument("--report-name", default=IDE_RUN_CONFIG["report_name"], help="Preprocess report filename")
    parser.add_argument("--pair-report-name", default=IDE_RUN_CONFIG["pair_report_name"])
    parser.add_argument("--pair-output-csv", default=IDE_RUN_CONFIG["pair_output_csv"])

    parser.add_argument("--train-output-subdir", default=IDE_RUN_CONFIG["train_output_subdir"], help="Training output subdir")
    parser.add_argument("--training-suite", choices=["single", "comparison"], default=IDE_RUN_CONFIG["training_suite"])
    parser.add_argument(
        "--experiment-routes",
        default=IDE_RUN_CONFIG["experiment_routes"],
        help="Comma-separated route names for --training-suite comparison, or all.",
    )
    parser.add_argument("--comparison-output-csv", default=IDE_RUN_CONFIG["comparison_output_csv"])
    parser.add_argument("--comparison-output-json", default=IDE_RUN_CONFIG["comparison_output_json"])
    parser.add_argument("--epochs-ref", type=int, default=None, help="Ignored compatibility option; reference models are no longer trained")
    parser.add_argument("--epochs-gate", type=int, default=IDE_RUN_CONFIG["epochs_gate"])
    parser.add_argument("--batch-size", type=int, default=IDE_RUN_CONFIG["batch_size"])
    parser.add_argument("--val-fraction", type=float, default=IDE_RUN_CONFIG["val_fraction"])
    parser.add_argument("--lr-ref", type=float, default=None, help="Ignored compatibility option; reference models are no longer trained")
    parser.add_argument("--lr-gate", type=float, default=IDE_RUN_CONFIG["lr_gate"])
    parser.add_argument("--seed", type=int, default=IDE_RUN_CONFIG["seed"])
    parser.add_argument("--normalize-features", type=str_to_bool, default=IDE_RUN_CONFIG["normalize_features"])
    parser.add_argument("--z-clip", type=float, default=IDE_RUN_CONFIG["z_clip"])
    parser.add_argument("--save-diagnostics", type=str_to_bool, default=IDE_RUN_CONFIG["save_diagnostics"])
    parser.add_argument("--export-fdeep", action="store_true", default=IDE_RUN_CONFIG["export_fdeep"], help="Export trained Keras models to frugally-deep JSON")
    parser.add_argument("--convert-script", default=IDE_RUN_CONFIG["convert_script"], help="Path to frugally-deep convert_model.py")
    parser.add_argument("--runtime-bundle-dir", default=IDE_RUN_CONFIG["runtime_bundle_dir"])
    parser.add_argument("--gate-target-column", default=IDE_RUN_CONFIG["gate_target_column"])
    parser.add_argument("--gate-architecture", choices=["lstm", "mlp"], default=IDE_RUN_CONFIG["gate_architecture"])
    parser.add_argument("--gate-output-activation", choices=["sigmoid", "hard_sigmoid"], default=IDE_RUN_CONFIG["gate_output_activation"])
    parser.add_argument("--sequence-length", type=int, default=IDE_RUN_CONFIG["sequence_length"])
    parser.add_argument("--max-align-error-norm", type=float, default=IDE_RUN_CONFIG["max_align_error_norm"])
    parser.add_argument("--attack-clean-flight-ratio", type=float, default=IDE_RUN_CONFIG["attack_clean_flight_ratio"])
    parser.add_argument("--min-ml-pid-gap", type=float, default=IDE_RUN_CONFIG["min_ml_pid_gap"])
    parser.add_argument("--enforce-target-reachability", type=str_to_bool, default=IDE_RUN_CONFIG["enforce_target_reachability"])
    parser.add_argument("--target-reachability-margin", type=float, default=IDE_RUN_CONFIG["target_reachability_margin"])
    parser.add_argument("--alpha-target-cap", type=float, default=IDE_RUN_CONFIG["alpha_target_cap"])
    parser.add_argument("--gate-loss", choices=["alpha_huber", "fused_huber"], default=IDE_RUN_CONFIG["gate_loss"])
    parser.add_argument("--alpha-l2-penalty", type=float, default=IDE_RUN_CONFIG["alpha_l2_penalty"])
    parser.add_argument("--fit-verbose", type=int, choices=[0, 1, 2], default=IDE_RUN_CONFIG["fit_verbose"])
    parser.add_argument("--fill-strategy", choices=["ffill", "ffill_bfill", "none"], default=IDE_RUN_CONFIG["fill_strategy"])
    parser.add_argument("--binary-label-policy", choices=["keep", "clip", "drop-invalid"], default=IDE_RUN_CONFIG["binary_label_policy"])
    parser.add_argument("--clean-min-alt-m", type=float, default=IDE_RUN_CONFIG["clean_min_alt_m"])
    parser.add_argument("--drop-exact-duplicates", action="store_true", default=IDE_RUN_CONFIG["drop_exact_duplicates"])
    parser.add_argument("--keep-remaining-nan", action="store_true", default=IDE_RUN_CONFIG["keep_remaining_nan"])
    parser.add_argument("--disable-clean-unit-harmonization", action="store_true", default=IDE_RUN_CONFIG["disable_clean_unit_harmonization"])
    parser.add_argument("--quick", action="store_true", default=IDE_RUN_CONFIG["quick"], help="Shortcut for lighter gate training (gate=6)")
    parser.add_argument("--skip-training", action="store_true", default=IDE_RUN_CONFIG["skip_training"], help="Only run extraction + preprocessing")
    parser.add_argument(
        "--reset-work-dir",
        type=str_to_bool,
        default=IDE_RUN_CONFIG["reset_work_dir"],
        help="Delete and recreate work-dir before run [true/false]",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    no_cli_args = len(sys.argv) == 1
    if USE_IDE_CONFIG_WHEN_NO_ARGS and no_cli_args:
        args = build_args_for_ide()
        print("[INFO] No CLI args detected; using IDE_RUN_CONFIG.")
    else:
        args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    python_exe = resolve_path(args.python, project_root)
    clean_long_source = resolve_optional_input(args.clean_long, project_root, "clean long csv")
    clean_wide_source = resolve_optional_input(args.clean_wide, project_root, "clean wide csv")
    bin_file = resolve_optional_input(args.bin_file, project_root, "bin file")
    pairs_manifest = resolve_optional_input(args.pairs_manifest, project_root, "pairs manifest")
    attack_csv = None if pairs_manifest is not None else resolve_optional_input(args.attack_csv, project_root, "attack csv")
    work_dir = resolve_work_dir(args.work_dir, project_root)

    discovered_pairs: List[dict] = []
    if (
        pairs_manifest is None
        and attack_csv is None
        and args.auto_discover_pairs
        and clean_long_source is None
        and clean_wide_source is None
        and bin_file is None
    ):
        discovered_pairs = discover_mission_pairs(project_root, args.mission_glob, args.attack_pattern)
        if discovered_pairs:
            mission_count = len({Path(row["clean_csv"]).parent for row in discovered_pairs})
            print(
                f"[INFO] Auto-discovered {len(discovered_pairs)} attack/clean pair(s) "
                f"from {mission_count} mission folder(s) using attack_pattern={args.attack_pattern}"
            )
        else:
            print("[INFO] Auto-discovery found no valid mission pairs.")

    bin_to_wide_script = resolve_path(args.bin_to_wide_script, project_root)
    preprocess_script = resolve_path(args.preprocess_script, project_root)
    pair_prepare_script = resolve_path(args.pair_prepare_script, project_root)
    train_script = resolve_path(args.train_script, project_root)
    convert_script = resolve_path(args.convert_script, project_root)

    # Some workspace mappings expose the training script only via Windows path.
    if (
        args.train_script == "train_pid_piper_fusion.py"
        and not train_script.is_file()
    ):
        fallback_train = Path(r"C:\home\jiran\work\pid-piper\train_pid_piper_fusion.py")
        if fallback_train.is_file():
            train_script = fallback_train.resolve()

    ensure_file(python_exe, "python executable")
    using_pairs_mode = (pairs_manifest is not None) or bool(discovered_pairs)
    if clean_long_source is None and clean_wide_source is None:
        if bin_file is not None:
            ensure_file(bin_to_wide_script, "bin_to_wide script")
    if using_pairs_mode:
        ensure_file(pair_prepare_script, "pair prepare script")
    else:
        ensure_file(preprocess_script, "preprocess script")
    if not args.skip_training:
        ensure_file(train_script, "train script")
        if args.export_fdeep:
            ensure_file(convert_script, "convert_model.py")

    if args.reset_work_dir and work_dir.exists():
        if work_dir == project_root:
            raise ValueError("Refusing to reset project root as work-dir")
        shutil.rmtree(work_dir)
        print(f"[OK] Reset work-dir: {work_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    if pairs_manifest is None and discovered_pairs:
        pairs_manifest = write_auto_manifest(discovered_pairs, work_dir, args.auto_manifest_name)
        print(f"[OK] Generated pairs manifest: {pairs_manifest}")

    if pairs_manifest is None and attack_csv is None:
        raise ValueError(
            "No training input found. Provide --pairs-manifest, --attack-csv, or place mission*/(*.csv + *.BIN) for auto-discovery."
        )

    clean_wide = work_dir / "Data_Piper_WIDE_clean.csv"
    clean_long = work_dir / "Data_Piper_WIDE_clean.long.csv"
    attack_for_training = work_dir / args.pair_output_csv if pairs_manifest is not None else work_dir / "Data_Piper_Training_Wide.csv"
    report_path = work_dir / args.report_name

    if pairs_manifest is not None:
        pair_cmd = [
            str(python_exe),
            str(pair_prepare_script),
            "--pairs-manifest",
            str(pairs_manifest),
            "--output-dir",
            str(work_dir),
            "--output-csv",
            args.pair_output_csv,
            "--report-name",
            args.pair_report_name,
            "--python-exe",
            str(python_exe),
            "--bin-to-wide-script",
            str(bin_to_wide_script),
            "--freq",
            str(args.freq),
            "--semantic-mode",
            args.semantic_mode,
            "--max-align-error-norm",
            str(args.max_align_error_norm),
            "--attack-clean-flight-ratio",
            str(args.attack_clean_flight_ratio),
            "--min-ml-pid-gap",
            str(args.min_ml_pid_gap),
        ]
        if args.gt_source:
            pair_cmd.extend(["--gt-source", args.gt_source])
        run_cmd(pair_cmd)
    else:
        copy_if_needed(attack_csv, attack_for_training, "attack csv")

    if pairs_manifest is not None:
        print("[INFO] pairs-manifest supplied; using paired y_target CSV for continuous gate training.")
    elif clean_long_source is not None:
        copy_if_needed(clean_long_source, clean_long, "clean long csv")
        print("[INFO] clean-long supplied for compatibility; gate-only training does not train reference models.")
    elif clean_wide_source is not None or bin_file is not None:
        if clean_wide_source is not None:
            copy_if_needed(clean_wide_source, clean_wide, "clean wide csv")
        else:
            # 1) Clean extraction from BIN, kept as an explicit fallback path.
            extract_cmd = [
                str(python_exe),
                str(bin_to_wide_script),
                str(bin_file),
                "-o",
                str(clean_wide),
                "--freq",
                str(args.freq),
                "--semantic-mode",
                args.semantic_mode,
            ]
            if args.gt_source:
                extract_cmd.extend(["--gt-source", args.gt_source])
            run_cmd(extract_cmd)

        # 2) Preprocess clean wide + attack wide.
        preprocess_cmd = [
            str(python_exe),
            str(preprocess_script),
            "--input-dir",
            str(work_dir),
            "--output-dir",
            str(work_dir),
            "--report-name",
            args.report_name,
            "--fill-strategy",
            args.fill_strategy,
            "--binary-label-policy",
            args.binary_label_policy,
            "--clean-min-alt-m",
            str(args.clean_min_alt_m),
        ]
        if args.drop_exact_duplicates:
            preprocess_cmd.append("--drop-exact-duplicates")
        if args.keep_remaining_nan:
            preprocess_cmd.append("--keep-remaining-nan")
        if args.disable_clean_unit_harmonization:
            preprocess_cmd.append("--disable-clean-unit-harmonization")
        run_cmd(preprocess_cmd)

        attack_for_training = work_dir / "Data_Piper_Training_Wide.processed.csv"
        ensure_file(report_path, "preprocess report")
    else:
        print("[INFO] No clean source supplied; using raw attack CSV directly for gate-only training.")

    ensure_file(attack_for_training, "attack csv for training")

    if clean_long.exists():
        print(f"[OK] Training inputs ready: {clean_long}, {attack_for_training}")
    else:
        print(f"[OK] Training inputs ready: {attack_for_training}")

    if args.skip_training:
        print("[OK] skip-training enabled; pipeline finished after preparing inputs.")
        return

    # 3) Train
    gate_target_column = "y_target" if pairs_manifest is not None else args.gate_target_column

    # UNC/mounted paths can fail with h5py file lock; keep this disabled for training subprocess.
    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    if args.training_suite == "comparison":
        route_names = parse_route_names(args.experiment_routes)
        print(f"[INFO] Training comparison suite with {len(route_names)} route(s): {', '.join(route_names)}")
        comparison_rows: List[Dict[str, Any]] = []
        routes_manifest: List[Dict[str, Any]] = []
        for route_name in route_names:
            route_cfg = apply_route_config(args, route_name, TRAINING_ROUTE_PRESETS[route_name])
            routes_manifest.append(route_cfg)
            train_out = work_dir / str(route_cfg["train_output_subdir"])
            train_out.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Route {route_name}: {route_cfg['description']}")
            train_cmd = build_train_cmd(
                python_exe=python_exe,
                train_script=train_script,
                attack_for_training=attack_for_training,
                train_out=train_out,
                args=args,
                gate_target_column=gate_target_column,
                route_cfg=route_cfg,
                clean_long=clean_long,
                convert_script=convert_script,
            )
            run_cmd(train_cmd, env=env)
            summary_path = train_out / "training_summary.json"
            ensure_file(summary_path, f"{route_name} training summary")
            comparison_rows.extend(summarize_training_result(route_cfg, summary_path))

        comparison_csv = work_dir / args.comparison_output_csv
        comparison_json = work_dir / args.comparison_output_json
        write_comparison_outputs(comparison_rows, comparison_csv, comparison_json)
        routes_manifest_path = work_dir / "experiment_routes_manifest.json"
        routes_manifest_path.write_text(json.dumps(routes_manifest, indent=2, ensure_ascii=True), encoding="utf-8")

        print("[OK] Pipeline completed successfully")
        print(f"[OK] Working directory: {work_dir}")
        print(f"[OK] Comparison CSV: {comparison_csv}")
        print(f"[OK] Comparison JSON: {comparison_json}")
        print(f"[OK] Route manifest: {routes_manifest_path}")
        return

    route_cfg = apply_route_config(args, None, None)
    train_out = work_dir / str(route_cfg["train_output_subdir"])
    train_out.mkdir(parents=True, exist_ok=True)
    train_cmd = build_train_cmd(
        python_exe=python_exe,
        train_script=train_script,
        attack_for_training=attack_for_training,
        train_out=train_out,
        args=args,
        gate_target_column=gate_target_column,
        route_cfg=route_cfg,
        clean_long=clean_long,
        convert_script=convert_script,
    )
    run_cmd(train_cmd, env=env)

    summary_path = train_out / "training_summary.json"
    ensure_file(summary_path, "training summary")

    print("[OK] Pipeline completed successfully")
    print(f"[OK] Working directory: {work_dir}")
    print(f"[OK] Training summary: {summary_path}")


if __name__ == "__main__":
    main()
