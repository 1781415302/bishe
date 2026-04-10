#!/usr/bin/env python3
"""One-click pipeline for PID-Piper residual-model experiments.

Pipeline stages:
1) Extract clean wide table from DataFlash BIN with bin_to_wide.py.
2) Preprocess clean + attack tables with preprocess_two_csv.py.
3) Train fusion models with train_pid_piper_fusion.py.

Examples:
  python run_pid_piper_pipeline.py
  python run_pid_piper_pipeline.py --quick
  python run_pid_piper_pipeline.py --work-dir analysis_exp1 --bin-file 00000005.BIN
  python run_pid_piper_pipeline.py --skip-training
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


# -----------------------------------------------------------------------------
# IDE RUN CONFIG (edit only this section for click-to-run usage)
# -----------------------------------------------------------------------------
USE_IDE_CONFIG_WHEN_NO_ARGS = True

IDE_RUN_CONFIG = {
    "python": sys.executable,
    "bin_file": "00000005.BIN",
    "attack_csv": "Data_Piper_Training_Wide.csv",
    "work_dir": "analysis_one_click",
    "bin_to_wide_script": r"c:\home\jiran\ardupilot\Tools\bin_to_wide.py",
    "preprocess_script": "preprocess_two_csv.py",
    "train_script": "train_pid_piper_fusion.py",
    "freq": 50.0,
    "semantic_mode": "attack-compatible",
    "gt_source": None,
    "report_name": "preprocessing_report.json",
    "train_output_subdir": "training_artifacts",
    "epochs_ref": 40,
    "epochs_gate": 40,
    "batch_size": 512,
    "val_fraction": 0.2,
    "lr_ref": 1e-3,
    "lr_gate": 1e-3,
    "seed": 42,
    "normalize_features": True,
    "z_clip": 8.0,
    "save_diagnostics": True,
    "quick": False,
    "skip_training": False,
    "reset_work_dir": False,
}
# -----------------------------------------------------------------------------


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


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_cmd(cmd: List[str], env: dict | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def build_args_for_ide() -> argparse.Namespace:
    return argparse.Namespace(**IDE_RUN_CONFIG)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click extraction + preprocessing + training")

    parser.add_argument("--python", default=IDE_RUN_CONFIG["python"], help="Python executable to run child scripts")

    parser.add_argument("--bin-file", default=IDE_RUN_CONFIG["bin_file"], help="Path to clean simulator BIN log")
    parser.add_argument(
        "--attack-csv",
        default=IDE_RUN_CONFIG["attack_csv"],
        help="Path to attack simulator wide CSV",
    )
    parser.add_argument(
        "--work-dir",
        default=IDE_RUN_CONFIG["work_dir"],
        help="Output working directory for this run",
    )

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

    parser.add_argument("--train-output-subdir", default=IDE_RUN_CONFIG["train_output_subdir"], help="Training output subdir")
    parser.add_argument("--epochs-ref", type=int, default=IDE_RUN_CONFIG["epochs_ref"])
    parser.add_argument("--epochs-gate", type=int, default=IDE_RUN_CONFIG["epochs_gate"])
    parser.add_argument("--batch-size", type=int, default=IDE_RUN_CONFIG["batch_size"])
    parser.add_argument("--val-fraction", type=float, default=IDE_RUN_CONFIG["val_fraction"])
    parser.add_argument("--lr-ref", type=float, default=IDE_RUN_CONFIG["lr_ref"])
    parser.add_argument("--lr-gate", type=float, default=IDE_RUN_CONFIG["lr_gate"])
    parser.add_argument("--seed", type=int, default=IDE_RUN_CONFIG["seed"])
    parser.add_argument("--normalize-features", type=str_to_bool, default=IDE_RUN_CONFIG["normalize_features"])
    parser.add_argument("--z-clip", type=float, default=IDE_RUN_CONFIG["z_clip"])
    parser.add_argument("--save-diagnostics", type=str_to_bool, default=IDE_RUN_CONFIG["save_diagnostics"])
    parser.add_argument("--quick", action="store_true", default=IDE_RUN_CONFIG["quick"], help="Shortcut for lighter training (ref=8, gate=6)")
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
    if USE_IDE_CONFIG_WHEN_NO_ARGS and len(sys.argv) == 1:
        args = build_args_for_ide()
        print("[INFO] No CLI args detected; using IDE_RUN_CONFIG.")
    else:
        args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    python_exe = resolve_path(args.python, project_root)
    bin_file = resolve_path(args.bin_file, project_root)
    attack_csv = resolve_path(args.attack_csv, project_root)
    work_dir = resolve_path(args.work_dir, project_root)

    bin_to_wide_script = resolve_path(args.bin_to_wide_script, project_root)
    preprocess_script = resolve_path(args.preprocess_script, project_root)
    train_script = resolve_path(args.train_script, project_root)

    # Some workspace mappings expose the training script only via Windows path.
    if (
        args.train_script == "train_pid_piper_fusion.py"
        and not train_script.is_file()
    ):
        fallback_train = Path(r"C:\home\jiran\work\pid-piper\train_pid_piper_fusion.py")
        if fallback_train.is_file():
            train_script = fallback_train.resolve()

    ensure_file(python_exe, "python executable")
    ensure_file(bin_file, "bin file")
    ensure_file(attack_csv, "attack csv")
    ensure_file(bin_to_wide_script, "bin_to_wide script")
    ensure_file(preprocess_script, "preprocess script")
    if not args.skip_training:
        ensure_file(train_script, "train script")

    if args.reset_work_dir and work_dir.exists():
        if work_dir == project_root:
            raise ValueError("Refusing to reset project root as work-dir")
        shutil.rmtree(work_dir)
        print(f"[OK] Reset work-dir: {work_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    attack_in_work = work_dir / "Data_Piper_Training_Wide.csv"
    shutil.copy2(attack_csv, attack_in_work)
    print(f"[OK] Copied attack csv -> {attack_in_work}")

    clean_wide = work_dir / "Data_Piper_WIDE_clean.csv"

    # 1) Clean extraction
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

    # 2) Preprocess
    preprocess_cmd = [
        str(python_exe),
        str(preprocess_script),
        "--input-dir",
        str(work_dir),
        "--output-dir",
        str(work_dir),
        "--report-name",
        args.report_name,
    ]
    run_cmd(preprocess_cmd)

    clean_long = work_dir / "Data_Piper_WIDE_clean.long.csv"
    attack_processed = work_dir / "Data_Piper_Training_Wide.processed.csv"
    report_path = work_dir / args.report_name

    ensure_file(clean_long, "clean long csv")
    ensure_file(attack_processed, "attack processed csv")
    ensure_file(report_path, "preprocess report")

    print(f"[OK] Preprocess outputs ready: {clean_long}, {attack_processed}")

    if args.skip_training:
        print("[OK] skip-training enabled; pipeline finished after preprocessing.")
        return

    # 3) Train
    epochs_ref = 8 if args.quick else args.epochs_ref
    epochs_gate = 6 if args.quick else args.epochs_gate

    train_out = work_dir / args.train_output_subdir
    train_out.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        str(python_exe),
        str(train_script),
        "--clean-long",
        str(clean_long),
        "--attack",
        str(attack_processed),
        "--output-dir",
        str(train_out),
        "--epochs-ref",
        str(epochs_ref),
        "--epochs-gate",
        str(epochs_gate),
        "--batch-size",
        str(args.batch_size),
        "--val-fraction",
        str(args.val_fraction),
        "--lr-ref",
        str(args.lr_ref),
        "--lr-gate",
        str(args.lr_gate),
        "--seed",
        str(args.seed),
        "--normalize-features",
        "true" if args.normalize_features else "false",
        "--z-clip",
        str(args.z_clip),
        "--save-diagnostics",
        "true" if args.save_diagnostics else "false",
    ]

    # UNC/mounted paths can fail with h5py file lock; keep this disabled for training subprocess.
    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    run_cmd(train_cmd, env=env)

    summary_path = train_out / "training_summary.json"
    ensure_file(summary_path, "training summary")

    print("[OK] Pipeline completed successfully")
    print(f"[OK] Working directory: {work_dir}")
    print(f"[OK] Training summary: {summary_path}")


if __name__ == "__main__":
    main()
