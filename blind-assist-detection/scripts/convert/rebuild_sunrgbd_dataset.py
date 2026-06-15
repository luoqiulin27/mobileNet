from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT = PROJECT_ROOT / "scripts" / "convert" / "convert_sunrgbd.py"
SPLIT = PROJECT_ROOT / "scripts" / "convert" / "split_phase1.py"
VERIFY = PROJECT_ROOT / "scripts" / "convert" / "verify_phase1.py"


def run_step(args: list[str]) -> None:
    print("[Rebuild SUNRGBD] Running:", " ".join(args), flush=True)
    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild SUNRGBD indoor detection data")
    parser.add_argument("--profile", type=str, default="sunrgbd_indoor_12class")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT.parent / "data" / "SUNRGBD"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "sunrgbd_indoor_12class"))
    parser.add_argument("--min-area", type=int, default=180)
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit-samples", type=int, default=0)
    args = parser.parse_args()

    py = sys.executable
    convert_args = [
        py,
        str(CONVERT),
        "--profile",
        args.profile,
        "--dataset",
        args.dataset,
        "--output",
        args.output,
        "--min-area",
        str(args.min_area),
        "--min-box-size",
        str(args.min_box_size),
        "--clean-output",
    ]
    if args.limit_samples > 0:
        convert_args += ["--limit-samples", str(args.limit_samples)]

    run_step(convert_args)
    run_step([
        py,
        str(SPLIT),
        "--data-dir",
        args.output,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
    ])
    run_step([py, str(VERIFY), "--data-dir", args.output])
    print("[Rebuild SUNRGBD] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
