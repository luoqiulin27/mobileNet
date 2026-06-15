import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT = PROJECT_ROOT / "scripts" / "convert" / "convert_sanpo_phase1.py"
SPLIT = PROJECT_ROOT / "scripts" / "convert" / "split_phase1.py"
VERIFY = PROJECT_ROOT / "scripts" / "convert" / "verify_phase1.py"


def run_step(args: list[str]) -> None:
    print("[Rebuild] Running:", " ".join(args))
    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a SANPO converted dataset end-to-end")
    parser.add_argument("--profile", type=str, default="sanpo_obstacle_8class")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT.parent / "data" / "SANPO-Real-Labeled-Full"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(PROJECT_ROOT / "data" / args.profile)

    py = sys.executable
    run_step([
        py, str(CONVERT),
        "--profile", args.profile,
        "--dataset", args.dataset,
        "--output", output,
        "--min-area", str(args.min_area),
        "--clean-output",
    ])
    run_step([
        py, str(SPLIT),
        "--data-dir", output,
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
    ])
    run_step([
        py, str(VERIFY),
        "--data-dir", output,
    ])
    print("[Rebuild] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
