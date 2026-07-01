from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


TASKS = [
    {
        "name": "sanpo_ablation_se_formal100",
        "config": "src/configs/ssd_default_se.yaml",
        "batch_size": "32",
        "num_workers": "2",
    },
    {
        "name": "sanpo_ablation_cbam_formal100",
        "config": "src/configs/ssd_default_cbam.yaml",
        "batch_size": "32",
        "num_workers": "2",
    },
    {
        "name": "sanpo_ablation_iou_formal100",
        "config": "src/configs/ssd_default_iou.yaml",
        "batch_size": "32",
        "num_workers": "2",
    },
    {
        "name": "sanpo_ablation_diou_formal100",
        "config": "src/configs/ssd_default_diou.yaml",
        "batch_size": "32",
        "num_workers": "2",
    },
]


def active_train_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []

    lines = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "train_ssd.py" in line or "run_backbone_compare_queue.py" in line or "run_ch4_ablation_queue.py" in line:
            lines.append(line)
    return lines


def wait_for_gpu_slot(poll_seconds: int = 120) -> None:
    while True:
        active = [line for line in active_train_processes() if "run_ch4_ablation_queue.py" not in line]
        if not active:
            print("[Queue] no active training jobs, starting queued task", flush=True)
            return
        print("[Queue] waiting for existing training jobs to finish...", flush=True)
        for item in active:
            print(f"  {item}", flush=True)
        time.sleep(poll_seconds)


def run(cmd: list[str]) -> None:
    print("[Queue] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def checkpoint_dir_for(task: dict[str, str]) -> Path:
    return PROJECT_ROOT / "outputs" / "runs" / task["name"] / "checkpoints"


def metrics_dir_for(task: dict[str, str]) -> Path:
    return PROJECT_ROOT / "outputs" / "runs" / task["name"] / "metrics"


def build_train_command(task: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train/train_ssd.py",
        "--config",
        task["config"],
        "--gpu",
        "0",
        "--epochs",
        "100",
        "--batch-size",
        task["batch_size"],
        "--num-workers",
        task["num_workers"],
        "--val-every",
        "5",
        "--run-name",
        task["name"],
    ]

    last_checkpoint = checkpoint_dir_for(task) / "last.pth"
    if last_checkpoint.exists():
        print(f"[Queue] found existing checkpoint, resuming: {last_checkpoint}", flush=True)
        cmd.extend(["--resume", str(last_checkpoint)])

    return cmd


def maybe_run_eval(task: dict[str, str], *, checkpoint: Path, output_path: Path, script_path: str) -> None:
    if output_path.exists():
        print(f"[Queue] skip existing eval result: {output_path}", flush=True)
        return

    if not checkpoint.exists():
        print(f"[Queue] skip eval because checkpoint is missing: {checkpoint}", flush=True)
        return

    run(
        [
            sys.executable,
            script_path,
            "--config",
            task["config"],
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--gpu",
            "0",
            "--batch-size",
            "32",
            "--num-workers",
            "2",
            "--output",
            str(output_path),
        ]
    )


def main() -> int:
    wait_for_gpu_slot()

    for task in TASKS:
        run(build_train_command(task))

        checkpoint = checkpoint_dir_for(task) / "best.pth"
        metrics_dir = metrics_dir_for(task)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        maybe_run_eval(
            task,
            checkpoint=checkpoint,
            output_path=metrics_dir / "eval_map_val.json",
            script_path="scripts/eval/eval_map.py",
        )
        maybe_run_eval(
            task,
            checkpoint=checkpoint,
            output_path=metrics_dir / "eval_safety_val.json",
            script_path="scripts/eval/eval_safety_detection.py",
        )

    print("[Queue] all chapter 4 ablation tasks completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
