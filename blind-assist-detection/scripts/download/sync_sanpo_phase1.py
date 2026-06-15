from __future__ import annotations

import argparse
import csv
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


GCS_HTTP_PREFIX = "https://storage.googleapis.com/gresearch/"


def read_labeled_sessions(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["session_id"] for row in reader if row.get("session_id")}


def read_manifest_rows(path: Path, labeled_sessions: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["session_id"] in labeled_sessions:
                rows.append(row)
    return rows


def download_file(url: str, destination: Path, timeout: int, retries: int) -> tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True, "exists"
    last_error = ""
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            destination.write_bytes(data)
            return True, "downloaded"
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(1)
    return False, last_error


def build_tasks(dataset_dir: Path, rows: list[dict[str, str]]) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for row in rows:
        session_id = row["session_id"]
        frame_name = row["frame_file"]

        rgb_url = GCS_HTTP_PREFIX + row["rgb_object"]
        rgb_dest = dataset_dir / "images" / session_id / frame_name
        tasks.append((rgb_url, rgb_dest))

        mask_url = GCS_HTTP_PREFIX + row["mask_object"]
        mask_dest = dataset_dir / "labels_segmentation_masks" / session_id / frame_name
        tasks.append((mask_url, mask_dest))
    return tasks


def write_progress(
    progress_path: Path,
    stop_event: threading.Event,
    counters: dict[str, int],
    total_tasks: int,
) -> None:
    while not stop_event.is_set():
        progress_path.write_text(
            "\n".join(
                [
                    f"total_tasks={total_tasks}",
                    f"done={counters['done']}",
                    f"downloaded={counters['downloaded']}",
                    f"exists={counters['exists']}",
                    f"failed={counters['failed']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stop_event.wait(5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=r"D:\projects\MobileNet\mobileNet\data\SANPO-Real-Labeled-Full",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    metadata_dir = dataset_dir / "metadata"
    manifest_path = metadata_dir / "image_manifest.csv"
    labeled_sessions_path = metadata_dir / "labeled_sessions.csv"
    log_dir = dataset_dir / "_download_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / "sanpo_phase1_progress.txt"
    failures_path = log_dir / "sanpo_phase1_failures.txt"

    labeled_sessions = read_labeled_sessions(labeled_sessions_path)
    rows = read_manifest_rows(manifest_path, labeled_sessions)
    tasks = build_tasks(dataset_dir, rows)

    counters = {"done": 0, "downloaded": 0, "exists": 0, "failed": 0}
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=write_progress,
        args=(progress_path, stop_event, counters, len(tasks)),
        daemon=True,
    )
    watcher.start()

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, url, dest, args.timeout, args.retries): (url, dest)
            for url, dest in tasks
        }
        for future in as_completed(futures):
            url, dest = futures[future]
            ok, status = future.result()
            counters["done"] += 1
            if ok:
                counters[status] += 1
            else:
                counters["failed"] += 1
                failures.append(f"{dest}\t{url}\t{status}")

    stop_event.set()
    watcher.join(timeout=1)

    failures_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    progress_path.write_text(
        "\n".join(
            [
                f"total_tasks={len(tasks)}",
                f"done={counters['done']}",
                f"downloaded={counters['downloaded']}",
                f"exists={counters['exists']}",
                f"failed={counters['failed']}",
                f"labeled_sessions={len(labeled_sessions)}",
                f"manifest_rows={len(rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(progress_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
