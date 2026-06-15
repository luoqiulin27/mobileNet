"""Download the full labeled SANPO-Real RGB + segmentation mask subset.

This script uses the public Google Research bucket and only downloads
sessions that expose `camera_chest/left/segmentation_masks/`. For each
available session, it downloads the paired RGB frames from
`camera_chest/left/video_frames/` and the matching mask PNGs into a fresh
dataset directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen, urlretrieve

BUCKET_DOWNLOAD = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/{object_name}?alt=media"
BUCKET_LIST = "https://storage.googleapis.com/storage/v1/b/gresearch/o?{query}"
ROOT_PREFIX = "sanpo_dataset/v0/sanpo-real/"
LABELMAP_OBJECT = "sanpo_dataset/v0/labelmap.json"
LABELTYPE_OBJECT = "sanpo_dataset/v0/labeltype.json"


def storage_list(prefix: str, *, delimiter: str | None = None, max_results: int = 1000, page_token: str | None = None) -> dict:
    query = {"prefix": prefix, "maxResults": str(max_results)}
    if delimiter:
        query["delimiter"] = delimiter
    if page_token:
        query["pageToken"] = page_token
    url = BUCKET_LIST.format(query=urlencode(query))
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_object(object_name: str, output_path: Path, retries: int = 3) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = BUCKET_DOWNLOAD.format(object_name=quote(object_name, safe=""))
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, tmp_path)
            tmp_path.replace(output_path)
            return True
        except (HTTPError, URLError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt == retries:
                print(f"FAILED {object_name}: {exc}")
                return False
            time.sleep(1.5 * attempt)
    return False


def list_session_prefixes() -> list[str]:
    prefixes: list[str] = []
    token: str | None = None
    while True:
        payload = storage_list(ROOT_PREFIX, delimiter="/", max_results=1000, page_token=token)
        prefixes.extend(payload.get("prefixes", []))
        token = payload.get("nextPageToken")
        if not token:
            return prefixes


def has_mask_prefix(session_prefix: str) -> bool:
    mask_prefix = session_prefix + "camera_chest/left/segmentation_masks/"
    payload = storage_list(mask_prefix, max_results=1)
    return bool(payload.get("items"))


def discover_labeled_sessions(workers: int) -> list[str]:
    sessions = list_session_prefixes()
    labeled: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(has_mask_prefix, session): session for session in sessions}
        for completed, future in enumerate(as_completed(futures), start=1):
            session = futures[future]
            if future.result():
                labeled.append(session)
            if completed == len(futures) or completed % 50 == 0:
                elapsed = max(time.time() - started, 0.001)
                print(json.dumps({
                    "sessions_checked": completed,
                    "sessions_total": len(sessions),
                    "labeled_sessions": len(labeled),
                    "rate_per_sec": round(completed / elapsed, 2),
                }))
    return sorted(labeled)


def list_objects(prefix: str) -> list[dict]:
    items: list[dict] = []
    token: str | None = None
    while True:
        payload = storage_list(prefix, max_results=1000, page_token=token)
        items.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            return items


def collect_download_pairs(labeled_sessions: list[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for index, session_prefix in enumerate(labeled_sessions, start=1):
        rgb_prefix = session_prefix + "camera_chest/left/video_frames/"
        mask_prefix = session_prefix + "camera_chest/left/segmentation_masks/"
        rgb_items = {Path(item["name"]).name: item["name"] for item in list_objects(rgb_prefix)}
        mask_items = {Path(item["name"]).name: item["name"] for item in list_objects(mask_prefix)}
        common = sorted(set(rgb_items) & set(mask_items))
        session_id = session_prefix.rstrip("/").split("/")[-1]
        for filename in common:
            pairs.append((session_id, rgb_items[filename], mask_items[filename]))
        if index == len(labeled_sessions) or index % 20 == 0:
            print(json.dumps({
                "sessions_indexed": index,
                "sessions_total": len(labeled_sessions),
                "paired_frames": len(pairs),
            }))
    return pairs


def write_metadata(output_dir: Path, labeled_sessions: list[str], pairs: list[tuple[str, str, str]]) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    download_object(LABELMAP_OBJECT, metadata_dir / "labelmap.json")
    download_object(LABELTYPE_OBJECT, metadata_dir / "labeltype.json")

    session_rows = [{"session_id": prefix.rstrip("/").split("/")[-1]} for prefix in labeled_sessions]
    with (metadata_dir / "labeled_sessions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id"])
        writer.writeheader()
        writer.writerows(session_rows)

    with (metadata_dir / "image_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "frame_file", "rgb_object", "mask_object"])
        writer.writeheader()
        for session_id, rgb_object, mask_object in pairs:
            writer.writerow({
                "session_id": session_id,
                "frame_file": Path(rgb_object).name,
                "rgb_object": rgb_object,
                "mask_object": mask_object,
            })


def build_tasks(output_dir: Path, pairs: list[tuple[str, str, str]]) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for session_id, rgb_object, mask_object in pairs:
        filename = Path(rgb_object).name
        tasks.append((rgb_object, output_dir / "images" / session_id / filename))
        tasks.append((mask_object, output_dir / "labels_segmentation_masks" / session_id / filename))
    return tasks


def run_downloads(tasks: list[tuple[str, Path]], workers: int, retries: int, progress_every: int) -> tuple[int, int, int]:
    skipped = sum(1 for _, path in tasks if path.exists() and path.stat().st_size > 0)
    pending = [(obj, path) for obj, path in tasks if not (path.exists() and path.stat().st_size > 0)]
    downloaded = failed = completed = 0
    started = time.time()
    print(json.dumps({
        "download_tasks": len(tasks),
        "skipped_existing": skipped,
        "pending": len(pending),
        "workers": workers,
    }))

    def worker(task: tuple[str, Path]) -> bool:
        object_name, output_path = task
        return download_object(object_name, output_path, retries=retries)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(worker, task) for task in pending]
        for future in as_completed(futures):
            completed += 1
            if future.result():
                downloaded += 1
            else:
                failed += 1
            seen = skipped + completed
            if seen == len(tasks) or completed % progress_every == 0:
                elapsed = max(time.time() - started, 0.001)
                print(json.dumps({
                    "seen": seen,
                    "downloaded": downloaded,
                    "skipped": skipped,
                    "failed": failed,
                    "pending": len(tasks) - seen,
                    "rate_per_sec": round(completed / elapsed, 2),
                }))
    return downloaded, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/SANPO-Real-Labeled-Full")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output)
    metadata_dir = output_dir / "metadata"
    sessions_cache = metadata_dir / "labeled_session_prefixes.json"

    if sessions_cache.exists():
        labeled_sessions = json.loads(sessions_cache.read_text(encoding="utf-8"))
    else:
        labeled_sessions = discover_labeled_sessions(args.workers)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sessions_cache.write_text(json.dumps(labeled_sessions, indent=2), encoding="utf-8")

    pairs = collect_download_pairs(labeled_sessions)
    write_metadata(output_dir, labeled_sessions, pairs)
    tasks = build_tasks(output_dir, pairs)
    downloaded, skipped, failed = run_downloads(tasks, args.workers, args.retries, args.progress_every)

    print(json.dumps({
        "labeled_sessions": len(labeled_sessions),
        "paired_frames": len(pairs),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "failed_files": failed,
        "output": str(output_dir),
    }, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
