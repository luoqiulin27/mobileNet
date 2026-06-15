"""Download SANPO-Real segmentation masks matching the local RGB subset.

This complements data/SANPO-Real-RGB-6k, which only contains video_frames.
It reads metadata/image_manifest.csv and downloads the paired mask PNG from
Google Research public storage into labels_segmentation_masks/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen, urlretrieve

BUCKET_DOWNLOAD = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/{object_name}?alt=media"
BUCKET_LIST = "https://storage.googleapis.com/storage/v1/b/gresearch/o?{query}"
LABELMAP_OBJECT = "sanpo_dataset/v0/labelmap.json"
LABELTYPE_OBJECT = "sanpo_dataset/v0/labeltype.json"


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
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt == retries:
                print(f"FAILED {object_name}: {exc}")
                return False
            time.sleep(1.5 * attempt)
    return False


def paired_mask_object(source_object: str) -> str:
    return source_object.replace("/video_frames/", "/segmentation_masks/")


def mask_prefix(source_object: str) -> str:
    mask_object = paired_mask_object(source_object)
    return mask_object.rsplit("/", 1)[0] + "/"


def prefix_has_objects(prefix: str, retries: int = 3) -> bool:
    query = urlencode({"prefix": prefix, "maxResults": "1"})
    url = BUCKET_LIST.format(query=query)
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("items"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                print(f"PREFIX_CHECK_FAILED {prefix}: {exc}")
                return False
            time.sleep(1.5 * attempt)
    return False


def discover_available_prefixes(prefixes: list[str], cache_path: Path, workers: int, retries: int) -> set[str]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return {prefix for prefix, available in cached.items() if available}

    results: dict[str, bool] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(prefix_has_objects, prefix, retries): prefix for prefix in prefixes}
        for completed, future in enumerate(as_completed(futures), start=1):
            prefix = futures[future]
            results[prefix] = future.result()
            if completed == len(futures) or completed % 50 == 0:
                available = sum(1 for value in results.values() if value)
                elapsed = max(time.time() - started, 0.001)
                print(json.dumps({
                    "prefixes_checked": completed,
                    "prefixes_total": len(prefixes),
                    "available": available,
                    "rate_per_sec": round(completed / elapsed, 2),
                }))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return {prefix for prefix, available in results.items() if available}


def read_manifest_rows(manifest_path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if limit and len(rows) >= limit:
                break
            rows.append(row)
    return rows


def build_tasks(rows: list[dict[str, str]], output_root: Path, available_prefixes: set[str]) -> tuple[list[tuple[str, Path]], int]:
    tasks: list[tuple[str, Path]] = []
    unavailable = 0
    for row in rows:
        source_object = row["source_object"]
        if mask_prefix(source_object) not in available_prefixes:
            unavailable += 1
            continue
        image_id = row["image_id"]
        session_id = row["session_id"]
        mask_object = paired_mask_object(source_object)
        output_path = output_root / session_id / image_id
        tasks.append((mask_object, output_path))
    return tasks, unavailable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/SANPO-Real-RGB-6k")
    parser.add_argument("--limit", type=int, default=0, help="0 means all manifest rows")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--refresh-availability", action="store_true")
    parser.add_argument("--no-discover-availability", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    manifest_path = dataset_dir / "metadata" / "image_manifest.csv"
    output_root = dataset_dir / "labels_segmentation_masks"
    metadata_dir = dataset_dir / "metadata"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    download_object(LABELMAP_OBJECT, metadata_dir / "labelmap.json")
    download_object(LABELTYPE_OBJECT, metadata_dir / "labeltype.json")

    rows = read_manifest_rows(manifest_path, args.limit)
    availability_cache = metadata_dir / "mask_available_prefixes.json"
    if args.refresh_availability and availability_cache.exists():
        availability_cache.unlink()

    if args.no_discover_availability:
        available_prefixes = {mask_prefix(row["source_object"]) for row in rows}
    else:
        unique_prefixes = sorted({mask_prefix(row["source_object"]) for row in rows})
        available_prefixes = discover_available_prefixes(unique_prefixes, availability_cache, args.workers, args.retries)

    all_tasks, unavailable = build_tasks(rows, output_root, available_prefixes)
    skipped = sum(1 for _, output_path in all_tasks if output_path.exists() and output_path.stat().st_size > 0)
    pending_tasks = [(obj, path) for obj, path in all_tasks if not (path.exists() and path.stat().st_size > 0)]
    total = len(all_tasks)
    downloaded = failed = completed = 0
    lock = Lock()
    started = time.time()

    print(json.dumps({
        "manifest_rows": len(rows),
        "downloadable": total,
        "unavailable_no_mask_prefix": unavailable,
        "skipped_existing": skipped,
        "pending": len(pending_tasks),
        "workers": args.workers,
    }))

    def worker(task: tuple[str, Path]) -> bool:
        object_name, output_path = task
        return download_object(object_name, output_path, retries=args.retries)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(worker, task) for task in pending_tasks]
        for future in as_completed(futures):
            ok = future.result()
            with lock:
                completed += 1
                if ok:
                    downloaded += 1
                else:
                    failed += 1
                seen = skipped + completed
                if seen == total or completed % args.progress_every == 0:
                    elapsed = max(time.time() - started, 0.001)
                    rate = completed / elapsed
                    print(json.dumps({
                        "seen": seen,
                        "downloaded": downloaded,
                        "skipped": skipped,
                        "failed": failed,
                        "pending": total - seen,
                        "rate_per_sec": round(rate, 2),
                    }))

    print(json.dumps({"seen": total, "downloaded": downloaded, "skipped": skipped, "failed": failed, "unavailable_no_mask_prefix": unavailable, "output": str(output_root)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
