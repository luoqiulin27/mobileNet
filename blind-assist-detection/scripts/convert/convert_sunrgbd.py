from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sunrgbd_profiles import canonicalize_name, get_sunrgbd_profile


DEFAULT_SOURCE_ROOTS = (
    "kv1/b3dodata",
    "kv1/NYUdata",
    "kv2/align_kv2",
    "kv2/kinect2data",
    "realsense/lg",
    "realsense/sa",
    "realsense/sh",
    "realsense/shr",
    "xtion/sun3ddata",
    "xtion/xtion_align_data",
)


def reset_output_dir(output_dir: Path) -> None:
    for name in ["images", "labels", "configs", "meta"]:
        target = output_dir / name
        if target.exists():
            shutil.rmtree(target)


def safe_stem(relative_sample_dir: Path) -> str:
    raw = "_".join(relative_sample_dir.parts)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return f"{stem}_000000"


def iter_sample_dirs(dataset_dir: Path, source_roots: tuple[str, ...]) -> list[Path]:
    sample_dirs: list[Path] = []
    for rel_root in source_roots:
        root = dataset_dir / rel_root
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "image").is_dir() and (child / "annotation2Dfinal" / "index.json").exists():
                sample_dirs.append(child)
    return sample_dirs


def find_image(sample_dir: Path) -> Path | None:
    image_dir = sample_dir / "image"
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images = sorted(image_dir.glob(pattern))
        if images:
            return images[0]
    return None


def load_annotation(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_boxes(
    annotation: dict,
    image_width: int,
    image_height: int,
    profile,
    min_area: int,
    min_box_size: int,
) -> tuple[list[tuple[int, float, float, float, float]], Counter, Counter]:
    class_to_id = profile.class_to_id
    ignored = set(profile.ignored_names)
    boxes: list[tuple[int, float, float, float, float]] = []
    raw_counts: Counter = Counter()
    ignored_counts: Counter = Counter()

    object_names = []
    for obj in annotation.get("objects", []):
        if isinstance(obj, dict):
            object_names.append(canonicalize_name(str(obj.get("name", ""))))
        else:
            object_names.append("")

    frames = annotation.get("frames") or []
    if not frames or not isinstance(frames[0], dict):
        return boxes, raw_counts, ignored_counts

    for polygon in frames[0].get("polygon", []):
        object_idx = polygon.get("object")
        if not isinstance(object_idx, int) or object_idx < 0 or object_idx >= len(object_names):
            ignored_counts["invalid_object_index"] += 1
            continue

        raw_name = object_names[object_idx]
        raw_counts[raw_name] += 1
        target_class = profile.aliases.get(raw_name)
        if target_class is None:
            ignored_counts[raw_name if raw_name in ignored else f"unmapped:{raw_name}"] += 1
            continue

        xs = polygon.get("x") or []
        ys = polygon.get("y") or []
        if not isinstance(xs, (list, tuple)) or not isinstance(ys, (list, tuple)):
            ignored_counts["invalid_polygon"] += 1
            continue
        if len(xs) < 2 or len(ys) < 2:
            ignored_counts["invalid_polygon"] += 1
            continue

        x_min = max(0.0, min(float(x) for x in xs))
        y_min = max(0.0, min(float(y) for y in ys))
        x_max = min(float(image_width), max(float(x) for x in xs))
        y_max = min(float(image_height), max(float(y) for y in ys))

        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w < min_box_size or box_h < min_box_size or box_w * box_h < min_area:
            ignored_counts["too_small"] += 1
            continue

        class_id = class_to_id[target_class]
        cx = (x_min + x_max) / 2.0 / image_width
        cy = (y_min + y_max) / 2.0 / image_height
        width = box_w / image_width
        height = box_h / image_height

        if width <= 0.01 or height <= 0.01:
            ignored_counts["too_small_normalized"] += 1
            continue

        boxes.append((class_id, cx, cy, width, height))

    return boxes, raw_counts, ignored_counts


def write_png(source_path: Path, target_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image.save(target_path, format="PNG")
            return width, height
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SUNRGBD 2D annotations to YOLO labels")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT.parent / "data" / "SUNRGBD"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--profile", type=str, default="sunrgbd_indoor_12class")
    parser.add_argument("--min-area", type=int, default=180)
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    profile = get_sunrgbd_profile(args.profile)
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "data" / profile.output_dir_name

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir}")
        return 1

    if args.clean_output:
        print(f"[SUNRGBD] Cleaning output: {output_dir}")
        reset_output_dir(output_dir)

    image_out = output_dir / "images" / "all"
    label_out = output_dir / "labels" / "all"
    config_out = output_dir / "configs"
    meta_out = output_dir / "meta"
    for path in (image_out, label_out, config_out, meta_out):
        path.mkdir(parents=True, exist_ok=True)

    (config_out / "classes.txt").write_text("\n".join(profile.classes) + "\n", encoding="utf-8")

    sample_dirs = iter_sample_dirs(dataset_dir, DEFAULT_SOURCE_ROOTS)
    if args.limit_samples > 0:
        sample_dirs = sample_dirs[: args.limit_samples]

    print(f"[SUNRGBD] Samples to process: {len(sample_dirs)}")

    total_images = 0
    frames_with_boxes = 0
    total_boxes = 0
    class_counts: Counter = Counter()
    raw_counts: Counter = Counter()
    ignored_counts: Counter = Counter()
    missing_images = 0
    bad_annotations = 0
    bad_images = 0
    t_start = time.time()

    for idx, sample_dir in enumerate(sample_dirs, 1):
        image_path = find_image(sample_dir)
        if image_path is None:
            missing_images += 1
            continue

        annotation_path = sample_dir / "annotation2Dfinal" / "index.json"
        annotation = load_annotation(annotation_path)
        if annotation is None:
            bad_annotations += 1
            continue

        stem = safe_stem(sample_dir.relative_to(dataset_dir))
        target_image = image_out / f"{stem}.png"
        size = write_png(image_path, target_image)
        if size is None:
            bad_images += 1
            continue

        image_width, image_height = size
        boxes, sample_raw_counts, sample_ignored_counts = extract_boxes(
            annotation,
            image_width,
            image_height,
            profile,
            args.min_area,
            args.min_box_size,
        )
        raw_counts.update(sample_raw_counts)
        ignored_counts.update(sample_ignored_counts)

        label_path = label_out / f"{stem}.txt"
        label_lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cid, cx, cy, w, h in boxes]
        label_path.write_text("".join(label_lines), encoding="utf-8")

        total_images += 1
        if boxes:
            frames_with_boxes += 1
        total_boxes += len(boxes)
        for cid, *_ in boxes:
            class_counts[profile.classes[cid]] += 1

        if idx % 250 == 0 or idx == len(sample_dirs):
            elapsed = time.time() - t_start
            print(f"[SUNRGBD] {idx}/{len(sample_dirs)} samples, boxes={total_boxes}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - t_start
    report = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "profile": profile.name,
        "classes": list(profile.classes),
        "min_area": args.min_area,
        "min_box_size": args.min_box_size,
        "samples_found": len(sample_dirs),
        "images_written": total_images,
        "frames_with_boxes": frames_with_boxes,
        "empty_label_files": total_images - frames_with_boxes,
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "top_raw_names": dict(raw_counts.most_common(80)),
        "ignored_counts": dict(ignored_counts.most_common(80)),
        "missing_images": missing_images,
        "bad_annotations": bad_annotations,
        "bad_images": bad_images,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(meta_out / "conversion_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[SUNRGBD] Done in {elapsed:.1f}s")
    print(f"[SUNRGBD] Images: {total_images}, with boxes: {frames_with_boxes}, boxes: {total_boxes}")
    for class_name in profile.classes:
        print(f"  {class_name}: {class_counts[class_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
