"""Convert SANPO panoptic segmentation masks to YOLO obstacle boxes.

SANPO mask PNG encoding observed in v0:
- R channel: semantic class id (labelmap.json)
- B channel: instance id for panoptic classes

Default keeps only obstacle-like classes useful for a MobileNet obstacle detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CLASSES = ["obstacle", "vehicle", "pedestrian", "rider", "animal", "stairs", "traffic sign", "traffic light", "pole", "bike rack"]


def load_labelmap(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def boxes_from_mask(mask_path: Path, class_ids: dict[int, int], min_area: int) -> list[tuple[int, float, float, float, float]]:
    mask = np.asarray(Image.open(mask_path).convert("RGB"))
    semantic = mask[:, :, 0].astype(np.int32)
    instance = mask[:, :, 2].astype(np.int32)
    height, width = semantic.shape
    boxes: list[tuple[int, float, float, float, float]] = []
    for source_class_id, yolo_class_id in class_ids.items():
        class_pixels = semantic == source_class_id
        if not class_pixels.any():
            continue
        instance_ids = np.unique(instance[class_pixels])
        for instance_id in instance_ids:
            component = class_pixels & (instance == instance_id)
            area = int(component.sum())
            if area < min_area:
                continue
            ys, xs = np.where(component)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            box_width = (x_max - x_min + 1) / width
            box_height = (y_max - y_min + 1) / height
            x_center = (x_min + x_max + 1) / 2 / width
            y_center = (y_min + y_max + 1) / 2 / height
            boxes.append((yolo_class_id, x_center, y_center, box_width, box_height))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/SANPO-Real-RGB-6k")
    parser.add_argument("--output", default="data/SANPO-Real-YOLO-obstacles")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    labelmap = load_labelmap(dataset_dir / "metadata" / "labelmap.json")
    selected = [name for name in args.classes if name in labelmap]
    class_ids = {labelmap[name]: index for index, name in enumerate(selected)}

    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "metadata" / "image_manifest.csv"
    rows_written = images_with_boxes = missing_masks = 0
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            session_id = row["session_id"]
            mask_path = dataset_dir / "labels_segmentation_masks" / session_id / image_id
            if not mask_path.exists():
                missing_masks += 1
                continue
            boxes = boxes_from_mask(mask_path, class_ids, args.min_area)
            label_name = Path(image_id).with_suffix(".txt").name
            label_path = output_dir / "labels" / session_id / label_name
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
            rows_written += 1
            if boxes:
                images_with_boxes += 1
            if args.copy_images:
                src = Path(row["local_path"])
                dst = output_dir / "images" / session_id / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

    (output_dir / "classes.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    (output_dir / "dataset.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(selected)),
        encoding="utf-8",
    )
    print(json.dumps({"labels_written": rows_written, "images_with_boxes": images_with_boxes, "missing_masks": missing_masks, "classes": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
