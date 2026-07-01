"""Evaluate detection checkpoints with obstacle-assistance oriented metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import resolve_project_path
from src.datasets.detection_dataset import DetectionDataset, collate_fn, load_image_list
from src.models.ssd_mobilenet import SSDMobileNetV2


@dataclass
class GroundTruth:
    image: str
    class_id: int
    box: np.ndarray
    matched: bool = False


@dataclass
class DetectionItem:
    image: str
    class_id: int
    score: float
    box: np.ndarray
    matched: bool = False


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        selected = precision[recall >= threshold]
        if len(selected) > 0:
            ap += float(np.max(selected))
    return ap / 11.0


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, 4)
    return torch.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        dim=1,
    )


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - inter
    return np.where(union > 0, inter / union, 0.0)


def box_center_x(box: np.ndarray) -> float:
    return float((box[0] + box[2]) / 2.0)


def box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def is_center_box(box: np.ndarray, left: float, right: float) -> bool:
    cx = box_center_x(box)
    return left <= cx <= right


def is_near_box(box: np.ndarray, bottom_threshold: float, area_threshold: float) -> bool:
    return float(box[3]) >= bottom_threshold or box_area(box) >= area_threshold


def parse_critical_classes(raw: str | None, classes: list[str]) -> set[int]:
    if not raw:
        names = {
            "person",
            "vehicle",
            "rider",
            "stairs",
            "pole",
            "obstacle",
            "indoor_obstacle",
            "seat",
            "table",
            "box_bag",
            "door",
        }
    else:
        names = {item.strip() for item in raw.split(",") if item.strip()}
    return {idx + 1 for idx, name in enumerate(classes) if name in names}


def build_groundtruth_index(
    groundtruths: list[GroundTruth],
) -> dict[int, dict[str, list[GroundTruth]]]:
    index: dict[int, dict[str, list[GroundTruth]]] = {}
    for item in groundtruths:
        index.setdefault(item.class_id, {}).setdefault(item.image, []).append(item)
    return index


def match_detections(
    detections: list[DetectionItem],
    groundtruths: list[GroundTruth],
    num_classes: int,
    iou_threshold: float,
) -> tuple[dict[int, float], dict[int, dict[str, int]]]:
    gt_index = build_groundtruth_index(groundtruths)
    per_class_ap: dict[int, float] = {}
    per_class_counts: dict[int, dict[str, int]] = {}

    for class_id in range(1, num_classes):
        class_dets = sorted(
            [det for det in detections if det.class_id == class_id],
            key=lambda item: item.score,
            reverse=True,
        )
        class_gts_by_image = gt_index.get(class_id, {})
        total_gt = sum(len(items) for items in class_gts_by_image.values())
        tp = np.zeros(len(class_dets), dtype=np.float32)
        fp = np.zeros(len(class_dets), dtype=np.float32)

        for det_idx, detection in enumerate(class_dets):
            candidates = class_gts_by_image.get(detection.image, [])
            unmatched = [item for item in candidates if not item.matched]
            if not unmatched:
                fp[det_idx] = 1
                continue

            gt_boxes = np.array([item.box for item in unmatched], dtype=np.float32)
            ious = iou_one_to_many(detection.box, gt_boxes)
            best_idx = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0

            if best_iou >= iou_threshold:
                tp[det_idx] = 1
                detection.matched = True
                unmatched[best_idx].matched = True
            else:
                fp[det_idx] = 1

        tp_total = int(tp.sum())
        fp_total = int(fp.sum())
        fn_total = int(total_gt - tp_total)
        recall = tp_total / total_gt if total_gt else 0.0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0

        if total_gt > 0 and len(class_dets) > 0:
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall_curve = tp_cumsum / max(total_gt, 1)
            precision_curve = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
            per_class_ap[class_id] = compute_ap(recall_curve, precision_curve)
        else:
            per_class_ap[class_id] = 0.0

        per_class_counts[class_id] = {
            "gt": int(total_gt),
            "pred": len(class_dets),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision,
            "recall": recall,
        }

    return per_class_ap, per_class_counts


def recall_for_subset(items: list[GroundTruth]) -> dict[str, float | int]:
    total = len(items)
    matched = sum(1 for item in items if item.matched)
    return {
        "gt": total,
        "matched": matched,
        "missed": total - matched,
        "recall": matched / total if total else 0.0,
    }


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> SSDMobileNetV2:
    model = SSDMobileNetV2(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        input_size=config["model"]["input_size"][0],
        backbone=config["model"].get("backbone", "mobilenet_v2"),
        use_eca=config["model"].get("use_eca", False),
        eca_stages=config["model"].get("eca_stages"),
    ).to(device)
    model.configure_anchors(
        config["anchors"]["feature_maps"],
        config["anchors"]["min_sizes"],
        config["anchors"]["max_sizes"],
        config["anchors"]["aspect_ratios"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def collect_predictions(
    model: SSDMobileNetV2,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    conf_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[list[DetectionItem], list[GroundTruth]]:
    detections: list[DetectionItem] = []
    groundtruths: list[GroundTruth] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["images"].to(device, non_blocking=True)
            names = batch["names"]
            results = model.detect(
                images,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
            )

            for item_idx, image_name in enumerate(names):
                gt_boxes = cxcywh_to_xyxy(batch["boxes"][item_idx]).numpy()
                gt_labels = batch["labels"][item_idx].numpy()
                for box, label in zip(gt_boxes, gt_labels):
                    class_id = int(label)
                    if 0 < class_id < num_classes:
                        groundtruths.append(
                            GroundTruth(
                                image=image_name,
                                class_id=class_id,
                                box=box.astype(np.float32),
                            )
                        )

                for box, score, label in zip(
                    results[item_idx]["boxes"].cpu().numpy(),
                    results[item_idx]["scores"].cpu().numpy(),
                    results[item_idx]["labels"].cpu().numpy(),
                ):
                    class_id = int(label)
                    if 0 < class_id < num_classes:
                        detections.append(
                            DetectionItem(
                                image=image_name,
                                class_id=class_id,
                                score=float(score),
                                box=box.astype(np.float32),
                            )
                        )

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(dataloader)}]", flush=True)

    return detections, groundtruths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/ssd_default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--critical-classes", type=str, default=None)
    parser.add_argument("--center-left", type=float, default=0.33)
    parser.add_argument("--center-right", type=float, default=0.67)
    parser.add_argument("--near-bottom", type=float, default=0.65)
    parser.add_argument("--near-area", type=float, default=0.08)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    split_key = {"train": "train_list", "val": "val_list", "test": "test_list"}[args.split]
    names = load_image_list(str(resolve_project_path(data_cfg[split_key])))

    with open(resolve_project_path(data_cfg["classes_file"]), "r", encoding="utf-8") as file:
        classes = [line.strip() for line in file if line.strip()]

    dataset = DetectionDataset(
        image_dir=str(resolve_project_path(data_cfg["image_dir"])),
        label_dir=str(resolve_project_path(data_cfg["label_dir"])),
        image_list=names,
        input_size=config["model"]["input_size"][0],
        classes=classes,
        augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    print(f"[SafetyEval] device={device}, split={args.split}, samples={len(names)}", flush=True)
    model = load_model(config, checkpoint_path, device)
    conf_threshold = (
        args.conf_threshold
        if args.conf_threshold is not None
        else config["evaluation"]["conf_threshold"]
    )
    iou_threshold = (
        args.iou_threshold
        if args.iou_threshold is not None
        else config["evaluation"]["iou_threshold"]
    )

    detections, groundtruths = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=config["model"]["num_classes"],
        conf_threshold=conf_threshold,
        nms_threshold=config["inference"]["nms_threshold"],
        max_detections=config["inference"]["max_detections"],
    )
    per_class_ap, per_class_counts = match_detections(
        detections=detections,
        groundtruths=groundtruths,
        num_classes=config["model"]["num_classes"],
        iou_threshold=iou_threshold,
    )

    critical_class_ids = parse_critical_classes(args.critical_classes, classes)
    center_items = [
        item
        for item in groundtruths
        if item.class_id in critical_class_ids
        and is_center_box(item.box, args.center_left, args.center_right)
    ]
    near_items = [
        item
        for item in groundtruths
        if item.class_id in critical_class_ids
        and is_near_box(item.box, args.near_bottom, args.near_area)
    ]
    center_near_items = [
        item
        for item in groundtruths
        if item.class_id in critical_class_ids
        and is_center_box(item.box, args.center_left, args.center_right)
        and is_near_box(item.box, args.near_bottom, args.near_area)
    ]
    critical_items = [item for item in groundtruths if item.class_id in critical_class_ids]

    map_value = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "samples": len(names),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "classes": classes,
        "mAP": map_value,
        "per_class": {
            classes[class_id - 1]: {
                "ap": float(per_class_ap[class_id]),
                **per_class_counts[class_id],
            }
            for class_id in sorted(per_class_counts)
        },
        "safety_metrics": {
            "critical_classes": [
                classes[class_id - 1]
                for class_id in sorted(critical_class_ids)
                if 0 < class_id <= len(classes)
            ],
            "critical_recall": recall_for_subset(critical_items),
            "center_critical_recall": recall_for_subset(center_items),
            "near_critical_recall": recall_for_subset(near_items),
            "center_near_critical_recall": recall_for_subset(center_near_items),
            "center_region": [args.center_left, args.center_right],
            "near_rule": {
                "bottom_y2_at_least": args.near_bottom,
                "area_at_least": args.near_area,
            },
        },
    }

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[SafetyEval] mAP={map_value:.4f}")
    print(
        "[SafetyEval] center_near_critical_recall="
        f"{result['safety_metrics']['center_near_critical_recall']['recall']:.4f}"
    )
    print(f"[SafetyEval] saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
