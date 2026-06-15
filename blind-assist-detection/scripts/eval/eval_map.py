"""Evaluate MobileNetV2-SSD checkpoints with image-aware VOC-style mAP@0.5."""

from __future__ import annotations

import argparse
import json
import sys
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
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate(
    model: SSDMobileNetV2,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    iou_threshold: float,
    conf_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[float, dict[int, float], dict[int, int], dict[int, int]]:
    model.eval()
    detections: dict[int, list[dict]] = {i: [] for i in range(1, num_classes)}
    groundtruths: dict[int, dict[str, list[np.ndarray]]] = {i: {} for i in range(1, num_classes)}
    prediction_counts = {i: 0 for i in range(1, num_classes)}
    gt_counts = {i: 0 for i in range(1, num_classes)}

    print("[Eval] Running inference...", flush=True)
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
                pred_boxes = results[item_idx]["boxes"].cpu().numpy()
                pred_scores = results[item_idx]["scores"].cpu().numpy()
                pred_labels = results[item_idx]["labels"].cpu().numpy()
                gt_boxes = cxcywh_to_xyxy(batch["boxes"][item_idx]).numpy()
                gt_labels = batch["labels"][item_idx].numpy()

                for class_id in range(1, num_classes):
                    class_gt = gt_boxes[gt_labels == class_id]
                    groundtruths[class_id][image_name] = [box for box in class_gt]
                    gt_counts[class_id] += int(len(class_gt))

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    class_id = int(label)
                    if class_id not in detections:
                        continue
                    detections[class_id].append(
                        {
                            "image": image_name,
                            "score": float(score),
                            "box": box.astype(np.float32),
                        }
                    )
                    prediction_counts[class_id] += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(dataloader)}]", flush=True)

    print("[Eval] Computing AP...", flush=True)
    per_class_ap: dict[int, float] = {}
    for class_id in range(1, num_classes):
        class_dets = sorted(detections[class_id], key=lambda item: item["score"], reverse=True)
        class_gts = groundtruths[class_id]
        total_gt = gt_counts[class_id]

        if total_gt == 0:
            per_class_ap[class_id] = 0.0
            continue

        used = {image: np.zeros(len(boxes), dtype=bool) for image, boxes in class_gts.items()}
        tp = np.zeros(len(class_dets), dtype=np.float32)
        fp = np.zeros(len(class_dets), dtype=np.float32)

        for det_idx, detection in enumerate(class_dets):
            image_name = detection["image"]
            gt_boxes = np.array(class_gts.get(image_name, []), dtype=np.float32)
            if gt_boxes.size == 0:
                fp[det_idx] = 1
                continue

            ious = iou_one_to_many(detection["box"], gt_boxes)
            image_used = used[image_name]
            ious[image_used] = -1.0
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold:
                tp[det_idx] = 1
                image_used[best_idx] = True
            else:
                fp[det_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / max(total_gt, 1)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
        per_class_ap[class_id] = compute_ap(recall, precision) if len(class_dets) > 0 else 0.0

    mAP = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return mAP, per_class_ap, gt_counts, prediction_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/ssd_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
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

    model = SSDMobileNetV2(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        input_size=config["model"]["input_size"][0],
    ).to(device)
    model.configure_anchors(
        config["anchors"]["feature_maps"],
        config["anchors"]["min_sizes"],
        config["anchors"]["max_sizes"],
        config["anchors"]["aspect_ratios"],
    )

    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}", flush=True)

    conf_threshold = args.conf_threshold if args.conf_threshold is not None else config["evaluation"]["conf_threshold"]
    mAP, per_class_ap, gt_counts, prediction_counts = evaluate(
        model,
        dataloader,
        device,
        config["model"]["num_classes"],
        iou_threshold=config["evaluation"]["iou_threshold"],
        conf_threshold=conf_threshold,
        nms_threshold=config["inference"]["nms_threshold"],
        max_detections=config["inference"]["max_detections"],
    )

    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "iou_threshold": config["evaluation"]["iou_threshold"],
        "conf_threshold": conf_threshold,
        "mAP": mAP,
        "per_class_ap": {classes[class_id - 1]: float(ap) for class_id, ap in per_class_ap.items()},
        "gt_counts": {classes[class_id - 1]: int(count) for class_id, count in gt_counts.items()},
        "prediction_counts": {
            classes[class_id - 1]: int(count) for class_id, count in prediction_counts.items()
        },
    }

    print("\n" + "=" * 50)
    print(f"Evaluation result ({args.split}, mAP@0.5)")
    print("=" * 50)
    for class_id, ap in per_class_ap.items():
        print(f"  {classes[class_id - 1]:20s}: {ap:.4f}")
    print("-" * 50)
    print(f"  mAP: {mAP:.4f}")

    output_path = (
        resolve_project_path(args.output)
        if args.output
        else PROJECT_ROOT / "outputs" / "metrics" / f"eval_{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"[Eval] Saved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
