from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import resolve_project_path
from src.inference import DetectionPredictor


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_label(label_path: Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Render qualitative prediction samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_key = {"train": "train_list", "val": "val_list", "test": "test_list"}[args.split]
    names = load_names(resolve_project_path(config["data"][split_key]))
    image_dir = resolve_project_path(config["data"]["image_dir"])
    label_dir = resolve_project_path(config["data"]["label_dir"])
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = DetectionPredictor(config_path, resolve_project_path(args.checkpoint), gpu=args.gpu)

    rendered = 0
    for name in names:
        if rendered >= args.limit:
            break
        if not has_label(label_dir / f"{name}.txt"):
            continue
        image_path = image_dir / f"{name}.png"
        if not image_path.exists():
            image_path = image_dir / f"{name}.jpg"
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        detections = predictor.predict(image, conf_threshold=args.conf_threshold)
        result = predictor.render(image, detections)
        result.save(output_dir / f"{rendered:03d}_{name}.jpg", quality=92)
        rendered += 1

    print(f"[Render] Saved {rendered} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
