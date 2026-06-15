from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def list_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def write_manifest(path: Path, samples: list[tuple[Path, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for image_path, label in samples:
            file.write(f"{image_path}\t{label}\n")


def split_samples(
    samples: list[tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build indoor/outdoor scene classification manifests")
    parser.add_argument("--indoor-dir", default=str(PROJECT_ROOT / "data" / "sunrgbd_indoor_12class" / "images" / "all"))
    parser.add_argument("--outdoor-dir", default=str(PROJECT_ROOT / "data" / "sanpo_obstacle_8class" / "images" / "all"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "scene_indoor_outdoor"))
    parser.add_argument("--max-per-class", type=int, default=7200)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    indoor_dir = Path(args.indoor_dir)
    outdoor_dir = Path(args.outdoor_dir)
    output_dir = Path(args.output)
    meta_dir = output_dir / "meta"
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    indoor_images = list_images(indoor_dir)
    outdoor_images = list_images(outdoor_dir)
    if not indoor_images:
        raise SystemExit(f"No indoor images found: {indoor_dir}")
    if not outdoor_images:
        raise SystemExit(f"No outdoor images found: {outdoor_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(indoor_images)
    rng.shuffle(outdoor_images)

    per_class = min(args.max_per_class, len(indoor_images), len(outdoor_images))
    indoor_samples = [(path, 0) for path in indoor_images[:per_class]]
    outdoor_samples = [(path, 1) for path in outdoor_images[:per_class]]
    rng.shuffle(indoor_samples)
    rng.shuffle(outdoor_samples)

    indoor_train, indoor_val, indoor_test = split_samples(indoor_samples, args.train_ratio, args.val_ratio)
    outdoor_train, outdoor_val, outdoor_test = split_samples(outdoor_samples, args.train_ratio, args.val_ratio)

    train = indoor_train + outdoor_train
    val = indoor_val + outdoor_val
    test = indoor_test + outdoor_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    write_manifest(meta_dir / "train.txt", train)
    write_manifest(meta_dir / "val.txt", val)
    write_manifest(meta_dir / "test.txt", test)
    (config_dir / "classes.txt").write_text("indoor\noutdoor\n", encoding="utf-8")

    report = {
        "indoor_dir": str(indoor_dir),
        "outdoor_dir": str(outdoor_dir),
        "indoor_available": len(indoor_images),
        "outdoor_available": len(outdoor_images),
        "per_class_used": per_class,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": args.seed,
    }
    (meta_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
