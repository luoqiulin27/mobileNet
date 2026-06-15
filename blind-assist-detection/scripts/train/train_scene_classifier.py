from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scene_dataset import SceneDataset


def build_model(num_classes: int = 2) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train indoor/outdoor scene classifier")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "scene_indoor_outdoor"))
    parser.add_argument("--run-name", default="scene_indoor_outdoor")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = PROJECT_ROOT / "outputs" / "runs" / args.run_name
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = SceneDataset(data_dir / "meta" / "train.txt", augment=True)
    val_dataset = SceneDataset(data_dir / "meta" / "val.txt", augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc = 0.0
    history = []
    print(f"[Scene] device={device}, train={len(train_dataset)}, val={len(val_dataset)}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(
            f"[Scene] epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": max(best_acc, val_acc),
            "classes": ["indoor", "outdoor"],
            "image_size": 224,
            "history": history,
        }
        torch.save(payload, checkpoint_dir / "last.pth")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(payload, checkpoint_dir / "best.pth")

        (metrics_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[Scene] complete best_val_acc={best_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
