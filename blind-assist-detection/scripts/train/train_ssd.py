"""Train MobileNetV2-SSD on the rebuilt SANPO obstacle dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import resolve_project_path
from src.datasets.detection_dataset import DetectionDataset, collate_fn, load_image_list
from src.losses.multibox_loss import MultiBoxLoss
from src.models.box_utils import cxcywh_to_xyxy, generate_anchors
from src.models.ssd_mobilenet import SSDMobileNetV2


def log(message: str) -> None:
    print(message, flush=True)


def load_config(path: str | Path) -> dict:
    with open(resolve_project_path(path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
    training_cfg = config["training"]
    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return optim.SGD(
        [
            {"params": backbone_params, "lr": training_cfg["learning_rate"] * 0.1},
            {"params": head_params, "lr": training_cfg["learning_rate"]},
        ],
        momentum=training_cfg["momentum"],
        weight_decay=training_cfg["weight_decay"],
    )


def ensure_output_dirs(output_root: Path, run_name: str | None) -> tuple[Path, Path, Path]:
    if run_name:
        run_dir = output_root / "runs" / run_name
        checkpoint_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        log_dir = run_dir / "logs"
    else:
        checkpoint_dir = output_root / "checkpoints"
        metrics_dir = output_root / "metrics"
        log_dir = output_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, metrics_dir, log_dir


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    best_loss: float,
    config: dict,
    run_args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "config": config,
        "run_args": vars(run_args),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    return start_epoch, best_loss


def load_model_weights(path: Path, *, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        log(f"[Train] missing keys while loading init weights: {missing_keys}")
    if unexpected_keys:
        log(f"[Train] unexpected keys while loading init weights: {unexpected_keys}")


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        return resume_path if resume_path.exists() else None
    if args.resume_last:
        resume_path = PROJECT_ROOT / "outputs" / "checkpoints" / "last.pth"
        return resume_path if resume_path.exists() else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/ssd_default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-last", action="store_true")
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Load model weights only, without optimizer/scheduler state. Use this for fine-tuning.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = resolve_project_path(args.output_root)
    checkpoint_dir, metrics_dir, log_dir = ensure_output_dirs(output_root, args.run_name)

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    log(f"[Train] device: {device}")

    data_cfg = config["data"]
    data_root = resolve_project_path(data_cfg["root"])
    image_dir = resolve_project_path(data_cfg.get("image_dir", data_root / "images"))
    label_dir = resolve_project_path(data_cfg.get("label_dir", data_root / "labels"))
    train_names = load_image_list(str(resolve_project_path(data_cfg["train_list"])))
    val_names = load_image_list(str(resolve_project_path(data_cfg["val_list"])))

    with open(resolve_project_path(data_cfg["classes_file"]), "r", encoding="utf-8") as file:
        classes = [line.strip() for line in file if line.strip()]

    input_size = config["model"]["input_size"][0]
    train_dataset = DetectionDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        image_list=train_names,
        input_size=input_size,
        classes=classes,
        augment=True,
    )
    val_dataset = DetectionDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        image_list=val_names,
        input_size=input_size,
        classes=classes,
        augment=False,
    )

    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    channels_last = device.type == "cuda" and not args.no_channels_last

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, **loader_kwargs)

    log(f"[Train] train samples: {len(train_names)}, val samples: {len(val_names)}")

    model = SSDMobileNetV2(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        input_size=input_size,
        backbone=config["model"].get("backbone", "mobilenet_v2"),
        use_eca=config["model"].get("use_eca", False),
        eca_stages=config["model"].get("eca_stages"),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    log(f"[Train] parameters: {total_params:,}")

    feature_maps = config["anchors"]["feature_maps"]
    min_sizes = config["anchors"]["min_sizes"]
    max_sizes = config["anchors"]["max_sizes"]
    aspect_ratios = config["anchors"]["aspect_ratios"]

    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        if channels_last:
            dummy = dummy.to(memory_format=torch.channels_last)
        dummy_conf, _ = model(dummy)
        actual_num_anchors = dummy_conf.size(1)

    anchors = generate_anchors(feature_maps, min_sizes, max_sizes, aspect_ratios).to(device)
    if anchors.size(0) != actual_num_anchors:
        log(
            f"[Train] anchor count mismatch: expected={anchors.size(0)} actual={actual_num_anchors}, fixing"
        )
        if anchors.size(0) > actual_num_anchors:
            anchors = anchors[:actual_num_anchors]
        else:
            pad = torch.zeros(actual_num_anchors - anchors.size(0), 4, device=device)
            anchors = torch.cat([anchors, pad], dim=0)
    anchors_xyxy = cxcywh_to_xyxy(anchors)
    log(f"[Train] anchors: {anchors.size(0)}")

    criterion = MultiBoxLoss(
        neg_pos_ratio=config["loss"]["neg_pos_ratio"],
        loc_weight=config["loss"]["loc_weight"],
        box_loss=config["loss"].get("box_loss", "smooth_l1"),
    )
    optimizer = build_optimizer(model, config)

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    freeze_epochs = config["training"]["freeze_backbone_epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_loss = float("inf")
    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        start_epoch, best_loss = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        log(f"[Train] resumed from {resume_path} at epoch {start_epoch}")
    elif args.init_weights:
        init_path = resolve_project_path(args.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"init weights not found: {init_path}")
        load_model_weights(init_path, model=model, device=device)
        log(f"[Train] initialized model weights from {init_path}")

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(str(log_dir))

    log("")
    log(f"[Train] epochs: {epochs}")
    log(f"[Train] freeze backbone epochs: {freeze_epochs}")
    log(f"[Train] amp: {use_amp}")
    log(f"[Train] batch_size: {batch_size}, num_workers: {num_workers}")
    log(f"[Train] channels_last: {channels_last}, val_every: {max(args.val_every, 1)}")
    log(f"[Train] checkpoint_dir: {checkpoint_dir}")
    log("=" * 60)

    for epoch in range(start_epoch, epochs):
        if epoch < freeze_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()

        model.train()
        train_loss = 0.0
        train_loc = 0.0
        train_conf = 0.0
        train_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            t_data_start = time.time()
            images = batch["images"].to(
                device=device,
                non_blocking=True,
                memory_format=torch.channels_last if channels_last else torch.contiguous_format,
            )
            gt_boxes = [tensor.to(device, non_blocking=True) for tensor in batch["boxes"]]
            gt_labels = [tensor.to(device, non_blocking=True) for tensor in batch["labels"]]
            data_time = time.time() - t_data_start

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    conf, loc = model(images)
                    loss, loc_loss, conf_loss = criterion(
                        conf, loc, gt_boxes, gt_labels, anchors, anchors_xyxy=anchors_xyxy
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                conf, loc = model(images)
                loss, loc_loss, conf_loss = criterion(
                    conf, loc, gt_boxes, gt_labels, anchors, anchors_xyxy=anchors_xyxy
                )
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_loc += loc_loss.item()
            train_conf += conf_loss.item()
            train_batches += 1

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log(
                    "  Epoch [{}/{}] Batch [{}/{}] Loss: {:.4f} "
                    "(loc:{:.4f} conf:{:.4f}) data:{:.3f}s".format(
                        epoch + 1,
                        epochs,
                        batch_idx + 1,
                        len(train_loader),
                        loss.item(),
                        loc_loss.item(),
                        conf_loss.item(),
                        data_time,
                    )
                )

        scheduler.step()

        if train_batches > 0:
            train_loss /= train_batches
            train_loc /= train_batches
            train_conf /= train_batches

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - epoch_start

        model.eval()
        val_loss = 0.0
        val_batches = 0
        run_val = ((epoch + 1) % max(args.val_every, 1) == 0) or (epoch + 1 == epochs)

        if run_val:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break

                    images = batch["images"].to(
                        device=device,
                        non_blocking=True,
                        memory_format=torch.channels_last if channels_last else torch.contiguous_format,
                    )
                    gt_boxes = [tensor.to(device, non_blocking=True) for tensor in batch["boxes"]]
                    gt_labels = [tensor.to(device, non_blocking=True) for tensor in batch["labels"]]

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            conf, loc = model(images)
                            loss, _, _ = criterion(
                                conf, loc, gt_boxes, gt_labels, anchors, anchors_xyxy=anchors_xyxy
                            )
                    else:
                        conf, loc = model(images)
                        loss, _, _ = criterion(
                            conf, loc, gt_boxes, gt_labels, anchors, anchors_xyxy=anchors_xyxy
                        )

                    val_loss += loss.item()
                    val_batches += 1

        if val_batches > 0:
            val_loss /= val_batches

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        has_val = val_batches > 0
        samples_per_sec = (train_batches * batch_size) / elapsed if elapsed > 0 else 0.0

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/loc", train_loc, epoch)
            writer.add_scalar("Loss/conf", train_conf, epoch)
            writer.add_scalar("LR/backbone", current_lrs[0], epoch)
            if len(current_lrs) > 1:
                writer.add_scalar("LR/head", current_lrs[1], epoch)
            writer.add_scalar("Perf/samples_per_sec", samples_per_sec, epoch)
            if has_val:
                writer.add_scalar("Loss/val", val_loss, epoch)

        log("")
        log(f"Epoch [{epoch + 1}/{epochs}] ({elapsed:.0f}s, {samples_per_sec:.1f} samples/s)")
        log(f"  Train Loss: {train_loss:.4f} (loc:{train_loc:.4f} conf:{train_conf:.4f})")
        if has_val:
            log(f"  Val   Loss: {val_loss:.4f}")
        elif run_val:
            log("  Val   Loss: N/A (no validation batches)")
        else:
            log(f"  Val   Loss: skipped (val_every={max(args.val_every, 1)})")

        summary_payload = {
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "train_loc_loss": train_loc,
            "train_conf_loss": train_conf,
            "val_loss": val_loss if has_val else None,
            "ran_validation": run_val,
            "samples_per_sec": samples_per_sec,
            "best_loss": best_loss if best_loss < float("inf") else None,
            "learning_rates": current_lrs,
            "elapsed_seconds": elapsed,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(metrics_dir / "train_status.json", summary_payload)

        if has_val and val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_loss=best_loss,
                config=config,
                run_args=args,
            )
            log(f"  -> saved best checkpoint (loss={best_loss:.4f})")

        save_checkpoint(
            checkpoint_dir / "last.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_loss=best_loss,
            config=config,
            run_args=args,
        )

    if writer is not None:
        writer.close()

    if best_loss < float("inf"):
        log(f"\n[Train Complete] best val loss: {best_loss:.4f}")
    else:
        log("\n[Train Complete] no validation result was produced")


if __name__ == "__main__":
    main()
