"""
convert_sanpo_phase1.py

从 SANPO-Real-Labeled-Full 原始全景分割 mask 中提取 5 类 bbox，
转换为 YOLO 格式标签，复制对应图像到 phase1 数据目录。

第一阶段 5 类映射:
  pedestrian(12) -> person(0)
  vehicle(21)    -> vehicle(1)
  pole(24)       -> pole(2)
  stairs(15)     -> stairs(3)
  obstacle(20)   -> obstacle(4)
  bike rack(26)  -> obstacle(4)
  animal(14)     -> obstacle(4)
  rider(13)      -> obstacle(4)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sanpo_profiles import get_sanpo_profile

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

def reset_output_dir(output_dir: Path) -> None:
    """Remove stale converted data so images/labels/meta stay in sync."""
    for name in ["images", "labels", "meta"]:
        target = output_dir / name
        if target.exists():
            shutil.rmtree(target)


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def get_active_sessions(dataset_dir: Path) -> list[str]:
    """扫描 images/ 目录，返回有实际 .png 文件的 session_id 列表。"""
    images_dir = dataset_dir / "images"
    active = []
    for session_dir in sorted(images_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        # 检查是否有至少一个 .png 文件
        pngs = list(session_dir.glob("*.png"))
        if pngs:
            active.append(session_dir.name)
    return active


def extract_boxes_from_mask(
    mask_path: Path,
    class_mapping: dict[int, int],
    min_area: int,
    img_w: int,
    img_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    从单个 mask 文件中提取所有目标 bbox。

    返回: [(class_id, cx, cy, w, h), ...]  所有值已归一化到 [0,1]
    """
    try:
        mask_img = Image.open(mask_path).convert("RGB")
    except Exception:
        return []

    mask = np.asarray(mask_img)
    semantic = mask[:, :, 0].astype(np.int32)  # R 通道 = 语义类别 ID
    instance = mask[:, :, 2].astype(np.int32)  # B 通道 = 实例 ID

    boxes: list[tuple[int, float, float, float, float]] = []

    for sanpo_id, phase1_id in class_mapping.items():
        # 找到该类别的所有像素
        class_pixels = semantic == sanpo_id
        if not class_pixels.any():
            continue

        # 提取该区域内的所有唯一 instance_id
        instance_ids = np.unique(instance[class_pixels])

        for iid in instance_ids:
            # 创建实例掩码
            component = class_pixels & (instance == iid)
            area = int(component.sum())
            if area < min_area:
                continue

            # 找到像素坐标
            ys, xs = np.where(component)
            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max()) + 1  # +1 因为 max 是闭区间
            y_max = int(ys.max()) + 1

            # 转换为 YOLO 归一化格式
            cx = (x_min + x_max) / 2.0 / img_w
            cy = (y_min + y_max) / 2.0 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h

            # clip 到 [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            # 过滤无效框 (与 DetectionDataset 的过滤逻辑一致)
            if w <= 0.01 or h <= 0.01:
                continue

            boxes.append((phase1_id, cx, cy, w, h))

    return boxes


def process_session(
    session_id: str,
    dataset_dir: Path,
    output_dir: Path,
    class_mapping: dict[int, int],
    min_area: int,
    copy_images: bool,
    limit_frames: int = 0,
) -> tuple[int, int, int, dict[int, int]]:
    """
    处理单个 session 的所有帧。

    返回: (total_frames, frames_with_boxes, total_boxes, class_box_counts)
    """
    img_session_dir = dataset_dir / "images" / session_id
    mask_session_dir = dataset_dir / "labels_segmentation_masks" / session_id

    out_img_dir = output_dir / "images" / "all"
    out_lbl_dir = output_dir / "labels" / "all"

    frame_files = sorted(img_session_dir.glob("*.png"))
    if limit_frames > 0:
        frame_files = frame_files[:limit_frames]

    total_frames = 0
    frames_with_boxes = 0
    total_boxes = 0
    class_box_counts: dict[int, int] = {}

    for frame_path in frame_files:
        frame_name = frame_path.stem  # e.g. "000000"
        mask_path = mask_session_dir / f"{frame_name}.png"

        total_frames += 1

        # 检查 mask 是否存在
        if not mask_path.exists():
            # 写入空标签 (图像仍复制)
            out_name = f"{session_id}_{frame_name}"
            if copy_images:
                dst_img = out_img_dir / f"{out_name}.png"
                if not dst_img.exists():
                    shutil.copy2(frame_path, dst_img)
            # 写空标签
            (out_lbl_dir / f"{out_name}.txt").write_text("", encoding="utf-8")
            continue

        # 读取图像尺寸
        try:
            with Image.open(frame_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        # 提取 bbox
        boxes = extract_boxes_from_mask(mask_path, class_mapping, min_area, img_w, img_h)

        # 写标签
        out_name = f"{session_id}_{frame_name}"
        if boxes:
            lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cid, cx, cy, w, h in boxes]
            (out_lbl_dir / f"{out_name}.txt").write_text("".join(lines), encoding="utf-8")
            frames_with_boxes += 1
            total_boxes += len(boxes)
            for cid, _, _, _, _ in boxes:
                class_box_counts[cid] = class_box_counts.get(cid, 0) + 1
        else:
            (out_lbl_dir / f"{out_name}.txt").write_text("", encoding="utf-8")

        # 复制图像
        if copy_images:
            dst_img = out_img_dir / f"{out_name}.png"
            if not dst_img.exists():
                shutil.copy2(frame_path, dst_img)

    return total_frames, frames_with_boxes, total_boxes, class_box_counts


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SANPO masks to YOLO labels")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "SANPO-Real-Labeled-Full"),
        help="SANPO 原始数据根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument("--profile", type=str, default="sanpo_obstacle_8class", help="SANPO 类别映射方案")
    parser.add_argument("--min-area", type=int, default=200, help="bbox 最小像素面积阈值")
    parser.add_argument("--copy-images", action="store_true", default=True, help="是否复制图像文件")
    parser.add_argument("--no-copy-images", action="store_false", dest="copy_images", help="不复制图像文件")
    parser.add_argument("--clean-output", action="store_true", help="转换前清空旧的输出目录")
    parser.add_argument("--limit-sessions", type=int, default=0, help="限制处理的 session 数量（0=全部，调试用）")
    parser.add_argument("--limit-frames", type=int, default=0, help="限制每个 session 处理的帧数（0=全部，调试用）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    profile = get_sanpo_profile(args.profile)
    output_dir = Path(args.output) if args.output else (PROJECT_ROOT / "data" / profile.output_dir_name)
    min_area = args.min_area
    copy_images = args.copy_images
    limit_sessions = args.limit_sessions
    limit_frames = args.limit_frames
    class_mapping = profile.mask_to_class
    class_names = list(profile.classes)
    class_name_map = {idx: name for idx, name in enumerate(class_names)}

    # 检查输入目录
    if not dataset_dir.exists():
        print(f"[ERROR] 数据集目录不存在: {dataset_dir}")
        return 1

    if args.clean_output:
        print(f"[Convert] 清理旧输出: {output_dir}")
        reset_output_dir(output_dir)

    # 创建输出目录
    out_img_dir = output_dir / "images" / "all"
    out_lbl_dir = output_dir / "labels" / "all"
    out_cfg_dir = output_dir / "configs"
    out_meta_dir = output_dir / "meta"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # 写 classes.txt
    classes_path = out_cfg_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"[Convert] 已写入 {classes_path}")

    # 扫描有效 session
    print("[Convert] 扫描有效 session...")
    active_sessions = get_active_sessions(dataset_dir)
    total_session_dirs = sum(1 for d in (dataset_dir / "images").iterdir() if d.is_dir())
    empty_sessions = total_session_dirs - len(active_sessions)
    print(f"[Convert] 找到 {len(active_sessions)} 个有效 session（{total_session_dirs} 个目录中有 {empty_sessions} 个为空）")

    # 处理所有 session
    t_start = time.time()
    grand_total_frames = 0
    grand_frames_with_boxes = 0
    grand_total_boxes = 0
    grand_class_counts: dict[int, int] = {}

    sessions_to_process = active_sessions
    if limit_sessions > 0:
        sessions_to_process = active_sessions[:limit_sessions]
        print(f"[Convert] 调试模式: 仅处理前 {limit_sessions} 个 session")

    for idx, session_id in enumerate(sessions_to_process, 1):
        total_frames, frames_with_boxes, total_boxes, class_counts = process_session(
            session_id=session_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            class_mapping=class_mapping,
            min_area=min_area,
            copy_images=copy_images,
            limit_frames=limit_frames,
        )
        grand_total_frames += total_frames
        grand_frames_with_boxes += frames_with_boxes
        grand_total_boxes += total_boxes
        for cid, count in class_counts.items():
            grand_class_counts[cid] = grand_class_counts.get(cid, 0) + count

        if idx % 10 == 0 or idx == len(sessions_to_process):
            elapsed = time.time() - t_start
            print(f"[Convert] Session {idx}/{len(sessions_to_process)}: {session_id[:20]}... "
                  f"→ {total_frames} 帧, {total_boxes} 个 bbox ({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start

    # 打印最终统计
    print("=" * 60)
    print(f"[Convert] 完成！耗时 {elapsed_total:.1f}s")
    print(f"[Convert] 总帧数: {grand_total_frames}")
    print(f"[Convert] 有标注帧数: {grand_frames_with_boxes}")
    print(f"[Convert] 无标注帧数: {grand_total_frames - grand_frames_with_boxes}")
    print(f"[Convert] 总 bbox 数: {grand_total_boxes}")
    print("[Convert] 类别分布:")
    for cid in range(len(class_names)):
        count = grand_class_counts.get(cid, 0)
        print(f"  {class_name_map[cid]}: {count}")

    # 生成 conversion_report.json
    report = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "min_area": min_area,
        "copy_images": copy_images,
        "sessions": {
            "total_dirs": total_session_dirs,
            "active": len(active_sessions),
            "empty": empty_sessions,
        },
        "frames": {
            "total": grand_total_frames,
            "with_boxes": grand_frames_with_boxes,
            "without_boxes": grand_total_frames - grand_frames_with_boxes,
        },
        "bboxes": {
            "total": grand_total_boxes,
            "by_class": {class_name_map[cid]: grand_class_counts.get(cid, 0) for cid in range(len(class_names))},
        },
        "profile": profile.name,
        "class_mapping": {str(k): class_name_map[v] for k, v in class_mapping.items()},
        "elapsed_seconds": round(elapsed_total, 1),
    }
    report_path = out_meta_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Convert] 已写入 {report_path}")

    print("[Convert] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
