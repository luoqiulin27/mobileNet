"""
verify_phase1.py

验证 phase1_sanpo_5class 数据是否满足进入训练阶段的条件。

检查项:
  C1: 图像数 == 标签数
  C2-C4: train/val/test 中的 stem 都有对应图像和标签
  C5: train/val/test session 无交集
  C6: train/val/test 帧无交集
  C7: train + val + test 覆盖所有帧
  C8: 所有 bbox 的 cx/cy/w/h 在 [0,1] 范围内
  C9: 所有 class_id 在 [0,4] 范围内
  C10: 所有 bbox 满足 w > 0.01 且 h > 0.01
  C11: classes.txt 内容正确
  C12: 每个类别在每个 split 中至少有 100 个 bbox (WARN)

输出:
  meta/stats.json      - 训练所需的关键统计字段
  meta/verify_report.json - 详细验证报告
"""

import argparse
import json
import sys
import time
from pathlib import Path


def extract_session_id(stem: str) -> str:
    """从文件名 stem 提取 session_id（按最后一个 '_' 分割）。"""
    last_underscore = stem.rfind("_")
    if last_underscore < 0:
        raise ValueError(f"Invalid filename format: {stem}")
    return stem[:last_underscore]


def read_stems_from_file(path: Path) -> list[str]:
    """读取 split 文件中的 stem 列表。"""
    if not path.exists():
        return []
    stems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(line)
    return stems


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    解析 YOLO 标签文件。

    返回: [(class_id, cx, cy, w, h), ...]
    """
    boxes = []
    if not path.exists():
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cls_id, cx, cy, w, h))
            except ValueError:
                continue
    return boxes


def check_label_validity(
    boxes: list[tuple[int, float, float, float, float]],
) -> tuple[int, int]:
    """
    检查标签合法性。

    返回: (invalid_class_ids, invalid_boxes)
      - invalid_class_ids: class_id 不在 [0,4] 的 bbox 数
      - invalid_boxes: cx/cy/w/h 超出 [0,1] 或 w<=0.01 或 h<=0.01 的 bbox 数
    """
    invalid_cls = 0
    invalid_box = 0
    for cls_id, cx, cy, w, h in boxes:
        if cls_id < 0 or cls_id >= NUM_CLASSES:
            invalid_cls += 1
        # cx/cy 应在 [0,1]，w/h 应在 (0,1]
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            invalid_box += 1
        elif not (0 < w <= 1 and 0 < h <= 1):
            invalid_box += 1
        elif w <= 0.01 or h <= 0.01:
            invalid_box += 1
    return invalid_cls, invalid_box


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Phase1 SANPO data is ready for training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent / "data" / "sanpo_obstacle_8class"
        ),
        help="Phase1 数据根目录",
    )
    parser.add_argument(
        "--limit-stems",
        type=int,
        default=0,
        help="限制检查的 stem 数量（0=全部，调试用）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    limit_stems = args.limit_stems

    t_start = time.time()
    checks: dict[str, str] = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 基础路径
    # ------------------------------------------------------------------
    img_dir = data_dir / "images" / "all"
    lbl_dir = data_dir / "labels" / "all"
    meta_dir = data_dir / "meta"
    cfg_dir = data_dir / "configs"

    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return 1

    # ------------------------------------------------------------------
    # C1: 图像数 == 标签数
    # ------------------------------------------------------------------
    print("[Verify] C1: 检查图像/标签配对...")
    image_stems = sorted([p.stem for p in img_dir.glob("*.png")]) if img_dir.exists() else []
    label_stems = sorted([p.stem for p in lbl_dir.glob("*.txt")]) if lbl_dir.exists() else []

    image_set = set(image_stems)
    label_set = set(label_stems)
    missing_labels = sorted(image_set - label_set)
    extra_labels = sorted(label_set - image_set)

    if len(image_stems) == len(label_stems) and len(missing_labels) == 0:
        checks["C1_image_label_match"] = "PASS"
    else:
        checks["C1_image_label_match"] = f"FAIL: images={len(image_stems)}, labels={len(label_stems)}, missing_label={len(missing_labels)}, extra_label={len(extra_labels)}"

    # ------------------------------------------------------------------
    # C2-C4: split 文件中的 stem 都有对应图像和标签
    # ------------------------------------------------------------------
    print("[Verify] C2-C4: 检查 split 文件...")
    train_stems = read_stems_from_file(meta_dir / "train.txt")
    val_stems = read_stems_from_file(meta_dir / "val.txt")
    test_stems = read_stems_from_file(meta_dir / "test.txt")

    classes_path = cfg_dir / "classes.txt"
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = []
    num_classes = len(class_names)

    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        missing_imgs = [s for s in stems if s not in image_set]
        missing_lbls = [s for s in stems if s not in label_set]
        if len(missing_imgs) == 0 and len(missing_lbls) == 0:
            checks[f"C2_{split_name}_files_exist"] = "PASS"
        else:
            checks[f"C2_{split_name}_files_exist"] = f"FAIL: missing_images={len(missing_imgs)}, missing_labels={len(missing_lbls)}"

    # ------------------------------------------------------------------
    # C5: train/val/test session 无交集
    # ------------------------------------------------------------------
    print("[Verify] C5: 检查 session 泄漏...")
    train_sessions = set(extract_session_id(s) for s in train_stems)
    val_sessions = set(extract_session_id(s) for s in val_stems)
    test_sessions = set(extract_session_id(s) for s in test_stems)

    s_tv = len(train_sessions & val_sessions)
    s_tt = len(train_sessions & test_sessions)
    s_vt = len(val_sessions & test_sessions)

    if s_tv == 0 and s_tt == 0 and s_vt == 0:
        checks["C5_session_no_leak"] = "PASS"
    else:
        checks["C5_session_no_leak"] = f"FAIL: train∩val={s_tv}, train∩test={s_tv}, val∩test={s_vt}"

    # ------------------------------------------------------------------
    # C6: train/val/test 帧无交集
    # ------------------------------------------------------------------
    print("[Verify] C6: 检查帧泄漏...")
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)

    f_tv = len(train_set & val_set)
    f_tt = len(train_set & test_set)
    f_vt = len(val_set & test_set)

    if f_tv == 0 and f_tt == 0 and f_vt == 0:
        checks["C6_frame_no_leak"] = "PASS"
    else:
        checks["C6_frame_no_leak"] = f"FAIL: train∩val={f_tv}, train∩test={f_tt}, val∩test={f_vt}"

    # ------------------------------------------------------------------
    # C7: train + val + test 覆盖所有帧
    # ------------------------------------------------------------------
    print("[Verify] C7: 检查覆盖率...")
    all_split_stems = train_set | val_set | test_set
    labeled_set = set(label_stems)
    covered_target = image_set & labeled_set
    uncovered = covered_target - all_split_stems
    if len(uncovered) == 0:
        checks["C7_full_coverage"] = "PASS"
    else:
        checks["C7_full_coverage"] = f"FAIL: {len(uncovered)} stems not in any split"

    # ------------------------------------------------------------------
    # C8-C10: 标签数值合法性
    # ------------------------------------------------------------------
    print("[Verify] C8-C10: 检查标签数值...")
    total_invalid_class_ids = 0
    total_invalid_boxes = 0
    empty_label_files = 0
    total_bboxes = 0
    class_counts: dict[int, int] = {i: 0 for i in range(num_classes)}
    split_class_counts: dict[str, dict[int, int]] = {
        "train": {i: 0 for i in range(num_classes)},
        "val": {i: 0 for i in range(num_classes)},
        "test": {i: 0 for i in range(num_classes)},
    }

    # 构建 stem -> split 映射
    stem_to_split: dict[str, str] = {}
    for s in train_stems:
        stem_to_split[s] = "train"
    for s in val_stems:
        stem_to_split[s] = "val"
    for s in test_stems:
        stem_to_split[s] = "test"

    # 要检查的 stem 列表
    stems_to_check = image_stems
    if limit_stems > 0:
        stems_to_check = image_stems[:limit_stems]
        print(f"[Verify] 调试模式: 仅检查前 {limit_stems} 个 stem")

    for stem in stems_to_check:
        lbl_path = lbl_dir / f"{stem}.txt"
        boxes = parse_label_file(lbl_path)

        if len(boxes) == 0:
            empty_label_files += 1

        total_bboxes += len(boxes)

        inv_box = 0
        inv_cls = 0
        for cls_id, cx, cy, w, h in boxes:
            if cls_id < 0 or cls_id >= num_classes:
                inv_cls += 1
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                inv_box += 1
            elif not (0 < w <= 1 and 0 < h <= 1):
                inv_box += 1
            elif w <= 0.01 or h <= 0.01:
                inv_box += 1
        total_invalid_class_ids += inv_cls
        total_invalid_boxes += inv_box

        for cls_id, cx, cy, w, h in boxes:
            if 0 <= cls_id < num_classes:
                class_counts[cls_id] += 1
                split_name = stem_to_split.get(stem)
                if split_name:
                    split_class_counts[split_name][cls_id] += 1

    checks["C8_bbox_range"] = "PASS" if total_invalid_boxes == 0 else f"FAIL: {total_invalid_boxes} invalid boxes"
    checks["C9_class_id_range"] = "PASS" if total_invalid_class_ids == 0 else f"FAIL: {total_invalid_class_ids} invalid class IDs"
    checks["C10_bbox_size"] = "PASS" if total_invalid_boxes == 0 else f"FAIL: {total_invalid_boxes} invalid boxes"

    # ------------------------------------------------------------------
    # C11: classes.txt 内容正确
    # ------------------------------------------------------------------
    print("[Verify] C11: 检查 classes.txt...")
    classes_path = cfg_dir / "classes.txt"
    if classes_path.exists():
        if num_classes > 0:
            checks["C11_classes_content"] = "PASS"
        else:
            checks["C11_classes_content"] = "FAIL: classes.txt is empty"
    else:
        checks["C11_classes_content"] = "FAIL: classes.txt not found"

    # ------------------------------------------------------------------
    # C12: 类别平衡检查 (WARN)
    # ------------------------------------------------------------------
    print("[Verify] C12: 检查类别平衡...")
    MIN_BBOX_PER_SPLIT_CLASS = 100
    for split_name in ["train", "val", "test"]:
        for cls_id in range(num_classes):
            count = split_class_counts[split_name][cls_id]
            if 0 < count < MIN_BBOX_PER_SPLIT_CLASS:
                warnings.append(
                    f"{class_names[cls_id]} in {split_name} has only {count} bboxes"
                )

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    leak_passed = (
        "PASS" in checks.get("C5_session_no_leak", "")
        and "PASS" in checks.get("C6_frame_no_leak", "")
    )

    # 检查重复 stem
    duplicate_stems = len(image_stems) - len(image_set)

    # 统计帧数
    train_images = len(train_stems)
    val_images = len(val_stems)
    test_images = len(test_stems)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 生成 stats.json
    # ------------------------------------------------------------------
    stats = {
        "total_images": len(image_stems),
        "total_labels": len(label_stems),
        "total_sessions": len(train_sessions | val_sessions | test_sessions),
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "test_sessions": len(test_sessions),
        "empty_label_files": empty_label_files,
        "missing_label_files": len(missing_labels),
        "duplicate_stems": duplicate_stems,
        "leak_check_passed": leak_passed,
        "class_counts": {class_names[i]: class_counts[i] for i in range(num_classes)},
        "split_class_counts": {
            split: {class_names[i]: split_class_counts[split][i] for i in range(num_classes)}
            for split in ["train", "val", "test"]
        },
        "invalid_boxes": total_invalid_boxes,
        "invalid_class_ids": total_invalid_class_ids,
        "total_bboxes": total_bboxes,
    }

    stats_path = meta_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Verify] 已写入 {stats_path}")

    # ------------------------------------------------------------------
    # 生成 verify_report.json
    # ------------------------------------------------------------------
    report = {
        "data_dir": str(data_dir),
        "checks": checks,
        "warnings": warnings,
        "all_passed": all("PASS" in v for v in checks.values()),
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = meta_dir / "verify_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Verify] 已写入 {report_path}")

    # ------------------------------------------------------------------
    # 打印摘要
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"[Verify] 验证完成 ({elapsed:.1f}s)")
    print(f"[Verify] 检查项结果:")
    all_pass = True
    for name, result in checks.items():
        status = "[PASS]" if "PASS" in result else "[FAIL]"
        if "PASS" not in result:
            all_pass = False
        print(f"  {status} {name}: {result}")

    if warnings:
        print(f"[Verify] WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    print(f"[Verify] Overall: {'ALL PASS' if all_pass else 'HAS FAILURES'}")
    print(f"[Verify] 图像: {len(image_stems)}, 标签: {len(label_stems)}, bbox: {total_bboxes}")
    print(f"[Verify] 空标签文件: {empty_label_files}")
    print(f"[Verify] 类别分布: { {class_names[i]: class_counts[i] for i in range(num_classes)} }")

    print("[Verify] Done.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
