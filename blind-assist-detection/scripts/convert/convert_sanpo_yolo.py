"""
SANPO-YOLO 数据转换
将 SANPO-Real-YOLO-obstacles 转换为统一格式并划分 train/val/test
"""
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 原始类别 (10类) → 统一类别 (8类)
# obstacle=0, vehicle=1, pedestrian=2, rider=3, animal=4, stairs=5, pole=6, bike_rack=7
# 映射到统一标签:
#   obstacle(0) → obstacle(6)
#   vehicle(1) → vehicle(1)
#   pedestrian(2) → pedestrian(0)
#   rider(3) → rider(2)
#   animal(4) → animal(3)
#   stairs(5) → stairs(4)
#   traffic_sign(6) → (忽略)
#   traffic_light(7) → (忽略)
#   pole(8) → pole(5)
#   bike_rack(9) → obstacle(6)

YOLO_TO_UNIFIED = {
    0: 6,   # obstacle → obstacle
    1: 1,   # vehicle → vehicle
    2: 0,   # pedestrian → pedestrian
    3: 2,   # rider → rider
    4: 3,   # animal → animal
    5: 4,   # stairs → stairs
    6: -1,  # traffic_sign → 忽略
    7: -1,  # traffic_light → 忽略
    8: 5,   # pole → pole
    9: 6,   # bike_rack → obstacle
}

UNIFIED_CLASSES = [
    "pedestrian",   # 0
    "vehicle",      # 1
    "rider",        # 2
    "animal",       # 3
    "stairs",       # 4
    "pole",         # 5
    "obstacle",     # 6
    "furniture",    # 7
]


def convert_and_split(
    src_root: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    转换 SANPO-YOLO 数据并划分

    Args:
        src_root: SANPO-Real-YOLO-obstacles 根目录
        output_dir: 输出目录 (data/processed)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    src_root = Path(src_root)
    src_images = src_root / "images"
    src_labels = src_root / "labels"

    out_dir = Path(output_dir)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # 收集所有样本
    sessions = sorted([d.name for d in src_labels.iterdir() if d.is_dir()])
    all_samples = []  # (session, frame_name)

    print(f"[Convert] 找到 {len(sessions)} 个 session")

    converted = 0
    skipped = 0

    for session in sessions:
        session_label_dir = src_labels / session
        session_img_dir = src_images / session

        label_files = sorted(session_label_dir.glob("*.txt"))
        for label_path in label_files:
            frame_name = label_path.stem

            # 找对应图像
            img_path = session_img_dir / f"{frame_name}.png"
            if not img_path.exists():
                skipped += 1
                continue

            # 转换标签
            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_cls = int(parts[0])
                    new_cls = YOLO_TO_UNIFIED.get(old_cls, -1)
                    if new_cls < 0:
                        continue
                    new_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            if not new_lines:
                skipped += 1
                continue

            # 复制图像
            dst_img = out_images / f"{frame_name}.png"
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # 写标签
            dst_label = out_labels / f"{frame_name}.txt"
            with open(dst_label, "w") as f:
                f.write("\n".join(new_lines) + "\n")

            all_samples.append(frame_name)
            converted += 1

    print(f"[Convert] 转换完成: {converted} 成功, {skipped} 跳过")

    # 按 session 划分 (防止同一序列跨集合)
    session_samples = {}
    for name in all_samples:
        # 从 frame_name 提取 session (格式: sessionID_000000)
        parts = name.rsplit("_", 1)
        session = parts[0] if len(parts) > 1 else name
        if session not in session_samples:
            session_samples[session] = []
        session_samples[session].append(name)

    session_keys = list(session_samples.keys())
    random.shuffle(session_keys)

    total = len(all_samples)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    train_names = []
    val_names = []
    test_names = []

    for key in session_keys:
        samples = session_samples[key]
        if len(train_names) < train_target:
            train_names.extend(samples)
        elif len(val_names) < val_target:
            val_names.extend(samples)
        else:
            test_names.extend(samples)

    # 写列表文件
    for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        list_path = out_dir / f"{split_name}.txt"
        with open(list_path, "w") as f:
            for name in names:
                f.write(f"{name}\n")
        print(f"[{split_name}] {len(names)} 样本")

    # 写类别文件
    classes_path = out_dir / "configs" / "classes.txt"
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_path, "w") as f:
        for cls in UNIFIED_CLASSES:
            f.write(f"{cls}\n")

    print(f"\n[完成] 输出到 {out_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default=str(PROJECT_ROOT / "data" / "raw" / "SANPO-Real-YOLO-obstacles"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_and_split(args.src, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
