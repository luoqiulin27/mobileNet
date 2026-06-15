# 第一阶段 SANPO 五类检测 baseline — 接口设计说明书

> 本文档为 `convert_sanpo_phase1.py`、`split_phase1.py`、`verify_phase1.py` 三个脚本及配套配置修改提供精确的接口规范。
> 编码时必须严格遵循本文档定义，不允许擅自发挥。

---

## 1. `convert_sanpo_phase1.py` 接口设计

### 1.1 职责

从 SANPO 原始全景分割 mask 中提取 5 类 bbox，转换为 YOLO 格式标签文件，并复制对应图像到输出目录。

**不做：** 不做数据划分、不生成 train/val/test 列表、不生成 stats 统计。

### 1.2 输入与输出

| 项目 | 路径 | 说明 |
|------|------|------|
| 输入 - 图像 | `data/SANPO-Real-Labeled-Full/images/<session_id>/NNNNNN.png` | 原始 RGB 图像 |
| 输入 - mask | `data/SANPO-Real-Labeled-Full/labels_segmentation_masks/<session_id>/NNNNNN.png` | 全景分割 mask |
| 输入 - labelmap | `data/SANPO-Real-Labeled-Full/metadata/labelmap.json` | 类名→ID 映射 |
| 输出 - 图像 | `data/phase1_sanpo_5class/images/<session_id>_NNNNNN.png` | 复制的图像，文件名扁平化 |
| 输出 - 标签 | `data/phase1_sanpo_5class/labels/<session_id>_NNNNNN.txt` | YOLO 格式标签 |
| 输出 - classes | `data/phase1_sanpo_5class/configs/classes.txt` | 5 类名列表 |

### 1.3 命令行参数

```
python convert_sanpo_phase1.py \
    --dataset data/SANPO-Real-Labeled-Full \
    --output data/phase1_sanpo_5class \
    --min-area 200 \
    --copy-images \
    --workers 4
```

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--dataset` | str | `data/SANPO-Real-Labeled-Full` | SANPO 原始数据根目录 |
| `--output` | str | `data/phase1_sanpo_5class` | 输出目录 |
| `--min-area` | int | `200` | bbox 最小像素面积阈值，低于此值的实例被丢弃 |
| `--copy-images` | flag | False | 是否复制图像文件（首次运行必须开启） |
| `--workers` | int | `4` | 并行处理的 worker 数 |

### 1.4 核心常量

```python
# SANPO mask R 通道值 → 5 类 ID 的映射
# 不在此映射中的 mask 值全部跳过
SANPO_MASK_TO_PHASE1 = {
    12: 0,   # pedestrian → person
    21: 1,   # vehicle → vehicle
    24: 2,   # pole → pole
    15: 3,   # stairs → stairs
    20: 4,   # obstacle → obstacle
    26: 4,   # bike rack → obstacle
    14: 4,   # animal → obstacle
    13: 4,   # rider → obstacle
}

PHASE1_CLASSES = ["person", "vehicle", "pole", "stairs", "obstacle"]
```

### 1.5 主要函数列表

#### `load_labelmap(path: str) -> dict[str, int]`

- **职责：** 读取 labelmap.json，返回 `{类名: mask_id}` 字典
- **输入：** labelmap.json 文件路径
- **输出：** `{"unlabeled": 0, "road": 1, ..., "vehicle": 21, ...}`

#### `get_active_sessions(dataset_dir: Path) -> list[str]`

- **职责：** 扫描 images/ 目录，返回有实际图像文件的 session_id 列表（过滤空目录）
- **输入：** 数据集根目录
- **输出：** `["-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG", ...]`
- **过滤规则：** session 目录下必须有至少一个 .png 文件

#### `extract_boxes_from_mask(mask_path: Path, class_mapping: dict[int, int], min_area: int, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]`

- **职责：** 从单个 mask 文件中提取所有目标 bbox
- **输入：**
  - `mask_path`: mask PNG 文件路径
  - `class_mapping`: `{sanpo_mask_id: phase1_class_id}` 映射表
  - `min_area`: 最小面积阈值
  - `img_w`, `img_h`: 图像宽高（用于归一化）
- **输出：** `[(class_id, cx, cy, w, h), ...]`，所有值已归一化到 [0,1]
- **处理逻辑：**
  1. 读取 mask，分离 R 通道（semantic）和 B 通道（instance）
  2. 对 class_mapping 中的每个 sanpo_mask_id：
     - 创建二值掩码 `semantic == sanpo_mask_id`
     - 如果无像素，跳过
     - 提取该区域内所有唯一 instance_id
     - 对每个 instance_id：
       - 创建实例掩码 `semantic == sanpo_mask_id AND instance == instance_id`
       - 计算像素面积，如果 < min_area 则跳过
       - 找到所有像素的 (y, x) 坐标
       - 计算外接矩形：`xmin = min(xs)`, `ymin = min(ys)`, `xmax = max(xs) + 1`, `ymax = max(ys) + 1`
       - 转换为 YOLO 格式：`cx = (xmin + xmax) / 2 / img_w`, `cy = (ymin + ymax) / 2 / img_h`, `w = (xmax - xmin) / img_w`, `h = (ymax - ymin) / img_h`
       - clip 到 [0, 1]：`cx = max(0, min(1, cx))`，其他同理
       - 过滤：如果 `w <= 0.01` 或 `h <= 0.01`，跳过（与 DetectionDataset 的过滤逻辑一致）
       - 添加到结果列表
  3. 返回结果列表

#### `process_session(session_id: str, dataset_dir: Path, output_dir: Path, class_mapping: dict, min_area: int, copy_images: bool) -> tuple[int, int, int]`

- **职责：** 处理单个 session 的所有帧
- **输入：** session ID、输入/输出目录、映射表、阈值、是否复制图像
- **输出：** `(total_frames, frames_with_boxes, total_boxes)` 统计元组
- **处理逻辑：**
  1. 扫描 `images/<session_id>/` 下所有 .png 文件
  2. 对每个帧：
     - 构造 mask 路径：`labels_segmentation_masks/<session_id>/<frame>.png`
     - 如果 mask 不存在，记录警告，跳过
     - 读取图像获取尺寸（或从 mask 读取，两者尺寸相同）
     - 调用 `extract_boxes_from_mask()` 提取 bbox
     - 写入标签文件：`output/labels/<session_id>_<frame>.txt`
     - 如果 `copy_images`：复制图像到 `output/images/<session_id>_<frame>.png`
  3. 返回统计

#### `main()`

- **职责：** 脚本入口，协调所有处理流程
- **流程：**
  1. 解析命令行参数
  2. 创建输出目录结构：`output/images/`, `output/labels/`, `output/configs/`
  3. 调用 `get_active_sessions()` 获取有效 session 列表
  4. 遍历每个 session，调用 `process_session()`
  5. 汇总统计，写入 `classes.txt`
  6. 打印最终统计信息

### 1.6 错误处理策略

| 情况 | 处理方式 |
|------|----------|
| session 目录为空 | 跳过，不报错，统计中记录 `skipped_sessions` |
| mask 文件不存在 | 跳过该帧，打印警告 `[WARN] mask not found: <path>` |
| mask 文件损坏 | 跳过该帧，打印警告 `[WARN] failed to read mask: <path>` |
| 图像文件不存在 | 跳过该帧，打印警告 `[WARN] image not found: <path>` |
| 提取到 0 个 bbox | 正常处理，写入空 .txt 文件 |
| 输出目录已存在 | 不覆盖已有文件，跳过已存在的标签文件 |

### 1.7 日志输出建议

```
[Convert] 扫描有效 session...
[Convert] 找到 94 个有效 session（146 个目录中有 52 个为空）
[Convert] 开始处理，输出到 data/phase1_sanpo_5class
[Convert] Session 1/94: -5OCP... → 528 帧, 1843 个 bbox
[Convert] Session 2/94: -PqSD... → 412 帧, 1205 个 bbox
...
[Convert] 完成: 38609 帧处理, 35200 帧有标注, 125000 个 bbox
[Convert] 类别分布: person=25000, vehicle=38000, pole=18000, stairs=4000, obstacle=40000
[Convert] 已写入 data/phase1_sanpo_5class/configs/classes.txt
```

### 1.8 生成的文件清单

```
data/phase1_sanpo_5class/
├── images/
│   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.png
│   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000001.png
│   └── ... (约 38,609 个文件)
├── labels/
│   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.txt
│   ├── -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000001.txt
│   └── ... (与图像一一对应)
└── configs/
    └── classes.txt
```

---

## 2. `split_phase1.py` 接口设计

### 2.1 职责

将 convert_sanpo_phase1.py 生成的数据按 session 划分为 train/val/test，生成图像名列表文件。

**不做：** 不复制文件、不移动文件、不修改标签。

### 2.2 输入与输出

| 项目 | 路径 | 说明 |
|------|------|------|
| 输入 - 标签目录 | `data/phase1_sanpo_5class/labels/` | 用于扫描所有已转换的样本 |
| 输出 - train.txt | `data/phase1_sanpo_5class/meta/train.txt` | 训练集图像名列表 |
| 输出 - val.txt | `data/phase1_sanpo_5class/meta/val.txt` | 验证集图像名列表 |
| 输出 - test.txt | `data/phase1_sanpo_5class/meta/test.txt` | 测试集图像名列表 |

### 2.3 命令行参数

```
python split_phase1.py \
    --data-dir data/phase1_sanpo_5class \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--data-dir` | str | `data/phase1_sanpo_5class` | phase1 数据根目录 |
| `--train-ratio` | float | `0.70` | 训练集占比 |
| `--val-ratio` | float | `0.15` | 验证集占比 |
| `--test-ratio` | float | `0.15` | 测试集占比 |
| `--seed` | int | `42` | 随机种子 |

### 2.4 session_id 提取规则

文件名格式：`<session_id>_<frame_number:06d>.txt`

session_id 可能包含下划线，因此**从右侧最后一个下划线分割**：

```python
def extract_session_id(filename_stem: str) -> str:
    """
    从文件名 stem 提取 session_id。
    例: "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000"
      → session_id = "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG"
    """
    # 从右侧分割，只分一次
    last_underscore = filename_stem.rfind('_')
    if last_underscore < 0:
        raise ValueError(f"Invalid filename format: {filename_stem}")
    return filename_stem[:last_underscore]
```

### 2.5 划分逻辑

```
1. 扫描 labels/ 目录下所有 .txt 文件，提取文件名 stem 列表
2. 从每个 stem 提取 session_id
3. 按 session_id 分组：{session_id: [stem1, stem2, ...]}
4. 获取 session_id 列表，用 --seed 固定随机种子后打乱
5. 按比例切分：
   - train_sessions = shuffled_sessions[:N_train]
   - val_sessions = shuffled_sessions[N_train:N_train+N_val]
   - test_sessions = shuffled_sessions[N_train+N_val:]
6. 展开为帧级列表：
   - train_names = [stem for s in train_sessions for stem in sessions[s]]
   - val_names = [stem for s in val_sessions for stem in sessions[s]]
   - test_names = [stem for s in test_sessions for stem in sessions[s]]
7. 写入 meta/train.txt, meta/val.txt, meta/test.txt
```

### 2.6 随机种子固定

```python
import random
random.seed(seed)
session_keys = list(session_groups.keys())
random.shuffle(session_keys)
```

必须在 shuffle 之前设置种子，确保可复现。

### 2.7 数据泄漏验证

划分完成后必须执行以下检查：

```python
# 检查 1: session 级无交集
train_sessions_set = set(extract_session_id(n) for n in train_names)
val_sessions_set = set(extract_session_id(n) for n in val_names)
test_sessions_set = set(extract_session_id(n) for n in test_names)

assert len(train_sessions_set & val_sessions_set) == 0, "train/val session 泄漏"
assert len(train_sessions_set & test_sessions_set) == 0, "train/test session 泄漏"
assert len(val_sessions_set & test_sessions_set) == 0, "val/test session 泄漏"

# 检查 2: 帧级无交集
assert len(set(train_names) & set(val_names)) == 0, "train/val 帧泄漏"
assert len(set(train_names) & set(test_names)) == 0, "train/test 帧泄漏"

# 检查 3: 总数一致
total = len(train_names) + len(val_names) + len(test_names)
assert total == len(all_names), f"总数不一致: {total} vs {len(all_names)}"
```

### 2.8 生成的文件格式

**train.txt / val.txt / test.txt：**
```
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000059
-PqSDmiEe2pXjmYHgxh4YEBsj0T5LU10_000000
```
- 每行一个图像文件名 stem（无扩展名，无路径前缀）
- 无空行
- 无 header

### 2.9 日志输出建议

```
[Split] 扫描 data/phase1_sanpo_5class/labels/...
[Split] 找到 38609 个标签文件，涉及 94 个 session
[Split] 随机种子: 42
[Split] 划分结果:
  train: 66 sessions, 27026 帧 (70.0%)
  val:   14 sessions, 5792 帧 (15.0%)
  test:  14 sessions, 5791 帧 (15.0%)
[Split] 数据泄漏检查: PASS
[Split] 已写入 meta/train.txt, meta/val.txt, meta/test.txt
```

---

## 3. `verify_phase1.py` 接口设计

### 3.1 职责

验证 convert + split 的输出是否正确、完整、无泄漏。生成统计报告。

**不做：** 不修改任何文件，只读检查。

### 3.2 命令行参数

```
python verify_phase1.py \
    --data-dir data/phase1_sanpo_5class \
    --visualize 10
```

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--data-dir` | str | `data/phase1_sanpo_5class` | phase1 数据根目录 |
| `--visualize` | int | `0` | 随机抽样可视化数量（0 表示不生成） |

### 3.3 检查清单

| 编号 | 检查项 | 通过条件 | 失败时报告 |
|------|--------|----------|-----------|
| C1 | 图像数 == 标签数 | `len(images/) == len(labels/)` | `[FAIL] C1: 图像数 N != 标签数 M` |
| C2 | train.txt 中的每个 stem 都有对应图像 | 每个 stem 在 images/ 中有 .png | `[FAIL] C2: train 中有 K 个 stem 找不到图像` |
| C3 | val.txt 同上 | 同上 | `[FAIL] C3: val 中有 K 个 stem 找不到图像` |
| C4 | test.txt 同上 | 同上 | `[FAIL] C4: test 中有 K 个 stem 找不到图像` |
| C5 | train/val/test session 无交集 | 三个集合的 session_id 两两交集为空 | `[FAIL] C5: train/val 共享 N 个 session` |
| C6 | train/val/test 帧无交集 | 三个集合的 stem 两两交集为空 | `[FAIL] C6: train/val 共享 N 个帧` |
| C7 | train + val + test 覆盖所有帧 | 三集合并集 == labels/ 中所有文件 stem | `[FAIL] C7: K 个帧未被任何集合覆盖` |
| C8 | 所有 bbox 值在 [0,1] 范围内 | cx, cy, w, h ∈ [0, 1] | `[FAIL] C8: K 个 bbox 值超出 [0,1]` |
| C9 | 所有 class_id 在 [0,4] 范围内 | class_id ∈ {0, 1, 2, 3, 4} | `[FAIL] C9: 发现无效 class_id: X` |
| C10 | w > 0.01 且 h > 0.01 | 每个 bbox 满足 | `[FAIL] C10: K 个 bbox 的 w 或 h <= 0.01` |
| C11 | classes.txt 内容正确 | 5 行，内容为 person/vehicle/pole/stairs/obstacle | `[FAIL] C11: classes.txt 内容不匹配` |
| C12 | 每个集合中每个类别至少有 100 个 bbox | 统计各类别 bbox 数 | `[WARN] C12: stairs 在 test 中仅 30 个 bbox` |

### 3.4 stats.json 字段定义

```json
{
  "total_images": 38609,
  "total_labels": 38609,
  "total_bboxes": 125000,
  "images_with_no_bbox": 3409,
  "sessions": {
    "total": 146,
    "active": 94,
    "empty": 52
  },
  "splits": {
    "train": {
      "sessions": 66,
      "images": 27026,
      "bboxes": 87500,
      "class_counts": {
        "person": 17500,
        "vehicle": 26600,
        "pole": 12600,
        "stairs": 2800,
        "obstacle": 28000
      }
    },
    "val": {
      "sessions": 14,
      "images": 5792,
      "bboxes": 18750,
      "class_counts": {
        "person": 3750,
        "vehicle": 5700,
        "pole": 2700,
        "stairs": 600,
        "obstacle": 6000
      }
    },
    "test": {
      "sessions": 14,
      "images": 5791,
      "bboxes": 18750,
      "class_counts": {
        "person": 3750,
        "vehicle": 5700,
        "pole": 2700,
        "stairs": 600,
        "obstacle": 6000
      }
    }
  },
  "class_counts_total": {
    "person": 25000,
    "vehicle": 38000,
    "pole": 18000,
    "stairs": 4000,
    "obstacle": 40000
  },
  "checks": {
    "C1_image_label_match": "PASS",
    "C2_train_images_exist": "PASS",
    "C3_val_images_exist": "PASS",
    "C4_test_images_exist": "PASS",
    "C5_session_no_leak": "PASS",
    "C6_frame_no_leak": "PASS",
    "C7_full_coverage": "PASS",
    "C8_bbox_range": "PASS",
    "C9_class_id_range": "PASS",
    "C10_bbox_size": "PASS",
    "C11_classes_content": "PASS",
    "C12_class_balance": "WARN: stairs in test = 30"
  },
  "generated_at": "2026-06-07T12:00:00"
}
```

### 3.5 可视化抽样（可选）

如果 `--visualize N` 且 N > 0：
1. 从 train/val/test 中各随机抽取 N 张图像
2. 在图像上绘制 bbox（不同类别不同颜色）
3. 保存到 `data/phase1_sanpo_5class/visualizations/<split>/` 目录
4. 文件名：`<stem>_vis.png`

颜色映射：
```python
CLASS_COLORS = {
    0: (255, 0, 0),     # person: 红
    1: (0, 255, 0),     # vehicle: 绿
    2: (0, 0, 255),     # pole: 蓝
    3: (255, 255, 0),   # stairs: 黄
    4: (255, 0, 255),   # obstacle: 品红
}
```

### 3.6 输出文件清单

```
data/phase1_sanpo_5class/
├── meta/
│   └── stats.json              ← 验证报告
└── visualizations/             ← 仅在 --visualize > 0 时生成
    ├── train/
    │   └── xxx_vis.png
    ├── val/
    │   └── xxx_vis.png
    └── test/
        └── xxx_vis.png
```

---

## 4. 文件格式契约

### 4.1 classes.txt

| 属性 | 值 |
|------|-----|
| 文件编码 | UTF-8（无 BOM） |
| 每行内容 | 一个类名，无前后空格 |
| 行数 | 严格 5 行 |
| 允许空行 | 不允许 |
| 顺序 | person(0), vehicle(1), pole(2), stairs(3), obstacle(4) |
| 行号与 ID | 第 1 行 = ID 0，第 2 行 = ID 1，...（0-indexed） |

```
person
vehicle
pole
stairs
obstacle
```

### 4.2 train.txt / val.txt / test.txt

| 属性 | 值 |
|------|-----|
| 文件编码 | UTF-8（无 BOM） |
| 每行内容 | 一个图像文件名 stem（无扩展名，无路径前缀） |
| 允许空行 | 不允许 |
| 文件名格式 | `{session_id}_{frame_number:06d}` |
| 示例 | `-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000` |
| 与图像的匹配方式 | `images/{stem}.png` |
| 与标签的匹配方式 | `labels/{stem}.txt` |
| 三个文件的 stem 交集 | 必须为空（无泄漏） |
| 三个文件的 stem 并集 | 必须等于 labels/ 目录下所有 .txt 文件的 stem 集合 |

### 4.3 YOLO label .txt 文件

| 属性 | 值 |
|------|-----|
| 文件编码 | UTF-8（无 BOM） |
| 每行格式 | `{class_id} {cx} {cy} {w} {h}` |
| 字段分隔 | 单个空格 |
| class_id | 整数，范围 [0, 4] |
| cx, cy, w, h | 浮点数，6 位小数，范围 (0, 1) |
| 允许空行 | 不允许（文件末尾无多余换行） |
| 空标签文件 | 允许（0 字节或仅包含空内容） |
| 每行末尾 | 无多余空格 |
| 示例 | `0 0.450000 0.600000 0.080000 0.200000` |

**一个 bbox 的完整写入格式：**
```python
f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
```

### 4.4 图像文件

| 属性 | 值 |
|------|-----|
| 格式 | PNG（从原始数据复制，不转换） |
| 命名 | `{session_id}_{frame_number:06d}.png` |
| 示例 | `-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.png` |
| 与标签的对应 | `images/{stem}.png` ↔ `labels/{stem}.txt` |

### 4.5 文件名与图像的匹配关系

```
images/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.png
  ↔ labels/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000.txt
  ↔ train.txt 中的行: -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000
```

**关键约束：** train.txt/val.txt/test.txt 中的每个 stem 必须在 images/ 和 labels/ 中都有对应文件。

---

## 5. `ssd_default.yaml` 修改清单

### 5.1 必须修改的字段

| 字段 | 当前值 | 新值 | 理由 |
|------|--------|------|------|
| `data.root` | `data/processed` | `data/phase1_sanpo_5class` | 指向新的 5 类数据目录 |
| `data.train_list` | `data/processed/train.txt` | `data/phase1_sanpo_5class/meta/train.txt` | 新的 train 列表路径 |
| `data.val_list` | `data/processed/val.txt` | `data/phase1_sanpo_5class/meta/val.txt` | 新的 val 列表路径 |
| `data.classes_file` | `data/configs/classes.txt` | `data/phase1_sanpo_5class/configs/classes.txt` | 新的 5 类 classes 文件 |
| `data.num_classes` | `8` | `5` | 5 类检测 |
| `model.num_classes` | `9` | `6` | 5 类 + 1 背景 |

### 5.2 暂时不要动的字段

| 字段 | 当前值 | 不动的理由 |
|------|--------|-----------|
| `model.backbone` | `mobilenet_v2` | 第一阶段验证此 backbone |
| `model.pretrained` | `true` | 保留 ImageNet 预训练权重 |
| `model.input_size` | `[300, 300]` | 保持与当前 SSD 架构一致 |
| `anchors.*` | 各项 | anchor 配置与类别数无关，第一阶段沿用 |
| `training.*` | 各项 | 训练超参第一阶段沿用，后续根据效果调整 |
| `loss.*` | 各项 | 损失函数配置第一阶段沿用 |
| `augmentation.*` | 各项 | 数据增强配置第一阶段沿用 |

### 5.3 后续可能需要调的字段

| 字段 | 可能调整 | 触发条件 |
|------|----------|----------|
| `anchors.min_sizes` / `max_sizes` | 重新优化 anchor 尺寸 | 如果小目标（pole, stairs）检测效果差 |
| `training.learning_rate` | 调整学习率 | 如果 loss 不收敛或震荡 |
| `training.batch_size` | 调整 batch size | 如果显存不足或收敛慢 |
| `loss.neg_pos_ratio` | 调整正负样本比 | 如果类别不平衡导致训练不稳定 |
| `training.freeze_backbone_epochs` | 调整冻结轮数 | 如果 backbone 需要更多/更少微调 |

### 5.4 修改后的 ssd_default.yaml 关键部分

```yaml
data:
  root: "data/phase1_sanpo_5class"
  train_list: "data/phase1_sanpo_5class/meta/train.txt"
  val_list: "data/phase1_sanpo_5class/meta/val.txt"
  classes_file: "data/phase1_sanpo_5class/configs/classes.txt"
  num_classes: 5

model:
  backbone: "mobilenet_v2"
  pretrained: true
  input_size: [300, 300]
  num_classes: 6  # 5类 + 背景
```

---

## 6. 数据加载模块应满足的最小接口要求

### 6.1 现有接口（不需要修改）

当前 `DetectionDataset` 的接口已经满足需求，**不需要修改**。原因：

| 需求 | 现有实现 | 是否满足 |
|------|----------|----------|
| 读取 YOLO 格式标签 | `__getitem__` 中解析 `class_id cx cy w h` | ✅ |
| class_id 自动 +1（背景=0） | `labels.append(cls_id + 1)` | ✅ |
| 过滤无效 bbox | `if w > 0.01 and h > 0.01` | ✅ |
| 空标签文件处理 | 返回 `zeros(0, 4)` 和 `zeros(0)` | ✅ |
| 图像 resize 到 300×300 | `image.resize((input_size, input_size))` | ✅ |
| ImageNet 归一化 | `Normalize(mean, std)` | ✅ |
| 支持 .png 和 .jpg | 先尝试 .png，再尝试 .jpg | ✅ |
| collate_fn 处理变长 bbox | 返回 List[Tensor] | ✅ |

### 6.2 关键数据流（不需要改动）

```
YOLO label 文件:  "0 0.450000 0.600000 0.080000 0.200000"
        ↓ DetectionDataset.__getitem__()
模型输入:         boxes = [[0.45, 0.60, 0.08, 0.20]]  (float32)
                  labels = [1]                          (int64, class_id+1)
        ↓ collate_fn()
batch:            images = [B, 3, 300, 300]
                  boxes = list of [N_i, 4]
                  labels = list of [N_i]
        ↓ model(images)
模型输出:         conf = [B, 2264, 6]   (6 = 5类 + 背景)
                  loc = [B, 2264, 4]
        ↓ MultiBoxLoss(conf, loc, gt_boxes, gt_labels, anchors)
损失:             total_loss, loc_loss, conf_loss
```

### 6.3 训练脚本需要修改的地方

`train_ssd.py` 中需要修改的**唯一**地方是 config 文件路径的默认值（如果使用新 config 文件）：

```python
# 当前
parser.add_argument("--config", type=str, default="src/configs/ssd_default.yaml")
# 不需要改，因为改的是 ssd_default.yaml 的内容，不是路径
```

**实际上 train_ssd.py 本身不需要修改任何代码。** 只需要修改 `ssd_default.yaml` 的内容即可。

### 6.4 YOLO 标注到 SSD 训练格式的转换

**不需要额外转换。** 当前代码中 YOLO 格式和 SSD 训练格式是直接兼容的：

| 格式 | cx | cy | w | h | class_id |
|------|----|----|---|---|----------|
| YOLO label 文件 | 归一化 [0,1] | 归一化 [0,1] | 归一化 [0,1] | 归一化 [0,1] | 0-indexed |
| SSD gt_boxes | 归一化 [0,1] | 归一化 [0,1] | 归一化 [0,1] | 归一化 [0,1] | — |
| SSD gt_labels | — | — | — | — | 1-indexed (+1 偏移) |

唯一转换：`class_id + 1`（在 DetectionDataset.__getitem__ 中自动完成）。

---

## 7. 第一轮编码时的边界与禁止事项

### 7.1 不允许擅自改动的地方

| 禁止项 | 原因 |
|--------|------|
| 不许修改 `DetectionDataset` 类 | 当前接口已满足需求，改动会引入回归风险 |
| 不许修改 `SSDMobileNetV2` 类 | 模型架构不属于数据转换任务范围 |
| 不许修改 `MultiBoxLoss` 类 | 损失函数不属于数据转换任务范围 |
| 不许修改 `collate_fn` | collate 逻辑已满足需求 |
| 不许修改 `generate_anchors` | anchor 生成不属于数据转换任务范围 |
| 不许修改 `box_utils.py` | 工具函数不属于数据转换任务范围 |
| 不许引入新的依赖库 | 只用 numpy, PIL, json, pathlib, random |
| 不许改变 YOLO 格式 | cx cy w h 归一化格式是与 SSD 的接口契约 |

### 7.2 不确定时必须停下来说明的情况

| 情况 | 应该怎么做 |
|------|-----------|
| 发现 mask 编码方式与文档不一致 | 停下来，打印 mask 的实际值范围，报告给用户 |
| 发现 session 内帧数与 manifest 不一致 | 停下来，报告差异，让用户决定 |
| 发现某些 bbox 的 cx/cy/w/h 超出 [0,1] | clip 到 [0,1]，记录警告日志 |
| 发现某个 session 的所有帧都没有 bbox | 正常处理（写入空标签），但在日志中报告 |
| 发现 class_id 映射后出现非预期值 | 停下来，报告实际出现的 mask 值 |

### 7.3 不属于当前任务范围的内容

| 不做 | 原因 |
|------|------|
| 不做 SUNRGBD 转换 | 第一阶段只用 SANPO |
| 不做数据增强 | 增强在训练时实时做，不在预处理阶段 |
| 不做图像 resize | resize 在训练时由 Dataset 做 |
| 不做模型修改 | 模型适配是后续任务 |
| 不做训练 | 数据准备完成后才训练 |
| 不做评估 | 评估是训练后的任务 |
| 不做提醒逻辑 | 不在感知层范围内 |
| 不做路径规划 | 不在感知层范围内 |
| 不做分类任务 | 目标是检测（bbox），不是分类 |
| 不做 anchor 优化 | 第一阶段沿用现有 anchor |
| 不做 ONNX/TFLite 导出 | 部署是后续任务 |

---

## 8. 下一步可直接编码任务单

### 任务 1：编写 `convert_sanpo_phase1.py`

| 项目 | 内容 |
|------|------|
| **目标** | 从 SANPO 原始 mask 提取 5 类 bbox，生成 YOLO 格式标签 |
| **输入** | `data/SANPO-Real-Labeled-Full/`（原始数据） |
| **输出** | `data/phase1_sanpo_5class/`（images/ + labels/ + configs/classes.txt） |
| **涉及文件** | 新建 `blind-assist-detection/scripts/convert/convert_sanpo_phase1.py` |
| **约束** | 只用 numpy, PIL, json, pathlib, argparse；不修改任何现有文件；不引入新依赖 |
| **验收标准** | 1. 运行后 `data/phase1_sanpo_5class/images/` 下有约 38,609 个 .png 文件<br>2. `data/phase1_sanpo_5class/labels/` 下有相同数量的 .txt 文件<br>3. 每个 .txt 文件格式符合 4.3 节契约<br>4. `classes.txt` 内容为 5 类名<br>5. 日志输出包含处理帧数和各类别 bbox 数量统计 |

### 任务 2：编写 `split_phase1.py`

| 项目 | 内容 |
|------|------|
| **目标** | 按 session 划分 train/val/test，生成图像名列表 |
| **输入** | `data/phase1_sanpo_5class/labels/`（扫描已转换的标签文件） |
| **输出** | `data/phase1_sanpo_5class/meta/train.txt`, `val.txt`, `test.txt` |
| **涉及文件** | 新建 `blind-assist-detection/scripts/convert/split_phase1.py` |
| **约束** | 只用 pathlib, random, argparse；固定种子 42；按 session 划分而非按帧 |
| **验收标准** | 1. 三个 .txt 文件中的 stem 两两无交集<br>2. 三个 .txt 文件的 stem 并集 == labels/ 下所有 .txt 文件的 stem<br>3. 同一 session 的所有帧在同一个集合中<br>4. train/val/test 的 session 数约为 66/14/14<br>5. 划分比例接近 70/15/15 |

### 任务 3：编写 `verify_phase1.py`

| 项目 | 内容 |
|------|------|
| **目标** | 验证转换和划分的正确性，生成统计报告 |
| **输入** | `data/phase1_sanpo_5class/`（完整数据目录） |
| **输出** | `data/phase1_sanpo_5class/meta/stats.json` |
| **涉及文件** | 新建 `blind-assist-detection/scripts/convert/verify_phase1.py` |
| **约束** | 只读检查，不修改任何文件；检查项必须覆盖 C1-C12 |
| **验收标准** | 1. 所有检查项 C1-C11 为 PASS<br>2. C12 为 PASS 或 WARN（不阻塞）<br>3. stats.json 包含 3.4 节定义的所有字段<br>4. 脚本退出码：全部 PASS 时为 0，有 FAIL 时为 1 |

---

当前任务已完成
