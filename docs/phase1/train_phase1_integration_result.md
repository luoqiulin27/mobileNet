# 训练接入配置适配结果

## 1. 检查的文件

| 文件 | 检查结果 |
|------|----------|
| `src/configs/ssd_default.yaml` | **需要修改** — 路径和 num_classes 不匹配 |
| `scripts/train/train_ssd.py` | **需要修改** — 路径拼接逻辑不兼容 `images/all/` 结构 |
| `src/datasets/detection_dataset.py` | **无需修改** — 通用实现，自动适配 |
| `src/models/ssd_mobilenet.py` | **无需修改** — num_classes 参数化，自动适配 |
| `src/losses/multibox_loss.py` | **无需修改** — 从 conf_pred.shape 读取 num_classes，自动适配 |
| `src/models/box_utils.py` | **无需修改** — anchor 生成与类别数无关 |

## 2. 修改的文件及原因

### 文件 1: `src/configs/ssd_default.yaml`

**修改内容：**

```yaml
# 修改前
data:
  root: "data/processed"
  train_list: "data/processed/train.txt"
  val_list: "data/processed/val.txt"
  classes_file: "data/configs/classes.txt"
  num_classes: 8

model:
  num_classes: 9  # 8类 + 背景

# 修改后
data:
  root: "data/phase1_sanpo_5class"
  image_dir: "data/phase1_sanpo_5class/images/all"
  label_dir: "data/phase1_sanpo_5class/labels/all"
  train_list: "data/phase1_sanpo_5class/meta/train.txt"
  val_list: "data/phase1_sanpo_5class/meta/val.txt"
  classes_file: "data/phase1_sanpo_5class/configs/classes.txt"
  num_classes: 5

model:
  num_classes: 6  # 5类 + 背景
```

**修改原因：**
- `data.root`: 指向新的 phase1 数据目录
- `data.image_dir` / `data.label_dir`: 新增字段，指向 `images/all` 和 `labels/all`（phase1 数据在 `all` 子目录下）
- `data.train_list` / `data.val_list`: 指向新的 split 文件
- `data.classes_file`: 指向新的 5 类 classes 文件
- `data.num_classes`: 从 8 改为 5
- `model.num_classes`: 从 9 改为 6（5+背景）

### 文件 2: `scripts/train/train_ssd.py`

**修改内容（3 处）：**

```python
# 修改前
data_root = config["data"]["root"]
train_names = load_image_list(config["data"]["train_list"])
...
train_dataset = DetectionDataset(
    image_dir=os.path.join(data_root, "images"),
    label_dir=os.path.join(data_root, "labels"),
...
val_dataset = DetectionDataset(
    image_dir=os.path.join(data_root, "images"),
    label_dir=os.path.join(data_root, "labels"),

# 修改后
data_root = config["data"]["root"]
image_dir = config["data"].get("image_dir", os.path.join(data_root, "images"))
label_dir = config["data"].get("label_dir", os.path.join(data_root, "labels"))
train_names = load_image_list(config["data"]["train_list"])
...
train_dataset = DetectionDataset(
    image_dir=image_dir,
    label_dir=label_dir,
...
val_dataset = DetectionDataset(
    image_dir=image_dir,
    label_dir=label_dir,
```

**修改原因：**
- phase1 数据结构是 `images/all/` 和 `labels/all/`，而非直接 `images/` 和 `labels/`
- 原代码硬编码 `os.path.join(data_root, "images")` 无法适配
- 新增 `image_dir` / `label_dir` 配置字段读取，向后兼容（config 中无此字段时回退到旧逻辑）

## 3. 确认无需修改的文件

| 文件 | 原因 |
|------|------|
| `detection_dataset.py` | `__getitem__` 读取 YOLO 格式 `class_id cx cy w h`，自动做 `cls_id + 1` 偏移。与类别数无关，与 5 类完全兼容 |
| `ssd_mobilenet.py` | `SSDMobileNetV2(num_classes=6)` 会自动调整检测头输出维度为 `[B, anchors, 6]` |
| `multibox_loss.py` | `CrossEntropyLoss` 从 `conf_pred.size(2)` 读取类别数，与 6 类完全兼容 |
| `box_utils.py` | anchor 生成基于 feature_maps/min_sizes/max_sizes，与类别数无关 |

## 4. 最小训练冒烟测试命令

**目标：** 验证配置能加载、数据能读到、模型能前向、loss 能计算。

**前置条件：** 至少有 1 个 session 的数据已通过 convert + split。

```bash
cd D:\project\mobileNet\blind-assist-detection

# 冒烟测试：1 个 epoch，batch_size=2，仅用少量数据
python scripts/train/train_ssd.py --config src/configs/ssd_default.yaml --gpu -1
```

**注意：** 当前只有 3 个样本（1 个 session），train 有 3 个，val 为 0。训练脚本在 val_loader 为空时可能会报错。如果需要完整冒烟测试，需要先全量运行 convert + split。

**如果 val 为空导致报错，可以临时用以下方式验证（不改代码）：**

```bash
# 先全量运行数据准备
python scripts/convert/convert_sanpo_phase1.py
python scripts/convert/split_phase1.py
python scripts/convert/verify_phase1.py

# 再运行训练冒烟测试
python scripts/train/train_ssd.py --config src/configs/ssd_default.yaml --gpu -1
```

## 5. 下一步推荐动作

```
第 1 步: 全量运行 convert_sanpo_phase1.py
  → 转换全部 ~38,609 帧
  → 产出: data/phase1_sanpo_5class/images/all/*.png, labels/all/*.txt

第 2 步: 全量运行 split_phase1.py
  → 按 session 划分 train/val/test
  → 产出: meta/train.txt, meta/val.txt, meta/test.txt

第 3 步: 全量运行 verify_phase1.py
  → 验证数据完整性
  → 产出: meta/stats.json, meta/verify_report.json

第 4 步: 训练冒烟测试
  → 确认 pipeline 端到端通畅
  → 检查 loss 是否正常下降

第 5 步: 全量训练
  → 100 epochs，建立 baseline
```

## 6. 兼容性说明

| 场景 | 行为 |
|------|------|
| config 中有 `image_dir` / `label_dir` | 使用 config 中的值（新行为） |
| config 中无 `image_dir` / `label_dir` | 回退到 `os.path.join(data_root, "images")`（旧行为） |
| 旧的 `data/processed` 数据 | 仍可通过修改 config 路径使用 |
