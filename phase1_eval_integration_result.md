# Phase 1 评估链路适配结果

## 1. 检查的评估相关文件

| 文件 | 存在 | 状态 |
|------|------|------|
| `scripts/eval/eval_map.py` | ✅ | 需要修复路径兼容性 |
| `src/metrics/__init__.py` | ✅ | 无需修改 |
| `outputs/metrics/` | ✅ | 输出目录，无需修改 |

## 2. 兼容性检查结果

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 读取 ssd_default.yaml | ✅ | 已兼容 |
| 读取 5 类 classes.txt | ✅ | 从 config 读取，自动适配 |
| num_classes=6 兼容 | ✅ | 从 config 读取，自动适配 |
| 加载 train_ssd.py 的 checkpoint | ✅ | 格式一致 |
| image_dir/label_dir 路径 | ❌ → ✅ | 已修复：支持 config 中的 image_dir/label_dir |
| 支持 test split | ❌ → ✅ | 已修复：新增 test split 支持 |
| json import 位置 | ⚠️ → ✅ | 已修复：移到文件顶部 |
| Windows num_workers | ⚠️ → ✅ | 已修复：支持 --num-workers 参数 |

## 3. 修改的文件

### 文件 1: `scripts/eval/eval_map.py`

**修改 1: image_dir/label_dir 路径兼容**

```python
# 修改前
image_dir=os.path.join(data_root, "images"),
label_dir=os.path.join(data_root, "labels"),

# 修改后
image_dir = config["data"].get("image_dir", os.path.join(data_root, "images"))
label_dir = config["data"].get("label_dir", os.path.join(data_root, "labels"))
```

**原因：** phase1 数据在 `images/all/` 和 `labels/all/`，硬编码路径无法找到数据。

**修改 2: 支持 test split**

```python
# 修改前
if args.split == "val":
    names = load_image_list(config["data"]["val_list"])
else:
    names = load_image_list(config["data"]["train_list"])

# 修改后
split_key = {"val": "val_list", "train": "train_list", "test": "test_list"}.get(args.split, "val_list")
names = load_image_list(config["data"][split_key])
```

**原因：** baseline 评估应在 test 集上进行，原脚本只支持 val/train。

**修改 3: json import 规范化**

```python
# 修改前：import json 在 main() 底部
# 修改后：import json 在文件顶部
```

**修改 4: num_workers 参数**

```python
# 新增 --num-workers 参数，Windows 兼容
parser.add_argument("--num-workers", type=int, default=None)
```

### 文件 2: `src/configs/ssd_default.yaml`

**修改: 新增 test_list**

```yaml
# 新增
test_list: "data/phase1_sanpo_5class/meta/test.txt"
```

**原因：** eval_map.py 支持 test split 后需要读取 test_list 配置。

## 4. 推荐的短程 baseline 训练命令

```bash
cd D:\project\mobileNet\blind-assist-detection

# 短程训练：10 epochs，验证训练+评估闭环
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu 0 \
    --epochs 10 \
    --num-workers 0
```

**预期耗时：** 约 30-60 分钟（取决于 GPU）。

## 5. 推荐的评估命令

```bash
cd D:\project\mobileNet\blind-assist-detection

# 在 val 集上评估
python scripts/eval/eval_map.py \
    --config src/configs/ssd_default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split val \
    --gpu 0 \
    --num-workers 0

# 在 test 集上评估（最终结果）
python scripts/eval/eval_map.py \
    --config src/configs/ssd_default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split test \
    --gpu 0 \
    --num-workers 0
```

## 6. 建议的"训练+评估"最小闭环顺序

```
第 1 步: 短程训练（10 epochs）
  python scripts/train/train_ssd.py --gpu 0 --epochs 10 --num-workers 0
  → 验证训练流程完整跑通
  → 产出: outputs/checkpoints/best.pth, last.pth

第 2 步: val 集评估
  python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split val --gpu 0 --num-workers 0
  → 验证评估流程完整跑通
  → 检查 mAP 和各类别 AP
  → 产出: outputs/metrics/eval_val.json

第 3 步: 根据 val 结果决定
  如果 mAP > 0.1（随机水平以上）：
    → 进入全量 100 epoch 训练
  如果 mAP ≈ 0 或训练不收敛：
    → 检查数据和配置问题

第 4 步: 全量训练（100 epochs）
  python scripts/train/train_ssd.py --gpu 0 --epochs 100 --num-workers 0
  → 建立正式 baseline

第 5 步: test 集最终评估
  python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split test --gpu 0 --num-workers 0
  → 最终 baseline 性能指标
  → 产出: outputs/metrics/eval_test.json
```

## 7. 评估指标说明

| 指标 | 含义 | 期望值（10 epochs） |
|------|------|---------------------|
| mAP@0.5 | 所有类别的平均 AP | > 0.05（至少高于随机） |
| person AP | 行人检测精度 | 应为最高（样本最多） |
| vehicle AP | 车辆检测精度 | 应较高 |
| pole AP | 杆状物检测精度 | 中等 |
| stairs AP | 台阶检测精度 | 可能较低（样本最少） |
| obstacle AP | 障碍物检测精度 | 中等 |
