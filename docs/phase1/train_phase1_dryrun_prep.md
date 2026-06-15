# 训练脚本审查 + Dry Run 适配结果

## 1. 检查的问题

| 编号 | 检查项 | 结果 |
|------|--------|------|
| 1 | val_loader 为空时是否报错 | 🔴 **发现阻塞** — `val_loss /= len(val_loader)` 除零 |
| 2 | train_loader 为空时是否报错 | 🔴 **发现阻塞** — `train_loss /= len(train_loader)` 除零 |
| 3 | 小样本时 loss 分母是否可能为 0 | ✅ 修复后安全 |
| 4 | DataLoader 输出格式与预期是否一致 | ✅ 一致 |
| 5 | anchors 数量与模型输出是否一致 | ✅ 验证通过：2264 anchors，6 类输出 `[1,2264,6]` |
| 6 | 小数据集能否进行最小训练冒烟测试 | 🟡 **需要新增参数** |
| 7 | 是否需要 dry run 参数 | 🟡 **需要新增** |

## 2. 发现的阻塞点

### 阻塞 1: val_loader 为空时除零

**位置：** `train_ssd.py` 第 ~190 行

```python
val_loss /= len(val_loader)  # val_loader 为空时 ZeroDivisionError
```

**触发条件：** val.txt 为空或不存在时。

**修复：** 使用 `val_batches` 计数器，仅在 `val_batches > 0` 时做除法。

### 阻塞 2: train_loader 为空时除零

**位置：** `train_ssd.py` 第 ~170 行

```python
train_loss /= len(train_loader)  # train_loader 为空时 ZeroDivisionError
```

**触发条件：** train.txt 为空或不存在时。

**修复：** 使用 `train_batches` 计数器，仅在 `train_batches > 0` 时做除法。

### 阻塞 3: 无 dry run 参数

**问题：** 当前只能通过修改 yaml 配置来控制 epochs、batch 数量等。对于冒烟测试不友好。

**修复：** 新增 4 个命令行参数：
- `--epochs`: 覆盖配置中的训练轮数
- `--max-train-batches`: 每个 epoch 最多训练 batch 数
- `--max-val-batches`: 每个 epoch 最多验证 batch 数
- `--num-workers`: 覆盖 DataLoader workers 数

## 3. 修改的文件

### 文件: `scripts/train/train_ssd.py`

**修改 1: 新增命令行参数**

```python
# 新增
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max-train-batches", type=int, default=0)
parser.add_argument("--max-val-batches", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=None)
```

**修改 2: 使用 num_workers 参数**

```python
num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
```

**修改 3: 使用 epochs 参数**

```python
epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
```

**修改 4: 训练循环支持 max_train_batches + 除零保护**

```python
train_batches = 0
for batch_idx, batch in enumerate(train_loader):
    if max_train_batches > 0 and batch_idx >= max_train_batches:
        break
    ...
    train_batches += 1
if train_batches > 0:
    train_loss /= train_batches
```

**修改 5: 验证循环支持 max_val_batches + 除零保护**

```python
val_batches = 0
for batch_idx, batch in enumerate(val_loader):
    if max_val_batches > 0 and batch_idx >= max_val_batches:
        break
    ...
    val_batches += 1
if val_batches > 0:
    val_loss /= val_batches
```

## 4. 最小 Dry Run 推荐命令

### 前置条件

需要先有数据。当前只有 3 个样本（1 个 session），train=3，val=0。

### 方案 A: 用当前 3 个样本快速验证 pipeline（推荐）

```bash
cd D:\project\mobileNet\blind-assist-detection

# 1 epoch，最多 2 个 train batch，跳过 val，CPU 模式，0 workers
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu -1 \
    --epochs 1 \
    --max-train-batches 2 \
    --max-val-batches 0 \
    --num-workers 0
```

**预期结果：**
- 模型加载成功（MobileNetV2 + SSD 检测头）
- 数据加载成功（3 个样本，1 个 batch）
- 前向传播成功（conf: [1,2264,6], loc: [1,2264,4]）
- loss 计算成功
- 反向传播成功
- 训练循环正常结束

### 方案 B: 全量数据后的完整冒烟测试

```bash
cd D:\project\mobileNet\blind-assist-detection

# 先全量准备数据
python scripts/convert/convert_sanpo_phase1.py
python scripts/convert/split_phase1.py
python scripts/convert/verify_phase1.py

# 冒烟测试: 1 epoch，最多 10 个 train batch，最多 5 个 val batch
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu -1 \
    --epochs 1 \
    --max-train-batches 10 \
    --max-val-batches 5 \
    --num-workers 0
```

## 5. Dry Run 前的推荐顺序

```
第 1 步: 全量运行 convert_sanpo_phase1.py
  → 转换全部 ~38,609 帧
  → 产出: data/phase1_sanpo_5class/images/all/*.png, labels/all/*.txt

第 2 步: 全量运行 split_phase1.py
  → 按 session 划分 train/val/test
  → 产出: meta/train.txt, meta/val.txt, meta/test.txt

第 3 步: 全量运行 verify_phase1.py
  → 验证数据完整性
  → 确认所有检查项 PASS

第 4 步: Dry Run（方案 A 或方案 B）
  → 确认训练 pipeline 端到端通畅
  → 确认 loss 正常计算

第 5 步: 全量训练
  → 100 epochs，建立 baseline
```

## 6. 新增参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | None（用配置值） | 覆盖训练轮数 |
| `--max-train-batches` | int | 0（全部） | 每 epoch 最多训练 batch 数 |
| `--max-val-batches` | int | 0（全部） | 每 epoch 最多验证 batch 数 |
| `--num-workers` | int | None（用配置值） | 覆盖 DataLoader workers 数 |

**向后兼容：** 所有新参数默认值均为 None 或 0，不传参时行为与原脚本完全一致。
