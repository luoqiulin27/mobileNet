# 训练性能优化报告

## 1. 当前训练慢 / GPU 利用率低的原因分析

### 根因定位

| 因素 | 影响程度 | 说明 |
|------|----------|------|
| `persistent_workers` 未设置 | 🔴 高 | 每个 epoch 结束后 worker 进程被销毁并重建，造成大量进程创建开销 |
| 无混合精度训练 | 🔴 高 | RTX 3060 的 Tensor Core 未利用，FP32 训练吞吐约为 AMP 的 50-60% |
| `optimizer.zero_grad()` 默认行为 | 🟡 中 | 默认将梯度清零为 0，`set_to_none=True` 可减少内存操作 |
| 无 `non_blocking` 传输 | 🟡 中 | CPU→GPU 数据传输阻塞，无法与计算重叠 |
| 无吞吐监控 | 🟡 中 | 无法量化瓶颈，调优无依据 |
| 图像尺寸大 | 🟢 低 | 原始 2208×1242 PNG 读取+resize 到 300×300，但 PIL 已足够快 |
| batch_size 偏保守 | 🟢 低 | 16 对 RTX 3060 6GB 来说合理，可尝试 32 |

### 为什么 MobileNetV2 + SSD 天然 GPU 利用率不高

- MobileNetV2 参数量仅 3.4M，前向传播极快（<10ms/batch on GPU）
- SSD 检测头也是轻量级
- **计算密度低**：模型太轻量，GPU 大部分时间在等数据
- 这是轻量模型的固有特征，不是 bug
- 优化方向：**最大化数据吞吐**，让 GPU 尽可能少等待

## 2. 检查的文件

| 文件 | 状态 |
|------|------|
| `scripts/train/train_ssd.py` | ✅ 已优化 |
| `src/datasets/detection_dataset.py` | ✅ 检查，无需修改 |
| `src/configs/ssd_default.yaml` | ✅ 检查，配置合理 |
| `src/models/ssd_mobilenet.py` | ✅ 检查，无需修改 |

## 3. 修改的文件及原因

### 文件：`scripts/train/train_ssd.py`

#### 修改 1: DataLoader 参数优化

```python
# 之前
train_loader = DataLoader(
    train_dataset, batch_size=..., shuffle=True,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
)

# 之后
loader_kwargs = {"batch_size": ..., "collate_fn": ..., "pin_memory": True}
if num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, **loader_kwargs)
```

**原因：**
- `persistent_workers=True`：worker 进程在 epoch 间保持存活，避免重复创建
- `prefetch_factor=2`：每个 worker 预取 2 个 batch，增加数据供给缓冲

#### 修改 2: 混合精度训练 (AMP)

```python
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

# 训练循环中
if use_amp:
    with torch.amp.autocast("cuda"):
        conf, loc = model(images)
        loss, loc_loss, conf_loss = criterion(...)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**原因：** RTX 3060 支持 FP16 Tensor Core，AMP 可将训练吞吐提升约 40-80%。

#### 修改 3: `optimizer.zero_grad(set_to_none=True)`

**原因：** 默认行为是将梯度张量填充为 0，`set_to_none=True` 直接将梯度设为 None，减少内存写入。

#### 修改 4: `non_blocking=True` 数据传输

```python
images = batch["images"].to(device, non_blocking=True)
gt_boxes = [b.to(device, non_blocking=True) for b in batch["boxes"]]
```

**原因：** 允许 CPU→GPU 传输与 GPU 计算重叠。

#### 修改 5: 吞吐监控

```python
samples_per_sec = (train_batches * batch_size) / elapsed
print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.0f}s, {samples_per_sec:.1f} samples/s)")
```

**原因：** 提供量化指标，便于比较不同配置。

## 4. 推荐的训练配置组合

### 配置 A：快速验证（推荐先跑）

```bash
python scripts/train/train_ssd.py \
    --gpu 0 --epochs 1 --num-workers 4 --max-train-batches 50
```

- batch_size: 16（配置默认）
- num_workers: 4
- 目标：验证优化后脚本正常运行，观察 samples/s

### 配置 B：性能对比

```bash
# B1: batch_size=16, num_workers=4
python scripts/train/train_ssd.py --gpu 0 --epochs 1 --num-workers 4

# B2: batch_size=32, num_workers=4（需修改 ssd_default.yaml 的 batch_size）
python scripts/train/train_ssd.py --gpu 0 --epochs 1 --num-workers 4

# B3: batch_size=16, num_workers=8
python scripts/train/train_ssd.py --gpu 0 --epochs 1 --num-workers 8
```

### 配置 C：正式训练

```bash
python scripts/train/train_ssd.py --gpu 0 --epochs 100 --num-workers 4
```

### Windows 特别注意

- `num_workers > 0` 在 Windows 上使用 `spawn` 方式创建进程，首次启动较慢
- `persistent_workers=True` 可缓解此问题（只在首次创建，后续复用）
- 如果遇到多进程问题，可退回 `num_workers=0`

## 5. 推荐的最小性能测试命令

```bash
cd D:\project\mobileNet\blind-assist-detection

# 测试 1: 优化后 baseline（1 epoch, 50 batches）
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu 0 --epochs 1 --num-workers 4 --max-train-batches 50

# 测试 2: 对比 num_workers=0（无多进程）
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu 0 --epochs 1 --num-workers 0 --max-train-batches 50

# 测试 3: 全量 1 epoch（观察完整 epoch 耗时）
python scripts/train/train_ssd.py \
    --config src/configs/ssd_default.yaml \
    --gpu 0 --epochs 1 --num-workers 4
```

## 6. 建议的执行顺序

```
第 1 步: 跑配置 A（50 batches），确认脚本正常，记录 samples/s
第 2 步: 跑测试 2（num_workers=0），对比 samples/s
第 3 步: 跑全量 1 epoch，记录完整 epoch 耗时和 samples/s
第 4 步: 根据结果决定是否调整 batch_size
第 5 步: 进入正式 100 epoch 训练
```

## 7. 本次未修改的内容

| 项目 | 原因 |
|------|------|
| `detection_dataset.py` | 数据加载逻辑本身合理，瓶颈在 DataLoader 参数 |
| `ssd_default.yaml` | 配置合理，batch_size=16 对 RTX 3060 6GB 是安全值 |
| 模型结构 | 不属于性能优化范围 |
| 数据增强 | 当前增强（翻转+颜色抖动）开销很小，无需优化 |
| Loss 函数 | 计算量小，不是瓶颈 |
