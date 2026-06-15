# Phase 1 评估优化结果

## 1. 性能瓶颈分析

### 根因

`eval_map.py` 的 AP 计算阶段（第 118-136 行）存在严重的性能问题：

```python
for d_idx, det in enumerate(dets):           # 遍历每个检测框
    det_box = torch.tensor(det["box"]).unsqueeze(0)  # 每次创建新 tensor
    for g_idx, gt in enumerate(gts):          # 遍历每个 GT
        gt_box = torch.tensor(gt["box"]).unsqueeze(0)  # 每次创建新 tensor
        iou = compute_iou(det_box, gt_box).item()      # 调用 jaccard
```

**问题：**
1. **O(D×G) 嵌套循环**：每个检测框与每个 GT 逐对比较
2. **每次迭代创建 tensor**：`torch.tensor()` + `.unsqueeze(0)` 在每次内层循环都执行
3. **调用 `jaccard` 函数**：该函数内部做 tensor 运算，有额外开销
4. **无跳过优化**：即使某个类别的检测框极少，也会遍历所有 GT

**估算：** 假设某类别有 1000 个检测框和 500 个 GT，需要 500,000 次 IoU 计算，每次创建 2 个 tensor。对于 5 个类别，总计可能超过百万次。

### 优化方案

将逐对 tensor 比较改为 **numpy 向量化计算**：

1. 将所有 GT box 一次性转为 numpy 数组 `(G, 4)`
2. 对每个检测框，用 numpy 向量化计算与所有 GT 的 IoU
3. 用 numpy 掩码跳过已使用的 GT
4. 移除 `torch.tensor` 创建和 `jaccard` 调用

## 2. 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `scripts/eval/eval_map.py` | 优化 AP 计算中的 IoU 比较循环 |

## 3. 优化点说明

### 优化 1: numpy 向量化 IoU

```python
# 之前：逐对 tensor 比较
det_box = torch.tensor(det["box"]).unsqueeze(0)
gt_box = torch.tensor(gt["box"]).unsqueeze(0)
iou = compute_iou(det_box, gt_box).item()

# 之后：numpy 向量化
gt_boxes_np = np.array([gt["box"] for gt in gts], dtype=np.float32)  # 一次性转
det_box = np.array(det["box"], dtype=np.float32)
# 向量化计算 IoU
x1 = np.maximum(det_box[0], gt_boxes_np[:, 0])
...
ious = np.where(union_area > 0, inter_area / union_area, 0.0)
```

**效果：** 每个检测框的 IoU 计算从 O(G) 次 tensor 操作变为 1 次 numpy 向量运算。

### 优化 2: numpy 掩码跳过已使用 GT

```python
# 之前：逐个检查 gt["used"]
for g_idx, gt in enumerate(gts):
    if gt["used"]:
        continue

# 之后：numpy 掩码
gt_used = np.zeros(num_gt, dtype=bool)
ious[gt_used] = -1.0  # 一次性屏蔽
```

### 优化 3: 移除 torch 依赖

AP 计算阶段不再需要 `torch.tensor` 和 `jaccard`，完全用 numpy 完成。

## 4. 评估命令

```bash
cd D:\project\mobileNet\blind-assist-detection
python scripts/eval/eval_map.py \
    --config src/configs/ssd_default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split val \
    --gpu 0 \
    --num-workers 0
```

## 5. 评估结果

**✅ 评估成功完成。**

### mAP@0.5

```
mAP: 0.1726
```

### 各类别 AP

| 类别 | AP |
|------|-----|
| person | 0.1803 |
| vehicle | 0.3545 |
| pole | 0.0909 |
| stairs | 0.0606 |
| obstacle | 0.1768 |

### 结果文件

`outputs/metrics/eval_val.json` 已正常生成。

## 6. Phase 1 Baseline 实验结论

### 训练结果（10 epochs）

| 指标 | 值 |
|------|-----|
| Train Loss | 10.58 → 4.07（下降 61.5%） |
| Val Loss | 7.40 → 5.69（下降 23.1%） |
| Best epoch | 9（val_loss=5.685） |
| 训练时间 | ~9.5 小时 |

### 评估结果（Val Set, mAP@0.5）

| 指标 | 值 | 分析 |
|------|-----|------|
| **mAP** | **0.1726** | 10 epoch 的 baseline，仍有提升空间 |
| vehicle | 0.3545 | 最好，样本量充足（75,608 train bbox） |
| person | 0.1803 | 次好，样本量充足（54,614 train bbox） |
| obstacle | 0.1768 | 中等，样本量最大（104,070 train bbox）但类别定义泛化 |
| pole | 0.0909 | 较差，杆状物检测本身较难 |
| stairs | 0.0606 | 最差，样本量极少（2,183 train bbox） |

### 关键发现

1. **模型确实在学习**：mAP=0.17 远高于随机水平，loss 持续下降
2. **vehicle 表现最好**：AP=0.35，符合预期（样本充足、特征明显）
3. **stairs 表现最差**：AP=0.06，与样本量极少（仅 2,183 个 train bbox）直接相关
4. **类别不平衡影响显著**：stairs 仅占总 bbox 的 0.8%，是最大的性能瓶颈

## 7. 下一步建议

### ✅ 可以继续进入 100 epoch 正式 baseline 训练

**理由：**
1. 训练流程完整跑通
2. 评估流程完整跑通
3. mAP=0.17 是 10 epoch 的结果，100 epoch 应有显著提升
4. Loss 仍在下降趋势中，尚未收敛

### 推荐路径

```
第 1 步: 100 epoch 全量训练
  python scripts/train/train_ssd.py --gpu 0 --epochs 100

第 2 步: 评估 mAP
  python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split val --gpu 0

第 3 步: 根据 mAP 决定优化方向
  - 如果 stairs AP 仍接近 0 → focal loss 或过采样
  - 如果整体 mAP > 0.3 → baseline 可接受，考虑 SUNRGBD 扩展
  - 如果整体 mAP < 0.2 → 检查 anchor 配置或数据质量
```
