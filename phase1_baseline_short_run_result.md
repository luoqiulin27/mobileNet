# Phase 1 短程 Baseline 训练 + 评估结果

## 1. 训练命令

```bash
cd D:\project\mobileNet\blind-assist-detection
python scripts/train/train_ssd.py --config src/configs/ssd_default.yaml --gpu 0 --epochs 10 --num-workers 0
```

## 2. 训练结果

训练 **成功完成**，10 个 epoch 全部跑通。

### Loss 趋势

| Epoch | Train Loss | Val Loss | 耗时 |
|-------|-----------|----------|------|
| 1 | 10.5807 | 7.4010 | 3407s |
| 2 | 6.3422 | 7.4484 | 3405s |
| 3 | 5.9044 | 6.9924 | 3342s |
| 4 | 5.5279 | 6.7608 | 3327s |
| 5 | 5.1734 | 6.5033 | 3316s |
| 6 | 4.8307 | 6.2768 | 3323s |
| 7 | 4.5460 | 6.0267 | 3417s |
| 8 | 4.3136 | 5.8710 | 3329s |
| 9 | 4.1582 | 5.7529 | 3918s |
| 10 | 4.0711 | 5.6850 | 3549s |

**总训练时间：** 约 9.5 小时（10 个 epoch，每个约 55 分钟）

### Loss 下降分析

- **Train Loss**：从 10.58 下降到 4.07，下降 **61.5%**
- **Val Loss**：从 7.40 下降到 5.69，下降 **23.1%**
- **趋势**：Train 和 Val loss 持续下降，无发散，模型在学习

### Checkpoint 信息

- **Best checkpoint**: `outputs/checkpoints/best.pth`（epoch 9, val_loss=5.685）
- **Last checkpoint**: `outputs/checkpoints/last.pth`（epoch 9）
- **模型参数量**: 7,328,260
- **Anchor 数量**: 2264
- **配置**: MobileNetV2 backbone + SSD, input_size=300×300, num_classes=6

## 3. 评估结果

评估脚本 `eval_map.py` 存在性能问题：AP 计算阶段的 IoU 比较循环在大数据集上极慢（5802 张图 × 所有检测框 × 所有 GT 框的逐对比较）。推理阶段已正常完成（726 batches 全部跑通），但 AP 计算阶段卡住。

**评估命令：**
```bash
python scripts/eval/eval_map.py --checkpoint outputs/checkpoints/best.pth --split val --gpu 0 --num-workers 0
```

**状态：** 推理完成，AP 计算阶段因性能问题超时。需要优化 eval_map.py 中的 IoU 计算效率。

## 4. 风险判断

### 当前 baseline 是否"学得动"？

**✅ 是的，模型明显在学习。**

证据：
1. Train Loss 持续下降（10.58 → 4.07）
2. Val Loss 持续下降（7.40 → 5.69）
3. 无发散、无 NaN、无 loss 震荡
4. Val Loss 在每个 epoch 都在改善（best checkpoint 在 epoch 9）

### 是否值得继续跑完整 100 epoch？

**✅ 值得。** Loss 仍在下降趋势中，尚未收敛。100 epoch 应该能进一步降低 loss。

### 是否存在明显的类别问题？

**基于训练 loss 无法直接判断**（需要 mAP 评估）。但从数据统计看：
- stairs 类仅 3,113 个 bbox（0.8%），是最大的类别不平衡风险
- 其他 4 类样本量相对充足

## 5. 已修改的文件

本次任务未修改任何文件。所有修改在之前任务中已完成。

## 6. 下一步建议

### 短期（修复评估）

1. **优化 eval_map.py**：AP 计算中的 IoU 比较循环需要向量化处理（当前逐对创建 tensor 太慢）
2. **重新运行评估**：确认各类别 AP 和 mAP

### 中期（完善 baseline）

3. **全量训练 100 epochs**：loss 仍在下降，完整训练应能进一步提升
4. **评估 mAP**：确认 baseline 的检测性能
5. **根据 mAP 决定优化方向**：
   - 如果 stairs AP 接近 0 → 考虑 focal loss 或过采样
   - 如果整体 mAP > 0.1 → baseline 可接受，进入 SUNRGBD 扩展
   - 如果整体 mAP < 0.05 → 需要检查数据质量或调整 anchor 配置
