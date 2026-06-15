# 训练脚本语义修复结果

## 1. 检查的两个问题

### 问题 1: scheduler T_max 与实际 epochs 不一致

**是否存在：✅ 存在**

```python
# 修改前（第 164-165 行）
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["training"]["epochs"], eta_min=1e-6  # 固定为 100
)
# ...
epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]  # 可能为 1
```

**问题：** `T_max` 始终为 config 中的 100，即使 `--epochs 1` 覆盖了实际训练轮数。导致 dry run 时调度器认为要跑 100 个 epoch，学习率衰减语义完全错误。

### 问题 2: val 为空时误存 best checkpoint

**是否存在：✅ 存在**

```python
# val_batches=0 时，val_loss 保持初始值 0.0
val_loss = 0.0
# ...
# 0.0 < float("inf") 为 True → 误存 best
if val_loss < best_loss:
    best_loss = val_loss  # 变为 0.0
    # 保存 best (loss=0.0000) ← 伪 best
```

**问题：** 没有验证数据时，val_loss=0.0 被当作最优结果保存，best_loss 被错误更新为 0.0。

## 2. 修改的代码

### 修复 1: scheduler T_max

```python
# 修改后：先解析 epochs，再创建 scheduler
epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
# ...
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

### 修复 2: val 为空时的 best checkpoint 逻辑

```python
# 修改后：增加 has_val 标志
has_val = val_batches > 0

# 日志中区分
if has_val:
    print("  Val   Loss: {:.4f}".format(val_loss))
else:
    print("  Val   Loss: N/A (no val batches)")

# best 保存条件增加 has_val
if has_val and val_loss < best_loss:
    # 保存 best

# 结束时区分
if best_loss < float("inf"):
    print("\n[训练完成] 最佳验证 loss: {:.4f}".format(best_loss))
else:
    print("\n[训练完成] 无有效验证结果")
```

## 3. 修改前后行为差异

| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| `--epochs 1` 时 scheduler T_max | 100（错误） | 1（正确） |
| val 为空时 val_loss 显示 | `0.0000` | `N/A (no val batches)` |
| val 为空时 best checkpoint | 保存 (loss=0.0) | 不保存 |
| val 为空时 best_loss | 被更新为 0.0 | 保持 inf |
| 训练结束时输出 | `最佳验证 loss: 0.0000` | `无有效验证结果` |
| TensorBoard val 日志 | 写入 0.0 | 不写入 |

## 4. 为什么这些修复对后续 baseline 有必要

**T_max 修复：**
- 全量训练 100 epoch 时，cosine 学习率调度需要正确的 T_max
- 如果 T_max 错误（如用 100 但实际只跑 10 epoch），学习率衰减曲线完全错误
- 影响模型收敛质量和最终 mAP

**best checkpoint 修复：**
- 全量训练时如果某轮 val_loader 出问题（如数据损坏导致全部跳过），不应保存伪 best
- best checkpoint 是后续评估和部署的依据，必须是真实验证结果
- 避免用 0.0 的假 loss 做模型选择

## 5. 当前是否可以进入全量训练

**✅ 可以。**

所有已知阻塞项已修复：
- ✅ 数据转换脚本就绪
- ✅ 数据划分脚本就绪
- ✅ 数据验证脚本就绪
- ✅ 训练配置已适配 5 类
- ✅ 训练脚本 dry run 通过
- ✅ scheduler 语义修复
- ✅ best checkpoint 语义修复

**下一步：**
```
1. 全量运行 convert_sanpo_phase1.py（~38k 帧）
2. 全量运行 split_phase1.py
3. 全量运行 verify_phase1.py
4. 全量训练（--epochs 100）
```
