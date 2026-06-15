# Phase 1 全量数据准备结果

## 1. Convert 阶段结果

| 指标 | 值 |
|------|-----|
| 处理 session 数 | 94（146 个目录中有 52 个为空） |
| 处理图像帧数 | 38,609 |
| 有标注帧数 | 37,619 |
| 无标注帧数 | 990（2.6%） |
| 生成标签文件数 | 38,609（含空标签） |
| 总 bbox 数 | 378,013 |
| 耗时 | 5,401 秒（约 90 分钟） |

### 各类别 bbox 数量

| 类别 | 数量 | 占比 |
|------|------|------|
| person | 80,967 | 21.4% |
| vehicle | 102,398 | 27.1% |
| pole | 38,760 | 10.3% |
| stairs | 3,113 | 0.8% |
| obstacle | 152,775 | 40.4% |
| **总计** | **378,013** | 100% |

### 异常情况

- 52 个空 session 目录已自动跳过
- 990 帧无标注（mask 中无目标类别），已生成空标签文件
- 无转换失败帧

## 2. Split 阶段结果

| 指标 | train | val | test | 总计 |
|------|-------|-----|------|------|
| session 数 | 63 | 15 | 16 | 94 |
| 图像帧数 | 27,119 | 5,802 | 5,688 | 38,609 |
| 占比 | 70.2% | 15.0% | 14.7% | 100% |

### 泄漏检查

| 检查项 | 结果 |
|--------|------|
| session 级泄漏 | ✅ PASS（0 泄漏） |
| 帧级泄漏 | ✅ PASS（0 泄漏） |
| 覆盖率 | ✅ PASS（100% 覆盖） |
| 重复 stem | ✅ 0 个 |
| 缺失标签 | ✅ 0 个 |

### 各 split 类别分布

| 类别 | train | val | test |
|------|-------|-----|------|
| person | 54,614 | 13,621 | 12,732 |
| vehicle | 75,608 | 11,711 | 15,079 |
| pole | 27,373 | 6,297 | 5,090 |
| stairs | 2,183 | 759 | 171 |
| obstacle | 104,070 | 22,742 | 25,963 |

## 3. Verify 阶段结果

### 检查项汇总

| 编号 | 检查项 | 结果 |
|------|--------|------|
| C1 | 图像/标签配对 | ✅ PASS |
| C2 | train 文件存在 | ✅ PASS |
| C2 | val 文件存在 | ✅ PASS |
| C2 | test 文件存在 | ✅ PASS |
| C5 | session 无泄漏 | ✅ PASS |
| C6 | 帧无泄漏 | ✅ PASS |
| C7 | 全覆盖 | ✅ PASS |
| C8 | bbox 范围 | ✅ PASS |
| C9 | class_id 范围 | ✅ PASS |
| C10 | bbox 尺寸 | ✅ PASS |
| C11 | classes.txt 内容 | ✅ PASS |
| **总体** | **ALL PASS** | ✅ |

### 核心统计（stats.json）

| 字段 | 值 |
|------|-----|
| total_images | 38,609 |
| total_labels | 38,609 |
| total_sessions | 94 |
| train_images | 27,119 |
| val_images | 5,802 |
| test_images | 5,688 |
| empty_label_files | 990 |
| missing_label_files | 0 |
| duplicate_stems | 0 |
| invalid_boxes | 0 |
| invalid_class_ids | 0 |
| total_bboxes | 378,013 |
| leak_check_passed | true |

## 4. 风险分析

### 类别不平衡

| 类别 | 数量 | 相对比例 | 风险 |
|------|------|----------|------|
| obstacle | 152,775 | 1.00x（基准） | 低 |
| vehicle | 102,398 | 0.67x | 低 |
| person | 80,967 | 0.53x | 低 |
| pole | 38,760 | 0.25x | 中 |
| stairs | 3,113 | 0.02x | 🔴 高 |

**stairs 类严重不足：** 仅 3,113 个 bbox，是 obstacle 的 2%。test 集中仅 171 个 stairs bbox，评估指标可能不可靠。

### 空标签比例

- 空标签帧数：990 / 38,609 = **2.6%**
- 比例很低，不构成问题

### 数据质量问题

- invalid_boxes：0 ✅
- invalid_class_ids：0 ✅
- missing_label_files：0 ✅
- 无数据质量问题

## 5. 结论与下一步建议

### ✅ 可以直接进入 baseline 训练

**理由：**
1. 所有验证检查 PASS
2. 数据量充足（38,609 帧，378,013 个 bbox）
3. 无数据质量问题
4. 泄漏检查通过
5. 类别分布在合理范围内（除 stairs 偏少外）

### stairs 类偏少的应对建议

| 方案 | 优先级 | 说明 |
|------|--------|------|
| 先不管，直接训练 | ✅ 推荐 | 先看 baseline 效果，再决定是否需要处理 |
| 训练时对 stairs 过采样 | 后续 | 如果 stairs mAP 太低再考虑 |
| 使用 focal loss | 后续 | 自然处理类别不平衡 |
| 降低 stairs 的评估权重 | 后续 | 评估时关注整体 mAP |

### 推荐的下一步动作

```
第 1 步: 全量训练 baseline
  python scripts/train/train_ssd.py --gpu 0 --epochs 100
  → 建立 5 类检测 baseline 性能

第 2 步: 评估 baseline
  → 检查各类别 mAP
  → 特别关注 stairs 的检测效果

第 3 步: 根据结果决定优化方向
  → 如果 stairs mAP 太低：考虑过采样或 focal loss
  → 如果整体 mAP 可接受：进入 SUNRGBD 室内数据扩展
```

## 附录：产物清单

```
data/phase1_sanpo_5class/
├── images/all/              ← 38,609 个 .png 文件
├── labels/all/              ← 38,609 个 .txt 文件
├── configs/classes.txt      ← 5 类名
└── meta/
    ├── train.txt            ← 27,119 个 stem
    ├── val.txt              ← 5,802 个 stem
    ├── test.txt             ← 5,688 个 stem
    ├── conversion_report.json
    ├── split_report.json
    ├── stats.json
    └── verify_report.json
```
