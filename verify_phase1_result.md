# verify_phase1.py 实现结果

## 1. 创建/修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `blind-assist-detection/scripts/convert/verify_phase1.py` | **新建** | 主脚本 |

未修改任何其他文件。

## 2. 脚本实现的检查项

| 编号 | 检查项 | 通过条件 | 失败时报告 |
|------|--------|----------|-----------|
| C1 | 图像数 == 标签数 | `len(images/) == len(labels/)` | 实际数量差异 |
| C2 | train/val/test 中的 stem 都有对应文件 | 每个 stem 在 images/ 和 labels/ 中都存在 | 缺失数量 |
| C5 | train/val/test session 无交集 | 三个集合的 session_id 两两交集为空 | 泄漏数量 |
| C6 | train/val/test 帧无交集 | 三个集合的 stem 两两交集为空 | 泄漏数量 |
| C7 | train + val + test 覆盖所有帧 | 三集合并集 == images/ 中所有 stem | 未覆盖数量 |
| C8 | bbox 值在 [0,1] 范围内 | cx, cy ∈ [0,1], w, h ∈ (0,1] | 无效 bbox 数 |
| C9 | class_id 在 [0,4] 范围内 | class_id ∈ {0,1,2,3,4} | 无效 class_id 数 |
| C10 | w > 0.01 且 h > 0.01 | 每个 bbox 满足 | 无效 bbox 数 |
| C11 | classes.txt 内容正确 | 5 行，内容为 person/vehicle/pole/stairs/obstacle | 实际内容 |
| C12 | 类别平衡 | 每个 split 中每个类别至少 100 个 bbox | WARN（不阻塞） |

## 3. stats.json 关键字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_images` | int | images/all/ 中的 .png 文件数 |
| `total_labels` | int | labels/all/ 中的 .txt 文件数 |
| `total_sessions` | int | 涉及的 session 总数 |
| `train_images` / `val_images` / `test_images` | int | 各 split 的帧数 |
| `train_sessions` / `val_sessions` / `test_sessions` | int | 各 split 的 session 数 |
| `empty_label_files` | int | 0 字节或无 bbox 的标签文件数 |
| `missing_label_files` | int | 有图像但无标签的文件数 |
| `duplicate_stems` | int | 重复的 stem 数 |
| `leak_check_passed` | bool | session 级 + 帧级泄漏检查是否全部通过 |
| `class_counts` | dict | 全局各类别 bbox 数 |
| `split_class_counts` | dict | 各 split 中各类别 bbox 数 |
| `invalid_boxes` | int | 数值不合法的 bbox 数 |
| `invalid_class_ids` | int | class_id 超出 [0,4] 的 bbox 数 |
| `total_bboxes` | int | bbox 总数 |

## 4. stats.json 与 verify_report.json 的分工

| 文件 | 定位 | 内容 |
|------|------|------|
| `stats.json` | **训练配置用** | 模型训练和数据加载所需的统计字段，结构固定 |
| `verify_report.json` | **调试审计用** | 每个检查项的详细结果、警告列表、耗时，结构灵活 |

训练脚本只读 `stats.json`；人工审查或 CI 流水线读 `verify_report.json`。

## 5. 小规模验证命令

```bash
cd D:\project\mobileNet

# 使用当前已有的 3 个样本验证
python blind-assist-detection/scripts/convert/verify_phase1.py --limit-stems 3

# 全量验证（需要先全量运行 convert + split）
python blind-assist-detection/scripts/convert/verify_phase1.py
```

## 6. 小规模验证结果

| 检查项 | 结果 |
|--------|------|
| C1 图像/标签配对 | PASS |
| C2 train/val/test 文件存在 | PASS |
| C5 session 泄漏 | PASS |
| C6 帧泄漏 | PASS |
| C7 覆盖率 | PASS |
| C8 bbox 范围 | PASS |
| C9 class_id 范围 | PASS |
| C10 bbox 尺寸 | PASS |
| C11 classes.txt | PASS |
| C12 类别平衡 | WARN（预期：仅 3 样本） |
| **总体** | **ALL PASS** |

## 7. 已知风险与下一步建议

| 风险 | 影响 | 建议 |
|------|------|------|
| 当前仅 3 样本 | 无法验证类别平衡和覆盖率 | 全量运行 convert + split 后再全量 verify |
| val/test 为空 | C12 无法检查 val/test 的类别分布 | 全量 94 session 时不会出现 |
| exit code = 1 时有 FAIL | CI 流水线应检查退出码 | 已实现：ALL PASS 时返回 0，有 FAIL 时返回 1 |

## 8. 后续任务

1. **全量运行 convert_sanpo_phase1.py** — 转换全部 38k 帧
2. **全量运行 split_phase1.py** — 按 session 划分
3. **全量运行 verify_phase1.py** — 验证全量数据
4. **修改 ssd_default.yaml** — num_classes 从 9 改为 6
