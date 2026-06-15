# split_phase1.py 实现结果

## 1. 创建/修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `blind-assist-detection/scripts/convert/split_phase1.py` | **新建** | 主脚本 |

未修改任何其他文件。

## 2. 脚本实现要点

### 核心流程

```
1. 扫描 images/all/ 获取所有 .png 文件 stem
2. 检查 labels/all/ 是否有对应 .txt 文件（记录缺失数）
3. 从 stem 中提取 session_id（按最后一个 '_' 分割）
4. 按 session 分组
5. 固定种子 42 打乱 session 顺序
6. 按 70/15/15 比例切分（session 级别，不拆分同一 session 的帧）
7. 验证无数据泄漏（帧级 + session 级）
8. 写入 train.txt / val.txt / test.txt
9. 生成 split_report.json
```

### session_id 提取规则

```python
# 文件名格式: {session_id}_{frame_number:06d}
# session_id 可能包含下划线，因此从右侧最后一个 '_' 分割
def extract_session_id(stem: str) -> str:
    last_underscore = stem.rfind("_")
    return stem[:last_underscore]

# 例: "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000"
#   → session_id = "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG"
```

### 划分逻辑

```python
# 1. 按 session 分组
session_groups = group_by_session(stems)

# 2. 固定种子打乱 session 顺序
random.seed(42)
random.shuffle(session_keys)

# 3. 按比例分配 session 到 train/val/test
#    优先填 train，再填 val，剩余给 test
#    同一 session 的所有帧进入同一个集合
```

### 泄漏验证

检查 6 个维度：
- 帧级：train∩val, train∩test, val∩test
- session 级：train∩val, train∩test, val∩test

全部为 0 才算 PASS。

## 3. 调试参数

```
--limit-stems N   限制使用的 stem 数量（0=全部）
                  注意: 按 session 顺序取，不拆分 session
```

## 4. 小规模验证命令

```bash
cd D:\project\mobileNet

# 使用当前已有的 3 个样本验证
python blind-assist-detection/scripts/convert/split_phase1.py --limit-stems 3

# 全量划分（需要先运行 convert_sanpo_phase1.py 全量转换）
python blind-assist-detection/scripts/convert/split_phase1.py
```

## 5. 小规模验证结果

| 检查项 | 结果 |
|--------|------|
| 脚本可运行 | ✅ |
| train.txt 格式正确 | ✅ 每行一个 stem，无扩展名 |
| split_report.json 字段完整 | ✅ 包含所有必要字段 |
| 泄漏检查 | ✅ PASS（0 泄漏） |
| session 级划分 | ✅ 1 个 session 全部进入 train |

**说明：** 当前只有 1 个 session 的 3 个样本，因此全部进入 train，val/test 为空。这是符合预期的行为（session 级划分不拆分同一 session）。

## 6. 输出文件格式

### train.txt / val.txt / test.txt

```
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000000
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG_000001
```

- 每行一个 stem（无扩展名，无路径）
- 无空行，无 header

### split_report.json

```json
{
  "total_images": 3,
  "total_labels": 3,
  "total_sessions": 1,
  "train_sessions": 1,
  "val_sessions": 0,
  "test_sessions": 0,
  "train_images": 3,
  "val_images": 0,
  "test_images": 0,
  "missing_label_files": 0,
  "duplicate_stems": 0,
  "leak_check_passed": true,
  "leak_details": { ... }
}
```

## 7. 已知风险与后续建议

| 风险 | 影响 | 建议 |
|------|------|------|
| 当前仅 3 个样本 | 无法验证多 session 划分效果 | 先全量运行 convert，再全量 split |
| session 大小不均匀 | 某些 session 帧数多，划分比例可能偏离 | 可考虑按帧数加权划分（当前未实现） |
| val/test 可能为空 | 如果 session 数太少 | 全量 94 个 session 时不会出现 |

## 8. 后续任务

1. **verify_phase1.py** — 验证转换结果 + 生成 stats.json
2. **修改 ssd_default.yaml** — num_classes 从 9 改为 6
