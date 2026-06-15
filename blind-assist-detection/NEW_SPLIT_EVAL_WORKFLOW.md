# New Split And Safety Evaluation Workflow

## Current Goal

This project is now organized around a more reliable path:

1. Build leakage-free train/val/test splits.
2. Ensure every important class has validation and test coverage.
3. Train lightweight detection models without overwriting the current demo checkpoints.
4. Evaluate with both detection metrics and obstacle-assistance safety metrics.

## Data Splits

SANPO uses a session-level split with class-coverage search:

```powershell
D:\mlp\anaconda3\python.exe scripts\convert\split_phase1.py `
  --data-dir data\sanpo_obstacle_8class `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --search-trials 800 `
  --min-val-bboxes 50 `
  --min-test-bboxes 50
```

SUNRGBD uses the same splitter, with fewer trials because it has one image per session:

```powershell
D:\mlp\anaconda3\python.exe scripts\convert\split_phase1.py `
  --data-dir data\sunrgbd_indoor_12class `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --search-trials 120 `
  --min-val-bboxes 30 `
  --min-test-bboxes 30
```

After splitting, always run:

```powershell
D:\mlp\anaconda3\python.exe scripts\convert\verify_phase1.py --data-dir data\sanpo_obstacle_8class
D:\mlp\anaconda3\python.exe scripts\convert\verify_phase1.py --data-dir data\sunrgbd_indoor_12class
```

## Safety Evaluation

The detailed evaluator is:

```text
scripts\eval\eval_safety_detection.py
```

It reports:

- mAP@0.5
- per-class AP, precision, recall
- critical obstacle recall
- center-region critical recall
- near-obstacle critical recall
- center-near critical recall

For blind-assist usage, the center-near critical recall is especially important because it approximates obstacles most likely to affect walking.

## Current Baselines On The New Splits

Old checkpoints evaluated on the new splits:

```text
SANPO val:
  mAP: 0.1649
  center_near_critical_recall: 0.4377

SUNRGBD val:
  mAP: 0.1993
  center_near_critical_recall: 0.5630
```

These are saved under:

```text
outputs\metrics\eval_safety_sanpo_val_old_best_new_split.json
outputs\metrics\eval_safety_sunrgbd_val_old_best_new_split.json
```

## SUNRGBD Model Selection

SUNRGBD experiments on the new split showed:

```text
old checkpoint, val, conf=0.20:
  mAP: 0.1993
  center_near_critical_recall: 0.5630

from-scratch new split checkpoint, val, conf=0.20:
  mAP: 0.1213
  center_near_critical_recall: 0.4658

fine-tuned from old checkpoint, val, conf=0.20:
  mAP: 0.2012
  center_near_critical_recall: 0.5638
```

The fine-tuned model is slightly better than the old model at `conf=0.20`, but the difference is very small.

Threshold sweep on `sunrgbd_newsplit_finetune_from_old\last.pth` showed that lower confidence thresholds are much better for obstacle-assistance recall:

```text
conf=0.05:
  mAP: 0.2197
  center_near_critical_recall: 0.7062

conf=0.10:
  mAP: 0.2137
  center_near_critical_recall: 0.6765

conf=0.20:
  mAP: 0.2012
  center_near_critical_recall: 0.5728
```

However, on the held-out test split at `conf=0.05`, the original old checkpoint is still slightly better overall:

```text
old checkpoint, test, conf=0.05:
  mAP: 0.2470
  center_near_critical_recall: 0.7058

fine-tuned last checkpoint, test, conf=0.05:
  mAP: 0.2417
  center_near_critical_recall: 0.7049
```

Current recommendation:

```text
Keep the web demo on:
  outputs\runs\sunrgbd_indoor_12class\checkpoints\best.pth

Keep indoor demo threshold at:
  conf_threshold = 0.05
```

Do not replace the demo with `sunrgbd_newsplit_finetune_from_old` yet. The useful project improvement from this round is the validated low-threshold safety setting, not a checkpoint swap.

## SANPO Threshold Selection

The SANPO old checkpoint was also evaluated on the new split with the demo-style low threshold:

```text
old SANPO checkpoint, val, conf=0.30:
  mAP: 0.1649
  center_near_critical_recall: 0.4377

old SANPO checkpoint, val, conf=0.05:
  mAP: 0.2097
  center_near_critical_recall: 0.7155

old SANPO checkpoint, test, conf=0.05:
  mAP: 0.2472
  center_near_critical_recall: 0.7948
```

Current outdoor recommendation:

```text
Keep the web demo on:
  outputs\checkpoints\best.pth

Keep outdoor demo threshold at:
  conf_threshold = 0.05
```

For blind-assist use, this low threshold is the safer operating point because missed close obstacles are more harmful than extra candidates. Any downstream reminder layer should filter or group detections rather than raising the detector threshold too early.

## Active Training Runs

New split training should use run directories so the current web demo stays usable.

SUNRGBD full retraining:

```text
outputs\runs\sunrgbd_newsplit_full
```

Monitor it with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\monitor_newsplit_training.ps1 `
  -RunName sunrgbd_newsplit_full `
  -LogPattern train_sunrgbd_newsplit_*.out.log
```

When it finishes, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\evaluate_newsplit_runs.ps1
```

That script evaluates any completed new split runs whose checkpoints exist.

## Next Recommended Order

1. Let `sunrgbd_newsplit_full` finish.
2. Evaluate its `best.pth` with `eval_safety_detection.py`.
3. Compare against `eval_safety_sunrgbd_val_old_best_new_split.json`.
4. If SUNRGBD improves, start SANPO full retraining with `tools\start_sanpo_newsplit_train.bat`.
5. After both are retrained, decide whether to point the web demo to the new checkpoints.
