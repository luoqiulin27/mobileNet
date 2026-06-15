# RTX 3070 Training Flow

## Current Run

- Project: `D:\projects\MobileNet\mobileNet\blind-assist-detection`
- GPU: NVIDIA GeForce RTX 3070
- Python: `D:\mlp\anaconda3\python.exe`
- Config: `src\configs\ssd_default.yaml`
- Dataset split:
  - train: 4353 images
  - val: 1202 images
  - test: 0 images in the current session-level split
- Checkpoints:
  - `outputs\checkpoints\last.pth`
  - `outputs\checkpoints\best.pth`
- Log files:
  - `outputs\logs\train_3070_20260611_165657.out.log`
  - `outputs\logs\train_3070_20260611_165657.err.log`

## Training Command

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection

D:\mlp\anaconda3\python.exe .\scripts\train\train_ssd.py `
  --config src\configs\ssd_default.yaml `
  --gpu 0 `
  --epochs 100 `
  --num-workers 6 `
  --batch-size 32 `
  --val-every 5
```

## Why This Configuration

- `--gpu 0`: use RTX 3070.
- `--batch-size 32`: keeps the original learning-rate assumption and avoids unstable large-batch behavior.
- `--num-workers 6`: matches the 6 physical CPU cores and keeps data loading active.
- AMP is enabled automatically on CUDA for faster tensor-core training.
- cuDNN benchmark, TF32, and channels-last are enabled in the script for faster CUDA kernels.
- `--val-every 5`: validates every 5 epochs to save time without changing training gradients.
- Data augmentation remains enabled, so this is the safer "keep training quality" run.

## Monitoring

```powershell
nvidia-smi
```

For a clearer live view:

```powershell
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits
```

Check current epoch from the last checkpoint:

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection
python -c "import torch; c=torch.load(r'outputs\checkpoints\last.pth', map_location='cpu'); print(c.get('epoch'), c.get('best_loss'))"
```

## Expected Time

Based on the first saved epochs, the run is roughly around 3 minutes per training epoch on this machine, with extra time on validation epochs. The full 100 epoch run is expected to take about 5 to 6 hours.

## Notes

- Windows Task Manager often shows the default GPU graph as `3D`; switch the graph to `CUDA` or `Compute_0` to see training utilization.
- `best.pth` from earlier short benchmark tests may stay stale until the first formal validation epoch finishes. The current formal run validates at epochs 5, 10, 15, and so on.
- If training is interrupted, resume from `outputs\checkpoints\last.pth`:

```powershell
D:\mlp\anaconda3\python.exe .\scripts\train\train_ssd.py `
  --config src\configs\ssd_default.yaml `
  --gpu 0 `
  --epochs 100 `
  --num-workers 6 `
  --batch-size 32 `
  --val-every 5 `
  --resume outputs\checkpoints\last.pth
```
