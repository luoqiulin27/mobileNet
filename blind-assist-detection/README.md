# Blind Assist Detection

## Goal

This project is tuned around one main target:

- accurate obstacle category recognition

The current mainline uses a rebuilt SANPO 8-class dataset, a MobileNetV2-SSD detector, and a local Flask demo for fast visual testing.

## Mainline Dataset

Recommended profile:

- `sanpo_obstacle_8class`

Classes:

- `person`
- `vehicle`
- `rider`
- `animal`
- `stairs`
- `pole`
- `bike_rack`
- `obstacle`

## Project Layout

```text
blind-assist-detection/
  data/                 converted datasets
  outputs/              checkpoints, logs, metrics
  scripts/
    convert/            dataset rebuild pipeline
    eval/               evaluation scripts
    train/              training entrypoints
  frontend/             standalone React + Vite frontend
  src/
    configs/            runtime and model config
    datasets/           dataset logic and SANPO profiles
    inference/          reusable inference predictor
    losses/
    models/
  web_demo/             Flask test frontend
```

## Dataset Rebuild

Run a full SANPO rebuild:

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection
python .\scripts\convert\rebuild_sanpo_dataset.py `
  --profile sanpo_obstacle_8class `
  --dataset ..\data\SANPO-Real-Labeled-Full `
  --output data\sanpo_obstacle_8class `
  --min-area 120 `
  --train-ratio 0.7 `
  --val-ratio 0.15
```

Outputs include:

- `data\sanpo_obstacle_8class\meta\conversion_report.json`
- `data\sanpo_obstacle_8class\meta\train.txt`
- `data\sanpo_obstacle_8class\meta\val.txt`
- `data\sanpo_obstacle_8class\meta\test.txt`
- `data\sanpo_obstacle_8class\meta\stats.json`
- `data\sanpo_obstacle_8class\meta\verify_report.json`

## Training

The default config points to `data\sanpo_obstacle_8class`.

Standard training:

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection
python .\scripts\train\train_ssd.py `
  --config src\configs\ssd_default.yaml `
  --gpu 0 `
  --epochs 100 `
  --num-workers 6 `
  --batch-size 32 `
  --val-every 5
```

Resume from the latest checkpoint:

```powershell
python .\scripts\train\train_ssd.py --config src\configs\ssd_default.yaml --gpu 0 --resume-last
```

Training artifacts:

- `outputs\checkpoints\best.pth`
- `outputs\checkpoints\last.pth`
- `outputs\metrics\train_status.json`
- TensorBoard events in `outputs\logs`

## Web Demo

Start the local frontend:

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection
start_web_demo.bat
```

Open:

- `http://localhost:5000`

Available endpoints:

- `GET /health`
- `GET /api/meta`
- `POST /api/predict`

## Split Frontend

The repo now includes a standalone frontend in `frontend/` for a cleaner API/UI split.

Suggested local workflow:

1. Start the Flask API:

```powershell
cd D:\projects\MobileNet\mobileNet
start_api_server.bat
```

2. Install frontend dependencies:

```powershell
cd D:\projects\MobileNet\mobileNet\blind-assist-detection\frontend
cmd /c npm install
```

3. Start the React frontend:

```powershell
cd D:\projects\MobileNet\mobileNet
start_frontend_demo.bat
```

Development URLs:

- Flask API / legacy demo: `http://127.0.0.1:5000`
- React frontend: `http://127.0.0.1:5173`

Checkpoint loading order:

1. `outputs\checkpoints\best.pth`
2. `outputs\checkpoints\last.pth`

## Notes

- Rebuild the dataset before starting a formal new training run.
- The old `phase1_sanpo_5class` dataset stays only as a historical baseline.
- `verify_report.json` should pass before trusting a new training run.
- The web demo can be used while training is still running because it can fall back to `last.pth`.
