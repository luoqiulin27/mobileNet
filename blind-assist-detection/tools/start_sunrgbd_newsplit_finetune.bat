@echo off
set PROJECT=D:\projects\MobileNet\mobileNet\blind-assist-detection
set PYTHON=D:\mlp\anaconda3\python.exe
set STAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set STAMP=%STAMP: =0%
set OUTLOG=%PROJECT%\outputs\logs\train_sunrgbd_newsplit_finetune_%STAMP%.out.log
set ERRLOG=%PROJECT%\outputs\logs\train_sunrgbd_newsplit_finetune_%STAMP%.err.log

cd /d "%PROJECT%"
"%PYTHON%" -u scripts\train\train_ssd.py ^
  --config src\configs\ssd_sunrgbd_indoor_finetune.yaml ^
  --init-weights outputs\runs\sunrgbd_indoor_12class\checkpoints\best.pth ^
  --gpu 0 ^
  --epochs 30 ^
  --num-workers 6 ^
  --batch-size 32 ^
  --val-every 5 ^
  --run-name sunrgbd_newsplit_finetune_from_old ^
  --log-interval 20 ^
  > "%OUTLOG%" 2> "%ERRLOG%"
