$ErrorActionPreference = "Stop"

$projectRoot = "D:\projects\MobileNet\mobileNet"
$downloadRoot = Join-Path $projectRoot "blind-assist-detection\scripts\download"
$logRoot = Join-Path $projectRoot "data\_download_jobs"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$sunScript = Join-Path $downloadRoot "download_sunrgbd.ps1"
$sanpoScript = Join-Path $downloadRoot "sync_sanpo_phase1.py"

Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$sunScript`"" `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logRoot "sunrgbd.out.log") `
    -RedirectStandardError (Join-Path $logRoot "sunrgbd.err.log")

Start-Process -FilePath "python.exe" `
    -ArgumentList "`"$sanpoScript`" --dataset `"D:\projects\MobileNet\mobileNet\data\SANPO-Real-Labeled-Full`" --workers 12 --timeout 60 --retries 3" `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logRoot "sanpo.out.log") `
    -RedirectStandardError (Join-Path $logRoot "sanpo.err.log")

Write-Host "Background downloads started."
Write-Host $logRoot
