param(
    [string]$Project = "D:\projects\MobileNet\mobileNet\blind-assist-detection"
)

$BatPath = Join-Path $Project "tools\start_sunrgbd_newsplit_finetune.bat"
if (-not (Test-Path $BatPath)) {
    throw "Missing batch launcher: $BatPath"
}

$Command = "cmd.exe /c $BatPath"
$Result = wmic process call create "$Command"
$Result

Write-Output ""
Write-Output "Fine-tuning launcher submitted."
Write-Output "Monitor with:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_newsplit_training.ps1 -RunName sunrgbd_newsplit_finetune_from_old -LogPattern train_sunrgbd_newsplit_finetune_*.out.log"
