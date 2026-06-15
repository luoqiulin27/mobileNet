param(
    [string]$Project = "D:\projects\MobileNet\mobileNet\blind-assist-detection",
    [string]$RunName = "sunrgbd_newsplit_full",
    [string]$LogPattern = "train_sunrgbd_newsplit_*.out.log",
    [int]$Tail = 60
)

$StatusPath = Join-Path $Project "outputs\runs\$RunName\metrics\train_status.json"
$LogDir = Join-Path $Project "outputs\logs"

Write-Output "== GPU =="
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits

Write-Output ""
Write-Output "== Python Processes =="
wmic process where "name='python.exe'" get ProcessId,CommandLine /format:list

Write-Output ""
Write-Output "== Train Status =="
if (Test-Path $StatusPath) {
    Get-Content -Raw $StatusPath
} else {
    Write-Output "Missing status: $StatusPath"
}

Write-Output ""
Write-Output "== Latest Log =="
$LatestLog = Get-ChildItem $LogDir -Filter $LogPattern |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestLog) {
    Write-Output $LatestLog.FullName
    Get-Content -Tail $Tail $LatestLog.FullName
} else {
    Write-Output "No log matched: $LogPattern"
}
