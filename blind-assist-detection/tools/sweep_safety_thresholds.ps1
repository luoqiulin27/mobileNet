param(
    [string]$Project = "D:\projects\MobileNet\mobileNet\blind-assist-detection",
    [string]$Python = "D:\mlp\anaconda3\python.exe",
    [string]$Config = "src\configs\ssd_sunrgbd_indoor_finetune.yaml",
    [string]$Checkpoint = "outputs\runs\sunrgbd_newsplit_finetune_from_old\checkpoints\last.pth",
    [string]$OutputDir = "outputs\metrics\threshold_sweep_sunrgbd_finetune_last",
    [string[]]$Thresholds = @("0.05", "0.10", "0.15", "0.20", "0.25", "0.30")
)

Set-Location $Project
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Rows = @()
foreach ($Threshold in $Thresholds) {
    $SafeName = $Threshold.Replace(".", "p")
    $Output = Join-Path $OutputDir ("eval_conf_" + $SafeName + ".json")
    Write-Output "[Sweep] conf=$Threshold"
    & $Python scripts\eval\eval_safety_detection.py `
        --config $Config `
        --checkpoint $Checkpoint `
        --split val `
        --gpu 0 `
        --num-workers 2 `
        --batch-size 24 `
        --conf-threshold $Threshold `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Threshold evaluation failed: $Threshold"
    }

    $Data = Get-Content -Raw $Output | ConvertFrom-Json
    $Rows += [PSCustomObject]@{
        conf_threshold = [double]$Threshold
        mAP = [double]$Data.mAP
        critical_recall = [double]$Data.safety_metrics.critical_recall.recall
        center_critical_recall = [double]$Data.safety_metrics.center_critical_recall.recall
        near_critical_recall = [double]$Data.safety_metrics.near_critical_recall.recall
        center_near_critical_recall = [double]$Data.safety_metrics.center_near_critical_recall.recall
        person_recall = [double]$Data.per_class.person.recall
        box_bag_recall = [double]$Data.per_class.box_bag.recall
        indoor_obstacle_recall = [double]$Data.per_class.indoor_obstacle.recall
    }
}

$SummaryJson = Join-Path $OutputDir "summary.json"
$SummaryCsv = Join-Path $OutputDir "summary.csv"
$Rows | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $SummaryJson
$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryCsv

Write-Output "[Sweep] Summary"
$Rows | Sort-Object center_near_critical_recall -Descending | Format-Table -AutoSize
Write-Output "[Sweep] saved $SummaryJson"
Write-Output "[Sweep] saved $SummaryCsv"
