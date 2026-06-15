param(
    [string]$Project = "D:\projects\MobileNet\mobileNet\blind-assist-detection",
    [string]$Python = "D:\mlp\anaconda3\python.exe"
)

Set-Location $Project

$Jobs = @(
    @{
        Name = "sunrgbd_newsplit_finetune_from_old"
        Config = "src\configs\ssd_sunrgbd_indoor_finetune.yaml"
        Checkpoint = "outputs\runs\sunrgbd_newsplit_finetune_from_old\checkpoints\best.pth"
        Split = "val"
        Conf = "0.2"
        Output = "outputs\metrics\eval_safety_sunrgbd_val_newsplit_finetune_from_old_best.json"
    },
    @{
        Name = "sunrgbd_newsplit_full"
        Config = "src\configs\ssd_sunrgbd_indoor.yaml"
        Checkpoint = "outputs\runs\sunrgbd_newsplit_full\checkpoints\best.pth"
        Split = "val"
        Conf = "0.2"
        Output = "outputs\metrics\eval_safety_sunrgbd_val_newsplit_full_best.json"
    },
    @{
        Name = "sanpo_newsplit_full"
        Config = "src\configs\ssd_default.yaml"
        Checkpoint = "outputs\runs\sanpo_newsplit_full\checkpoints\best.pth"
        Split = "val"
        Conf = "0.3"
        Output = "outputs\metrics\eval_safety_sanpo_val_newsplit_full_best.json"
    }
)

foreach ($Job in $Jobs) {
    if (-not (Test-Path $Job.Checkpoint)) {
        Write-Output "[Skip] $($Job.Name): checkpoint not found: $($Job.Checkpoint)"
        continue
    }

    Write-Output "[Eval] $($Job.Name)"
    & $Python scripts\eval\eval_safety_detection.py `
        --config $Job.Config `
        --checkpoint $Job.Checkpoint `
        --split $Job.Split `
        --gpu 0 `
        --num-workers 2 `
        --batch-size 24 `
        --conf-threshold $Job.Conf `
        --output $Job.Output

    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($Job.Name)"
    }
}
