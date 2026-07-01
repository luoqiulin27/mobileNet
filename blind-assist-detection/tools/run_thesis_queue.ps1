# ============================================================================
# run_thesis_queue.ps1
# 论文第4章消融实验顺序训练队列
#
# 在远程机器 lab6001 上运行，按顺序执行：
#   A3 eval → A4 train+eval → B1 train+eval → B2 train+eval
#   → B3 train+eval → B4 train+eval
#
# 用法（在远程机器上）：
#   powershell -ExecutionPolicy Bypass -File tools\run_thesis_queue.ps1
#
# 或后台启动：
#   powershell -Command "Start-Process powershell -ArgumentList `
#     '-ExecutionPolicy Bypass -File tools\run_thesis_queue.ps1' `
#     -WindowStyle Hidden -RedirectStandardOutput queue_stdout.log `
#     -RedirectStandardError queue_stderr.log"
# ============================================================================

$ErrorActionPreference = "Stop"

# ---------- paths (remote machine) ----------
$projectRoot = "D:\projects\MobileNet\mobileNet\blind-assist-detection"
$python      = "D:\mlp\anaconda3\python.exe"

Set-Location -LiteralPath $projectRoot

# ---------- helpers ----------
function Write-Stage($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $msg"
}

function Test-TrainingDone($runName) {
    $statusPath = "outputs\runs\$runName\metrics\train_status.json"
    if (-not (Test-Path -LiteralPath $statusPath)) { return $false }
    try {
        $s = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
        # epoch is 0-indexed, so epoch+1 == epochs means done
        return ($s.epoch + 1) -ge $s.epochs
    } catch {
        return $false
    }
}

function Test-EvalDone($runName, $evalType) {
    $outputPath = "outputs\runs\$runName\metrics\$evalType"
    return (Test-Path -LiteralPath $outputPath)
}

function Invoke-Train($config, $runName, $epochs = 100, $batchSize = 16, $numWorkers = 4, $valEvery = 5) {
    $runDir = "outputs\runs\$runName"
    $logDir  = "$runDir\logs"
    $ckptDir = "$runDir\checkpoints"
    $metDir  = "$runDir\metrics"
    New-Item -ItemType Directory -Force -Path $logDir, $ckptDir, $metDir | Out-Null

    $resumePath = "$ckptDir\last.pth"

    $args = @(
        "scripts\train\train_ssd.py",
        "--config", $config,
        "--gpu", "0",
        "--epochs", "$epochs",
        "--batch-size", "$batchSize",
        "--num-workers", "$numWorkers",
        "--val-every", "$valEvery",
        "--run-name", $runName
    )
    if (Test-Path -LiteralPath $resumePath) {
        $args += @("--resume", $resumePath)
        Write-Stage "resuming from $resumePath"
    }

    $stdoutLog = "$logDir\stdout.log"
    $stderrLog = "$logDir\stderr.log"

    Write-Stage "TRAIN START  config=$config run=$runName epochs=$epochs"
    $proc = Start-Process -FilePath $python -ArgumentList $args `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    if ($proc.ExitCode -ne 0) {
        Write-Stage "TRAIN FAILED (exit=$($proc.ExitCode))  stderr tail:"
        if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 30 }
        throw "Training failed for $runName"
    }
    Write-Stage "TRAIN DONE    run=$runName"
}

function Invoke-EvalMap($config, $runName, $split = "val", $confThreshold = $null) {
    $bestPath = "outputs\runs\$runName\checkpoints\best.pth"
    if (-not (Test-Path -LiteralPath $bestPath)) {
        Write-Stage "EVAL-MAP SKIP  no best.pth for $runName"
        return
    }
    $outputPath = "outputs\runs\$runName\metrics\eval_map_${split}.json"

    $args = @(
        "scripts\eval\eval_map.py",
        "--config", $config,
        "--checkpoint", $bestPath,
        "--split", $split,
        "--gpu", "0",
        "--batch-size", "32",
        "--num-workers", "2",
        "--output", $outputPath
    )
    if ($confThreshold -ne $null) {
        $args += @("--conf-threshold", "$confThreshold")
    }

    Write-Stage "EVAL-MAP START  run=$runName split=$split conf=$confThreshold"
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "eval_map failed for $runName" }
    Write-Stage "EVAL-MAP DONE   -> $outputPath"
}

function Invoke-EvalSafety($config, $runName, $split = "val", $confThreshold = $null) {
    $bestPath = "outputs\runs\$runName\checkpoints\best.pth"
    if (-not (Test-Path -LiteralPath $bestPath)) {
        Write-Stage "EVAL-SAFETY SKIP  no best.pth for $runName"
        return
    }
    $outputPath = "outputs\runs\$runName\metrics\eval_safety_${split}.json"

    $args = @(
        "scripts\eval\eval_safety_detection.py",
        "--config", $config,
        "--checkpoint", $bestPath,
        "--split", $split,
        "--gpu", "0",
        "--batch-size", "32",
        "--num-workers", "2",
        "--output", $outputPath
    )
    if ($confThreshold -ne $null) {
        $args += @("--conf-threshold", "$confThreshold")
    }

    Write-Stage "EVAL-SAFETY START  run=$runName split=$split conf=$confThreshold"
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "eval_safety failed for $runName" }
    Write-Stage "EVAL-SAFETY DONE   -> $outputPath"
}

function Invoke-TrainAndEval($tag, $config, $runName, $epochs = 100, $needsTraining = $true) {
    Write-Stage "============================================================"
    Write-Stage "EXPERIMENT: $tag"
    Write-Stage "============================================================"

    if ($needsTraining) {
        if (Test-TrainingDone $runName) {
            Write-Stage "TRAIN SKIP  already complete (epoch>=epochs)"
        } else {
            Invoke-Train $config $runName $epochs
        }
    } else {
        Write-Stage "TRAIN SKIP  (training not required for $tag)"
    }

    # eval with both conf=0.05 (thesis standard) and conf=0.30 (safety comparison)
    foreach ($conf in @(0.05, 0.30)) {
        $confStr = $conf.ToString("0.00")
        $confSafe = $confStr.Replace(".", "p")

        if (-not (Test-EvalDone $runName "eval_map_val_conf${confSafe}.json")) {
            Invoke-EvalMap $config $runName "val" $conf
        } else {
            Write-Stage "EVAL-MAP SKIP  conf=$confStr already exists"
        }

        if (-not (Test-EvalDone $runName "eval_safety_val_conf${confSafe}.json")) {
            Invoke-EvalSafety $config $runName "val" $conf
        } else {
            Write-Stage "EVAL-SAFETY SKIP  conf=$confStr already exists"
        }
    }

    Write-Stage "EXPERIMENT $tag COMPLETE"
    Write-Stage ""
}


# ============================================================================
#  Cleanup: delete old stale directories (safe to run multiple times)
# ============================================================================
Write-Stage "=== CLEANUP START ==="

$staleRuns = @(
    "bs12_resnet50_probe", "bs12_vgg16_probe", "bs16_resnet50_probe",
    "bs16_vgg16_probe", "bs32_resnet50_probe", "bs32_vgg16_probe",
    "remote_smoke_resnet50_cpu", "remote_smoke_resnet50_cpu2",
    "remote_smoke_vgg16_cpu", "remote_smoke_vgg16_cpu2",
    "sanpo_vgg16_debug_full", "sanpo_vgg16_formal100",
    "sanpo_vgg16_formal100_bs16", "sanpo_vgg16_formal100_bs64",
    "sanpo_ablation_eca_ciou_formal", "sanpo_ablation_eca_ciou_formal100_v2",
    "sanpo_ablation_eca_ciou_formal2", "sanpo_ablation_eca_ciou_formal3",
    "sanpo_ablation_eca_ciou_smoke", "sanpo_ablation_eca_ciou_formal100_bs48",
    "sanpo_ablation_ciou_formal100_bs128", "sanpo_ablation_ciou_formal100_bs256",
    "sanpo_ablation_ciou_formal100_bs384", "sanpo_ablation_ciou_formal100_bs48",
    "sanpo_ablation_ciou_formal100_v2", "sanpo_ablation_ciou_smoke",
    "sanpo_ablation_eca_formal100_bs48", "sanpo_ablation_eca_formal100_v2",
    "sanpo_ablation_eca_ft5", "sanpo_ablation_eca_smoke",
    "sanpo_newsplit_smoke", "sanpo_ch4_ablation_queue",
    "debug_a4_foreground"
)

foreach ($d in $staleRuns) {
    $p = "outputs\runs\$d"
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Stage "CLEAN  removed $p"
    }
}

# stale configs
$staleConfigs = @(
    "src\configs\ssd_vgg16.yaml",
    "src\configs\ssd_resnet50.yaml"
)
foreach ($c in $staleConfigs) {
    if (Test-Path -LiteralPath $c) {
        Remove-Item -LiteralPath $c -Force
        Write-Stage "CLEAN  removed $c"
    }
}

# stale scripts
$staleScripts = @(
    "scripts\train\run_backbone_compare_queue.py"
)
foreach ($s in $staleScripts) {
    if (Test-Path -LiteralPath $s) {
        Remove-Item -LiteralPath $s -Force
        Write-Stage "CLEAN  removed $s"
    }
}

# orphan log files at runs root
Get-ChildItem "outputs\runs\*.log" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Stage "CLEAN  removed orphan log $($_.Name)"
}

Write-Stage "=== CLEANUP DONE ==="
Write-Stage ""


# ============================================================================
#  Queue: thesis experiments in order
# ============================================================================
Write-Stage "============================================================"
Write-Stage "QUEUE START  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Stage "============================================================"

# ---- A3: CIoU (train done, eval only) ----
Invoke-TrainAndEval `
    -tag "A3_CIoU" `
    -config "src\configs\ssd_default_ciou.yaml" `
    -runName "sanpo_ablation_ciou_formal100" `
    -needsTraining $false

# ---- A4: ECA + CIoU ----
Invoke-TrainAndEval `
    -tag "A4_ECA_CIoU" `
    -config "src\configs\ssd_default_eca_ciou.yaml" `
    -runName "sanpo_ablation_eca_ciou_formal100"

# ---- B1: SE ----
Invoke-TrainAndEval `
    -tag "B1_SE" `
    -config "src\configs\ssd_default_se.yaml" `
    -runName "sanpo_ablation_se_formal100"

# ---- B2: CBAM ----
Invoke-TrainAndEval `
    -tag "B2_CBAM" `
    -config "src\configs\ssd_default_cbam.yaml" `
    -runName "sanpo_ablation_cbam_formal100"

# ---- B3: IoU ----
Invoke-TrainAndEval `
    -tag "B3_IoU" `
    -config "src\configs\ssd_default_iou.yaml" `
    -runName "sanpo_ablation_iou_formal100"

# ---- B4: DIoU ----
Invoke-TrainAndEval `
    -tag "B4_DIoU" `
    -config "src\configs\ssd_default_diou.yaml" `
    -runName "sanpo_ablation_diou_formal100"

Write-Stage "============================================================"
Write-Stage "QUEUE COMPLETE  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Stage "============================================================"
