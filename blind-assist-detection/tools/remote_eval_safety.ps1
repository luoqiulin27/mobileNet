param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Split = "val",
    [int]$Gpu = 0,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 2,
    [double]$ConfThreshold = -1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$python = "D:\mlp\anaconda3\python.exe"
$args = @(
    ".\scripts\eval\eval_safety_detection.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", $Split,
    "--gpu", "$Gpu",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--output", $Output
)

if ($ConfThreshold -ge 0) {
    $args += @("--conf-threshold", "$ConfThreshold")
}

Write-Host "[RemoteEvalSafety] cwd=$projectRoot"
Write-Host "[RemoteEvalSafety] command=$python $($args -join ' ')"
& $python @args
