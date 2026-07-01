param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$InitWeights = "",
    [int]$Gpu = 0,
    [int]$Epochs = 1,
    [int]$NumWorkers = 4,
    [int]$BatchSize = 16,
    [int]$ValEvery = 1,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$python = "D:\mlp\anaconda3\python.exe"
$args = @(
    ".\scripts\train\train_ssd.py",
    "--config", $Config,
    "--gpu", "$Gpu",
    "--epochs", "$Epochs",
    "--num-workers", "$NumWorkers",
    "--batch-size", "$BatchSize",
    "--val-every", "$ValEvery",
    "--run-name", $RunName
)

if ($InitWeights -ne "") {
    $args += @("--init-weights", $InitWeights)
}
if ($MaxTrainBatches -gt 0) {
    $args += @("--max-train-batches", "$MaxTrainBatches")
}
if ($MaxValBatches -gt 0) {
    $args += @("--max-val-batches", "$MaxValBatches")
}

Write-Host "[RemoteTrain] cwd=$projectRoot"
Write-Host "[RemoteTrain] command=$python $($args -join ' ')"
& $python @args
