$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectRoot = "D:\projects\MobileNet\mobileNet"
$dataRoot = Join-Path $projectRoot "data\SUNRGBD"
$zipPath = Join-Path $dataRoot "SUNRGBD.zip"
$toolboxPath = Join-Path $dataRoot "SUNRGBDtoolbox.zip"
$extractMarker = Join-Path $dataRoot "SUNRGBD\kv1"
$toolboxMarker = Join-Path $dataRoot "SUNRGBDtoolbox\Metadata"
$logDir = Join-Path $dataRoot "_download_logs"

New-Item -ItemType Directory -Force -Path $dataRoot, $logDir | Out-Null

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$Destination
    )
    if (Test-Path $Destination) {
        Write-Host "Exists: $Destination"
        return
    }
    Write-Host "Downloading $Url"
    curl.exe -L --fail --retry 5 --retry-delay 3 $Url -o $Destination
}

Download-IfMissing -Url "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip" -Destination $zipPath
Download-IfMissing -Url "https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip" -Destination $toolboxPath

if (-not (Test-Path $extractMarker)) {
    tar.exe -xf $zipPath -C $dataRoot
}

if (-not (Test-Path $toolboxMarker)) {
    tar.exe -xf $toolboxPath -C $dataRoot
}

$summary = @(
    "SUNRGBD_zip=$(Get-Item $zipPath).Length",
    "Toolbox_zip=$(Get-Item $toolboxPath).Length",
    "Extracted=$(Test-Path $extractMarker)",
    "ToolboxExtracted=$(Test-Path $toolboxMarker)"
)
$summary -join "`n" | Set-Content (Join-Path $logDir "sunrgbd_status.txt")
Get-Content (Join-Path $logDir "sunrgbd_status.txt")
