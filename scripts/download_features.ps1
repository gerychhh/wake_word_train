$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$features = Join-Path $root "data\features"
New-Item -ItemType Directory -Force -Path $features | Out-Null

Write-Host "This will download LARGE files. ACAV100M is ~17GB."
Write-Host "Press Ctrl+C to cancel."

$acav = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
$val  = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

Invoke-WebRequest -Uri $acav -OutFile (Join-Path $features "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
Invoke-WebRequest -Uri $val  -OutFile (Join-Path $features "validation_set_features.npy")

Write-Host "Done."
