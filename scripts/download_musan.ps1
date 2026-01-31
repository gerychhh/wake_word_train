$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "data\backgrounds"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Write-Host "Downloading MUSAN (~11GB) from OpenSLR..."
$musanUrl = "https://www.openslr.org/resources/17/musan.tar.gz"
$musanTar = Join-Path $dest "musan.tar.gz"

Invoke-WebRequest -Uri $musanUrl -OutFile $musanTar

Write-Host "Extracting..."
tar -xzf $musanTar -C $dest

Write-Host "MUSAN extracted to $dest\musan"
Write-Host "You may delete $musanTar afterwards."
