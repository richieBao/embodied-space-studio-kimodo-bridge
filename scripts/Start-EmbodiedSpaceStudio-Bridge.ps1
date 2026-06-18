param(
  [switch]$WithDemo,
  [switch]$Build
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bridge = Join-Path $Root "KimodoBridge"
Set-Location $Bridge

$services = @("text-encoder", "ue-bridge")
if ($WithDemo) { $services += "demo" }

if ($Build) {
  docker compose build
}

docker compose up -d @services
Write-Host "Bridge startup requested."
Write-Host "UE bridge: http://127.0.0.1:18027/bridge/v1/health"
Write-Host "Models:    http://127.0.0.1:18027/bridge/v1/models"
if ($WithDemo) { Write-Host "Demo:      http://127.0.0.1:7860" }
