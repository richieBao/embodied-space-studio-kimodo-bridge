param(
  [switch]$WithDemo,
  [switch]$Build
)

$ErrorActionPreference = "Stop"
$Bridge = Split-Path -Parent $PSScriptRoot
Set-Location $Bridge

if (-not (Test-Path -LiteralPath (Join-Path $Bridge "docker-compose.yaml"))) {
  throw "docker-compose.yaml not found. Run this script from the KimodoBridge/scripts folder."
}

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
