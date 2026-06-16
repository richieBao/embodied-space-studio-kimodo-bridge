param(
  [switch]$WithDemo,
  [switch]$Build
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (!(Test-Path -LiteralPath ".env")) {
  throw "Missing .env. Copy .env.example to .env and edit paths for your machine."
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