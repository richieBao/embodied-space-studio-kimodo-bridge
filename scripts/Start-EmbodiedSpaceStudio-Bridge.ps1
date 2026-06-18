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

function Resolve-BridgeHostPath([string]$Value, [string]$Fallback) {
  $trimmed = $Value.Trim().Trim('"')
  if ([string]::IsNullOrWhiteSpace($trimmed)) {
    $trimmed = $Fallback
  }
  if ([System.IO.Path]::IsPathRooted($trimmed)) {
    return $trimmed
  }
  return [System.IO.Path]::GetFullPath((Join-Path $Bridge $trimmed))
}

$envOutputHost = ""
$envFile = Join-Path $Bridge ".env"
if (Test-Path -LiteralPath $envFile) {
  $envOutputLine = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^\s*KIMODO_OUTPUT_HOST\s*=' } | Select-Object -First 1
  if ($envOutputLine) {
    $envOutputHost = ($envOutputLine -replace '^\s*KIMODO_OUTPUT_HOST\s*=\s*', '').Trim()
  }
}

if ([string]::IsNullOrWhiteSpace($env:KIMODO_OUTPUT_HOST)) {
  $env:KIMODO_OUTPUT_HOST = Resolve-BridgeHostPath $envOutputHost "outputs"
}
New-Item -ItemType Directory -Force -Path $env:KIMODO_OUTPUT_HOST | Out-Null

$services = @("text-encoder", "ue-bridge")
if ($WithDemo) { $services += "demo" }

if ($Build) {
  docker compose build
}

docker compose up -d @services
Write-Host "Bridge startup requested."
Write-Host "UE bridge: http://127.0.0.1:18027/bridge/v1/health"
Write-Host "Models:    http://127.0.0.1:18027/bridge/v1/models"
Write-Host "Output:    $env:KIMODO_OUTPUT_HOST"
if ($WithDemo) { Write-Host "Demo:      http://127.0.0.1:7860" }