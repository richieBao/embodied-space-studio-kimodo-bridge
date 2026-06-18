param(
  [switch]$RemoveContainers
)

$ErrorActionPreference = "Stop"
$Bridge = Split-Path -Parent $PSScriptRoot
Set-Location $Bridge

if (-not (Test-Path -LiteralPath (Join-Path $Bridge "docker-compose.yaml"))) {
  throw "docker-compose.yaml not found. Run this script from the KimodoBridge/scripts folder."
}

if ($RemoveContainers) {
  docker compose down
} else {
  docker compose stop ue-bridge text-encoder demo
}
