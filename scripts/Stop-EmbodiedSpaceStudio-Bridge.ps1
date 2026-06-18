param(
  [switch]$RemoveContainers
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bridge = Join-Path $Root "KimodoBridge"
Set-Location $Bridge

if ($RemoveContainers) {
  docker compose down
} else {
  docker compose stop ue-bridge text-encoder demo
}
