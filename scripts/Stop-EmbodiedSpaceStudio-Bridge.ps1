param(
  [switch]$RemoveContainers
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($RemoveContainers) {
  docker compose down
} else {
  docker compose stop ue-bridge text-encoder demo
}