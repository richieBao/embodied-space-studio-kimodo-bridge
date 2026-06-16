$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$envValues = @{}
if (Test-Path -LiteralPath ".env") {
  Get-Content -LiteralPath ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
      $envValues[$Matches[1].Trim()] = $Matches[2].Trim().Trim('"')
    }
  }
}

function Resolve-BridgePath([string]$value) {
  if (-not $value) { return $null }
  $normalized = $value -replace '/', '\'
  if ([System.IO.Path]::IsPathRooted($normalized)) { return $normalized }
  return Join-Path $Root $normalized
}

function Test-RequiredPath([string]$label, [string]$path, [ref]$missing) {
  if ($path -and (Test-Path -LiteralPath $path)) {
    Write-Host "OK      $label"
  } else {
    Write-Host "MISSING $label"
    $missing.Value += $label
  }
}

function Join-OptionalPath([string]$parent, [string]$child) {
  if (-not $parent) { return $null }
  return Join-Path $parent $child
}

$checks = @(
  "docker-compose.yaml",
  "Dockerfile",
  ".env",
  "kimodo",
  "MotionCorrection",
  "checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors",
  "checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml"
)

$missing = @()
foreach ($rel in $checks) {
  $path = Join-Path $Root $rel
  if (Test-Path -LiteralPath $path) {
    Write-Host "OK      $rel"
  } else {
    Write-Host "MISSING $rel"
    $missing += $rel
  }
}

if (-not (Test-Path -LiteralPath ".env")) {
  Write-Host "MISSING .env variables cannot be checked until .env exists."
} else {
  $baseModelRoot = Resolve-BridgePath $envValues["TEXT_ENCODER_BASE_MODEL_HOST"]
  $mntpAdapterRoot = Resolve-BridgePath $envValues["TEXT_ENCODER_MNTP_ADAPTER_HOST"]
  $supervisedAdapterRoot = Resolve-BridgePath $envValues["TEXT_ENCODER_ADAPTER_HOST"]
  $ueRoot = Resolve-BridgePath $envValues["ESS_UNREAL_PROJECT_ROOT"]

  Test-RequiredPath "TEXT_ENCODER_BASE_MODEL_HOST/model.safetensors.index.json" (Join-OptionalPath $baseModelRoot "model.safetensors.index.json") ([ref]$missing)
  Test-RequiredPath "TEXT_ENCODER_MNTP_ADAPTER_HOST/adapter_model.safetensors" (Join-OptionalPath $mntpAdapterRoot "adapter_model.safetensors") ([ref]$missing)
  Test-RequiredPath "TEXT_ENCODER_ADAPTER_HOST/adapter_model.safetensors" (Join-OptionalPath $supervisedAdapterRoot "adapter_model.safetensors") ([ref]$missing)
  Test-RequiredPath "ESS_UNREAL_PROJECT_ROOT Kimodo UE service" (Join-OptionalPath $ueRoot "Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py") ([ref]$missing)
}

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:18027/bridge/v1/health" -TimeoutSec 5
  Write-Host "UE bridge health reachable: $($health | ConvertTo-Json -Compress)"
} catch {
  Write-Host "UE bridge health not reachable yet. Start it with scripts/Start-EmbodiedSpaceStudio-Bridge.ps1"
}

if ($missing.Count -gt 0) { exit 2 }
