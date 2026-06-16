$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$checks = @(
  "docker-compose.yaml",
  "Dockerfile",
  ".env",
  "kimodo",
  "MotionCorrection",
  "checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors",
  "checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/model.safetensors.index.json",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_model.safetensors"
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

if (Test-Path -LiteralPath ".env") {
  $envLines = Get-Content -LiteralPath ".env" | Where-Object { $_ -match "^ESS_UNREAL_PROJECT_ROOT=" }
  if ($envLines.Count -gt 0) {
    $ueRoot = ($envLines[0] -replace "^ESS_UNREAL_PROJECT_ROOT=", "").Trim().Trim('"')
    $ueRoot = $ueRoot -replace '/', '\'
    $serverPath = Join-Path $ueRoot "Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py"
    if (Test-Path -LiteralPath $serverPath) {
      Write-Host "OK      ESS_UNREAL_PROJECT_ROOT contains Kimodo UE service."
    } else {
      Write-Host "MISSING ESS_UNREAL_PROJECT_ROOT Kimodo UE service: $serverPath"
      $missing += "ESS_UNREAL_PROJECT_ROOT/Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py"
    }
  } else {
    Write-Host "MISSING ESS_UNREAL_PROJECT_ROOT in .env"
    $missing += "ESS_UNREAL_PROJECT_ROOT"
  }
}

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:18027/bridge/v1/health" -TimeoutSec 5
  Write-Host "UE bridge health reachable: $($health | ConvertTo-Json -Compress)"
} catch {
  Write-Host "UE bridge health not reachable yet. Start it with scripts/Start-EmbodiedSpaceStudio-Bridge.ps1"
}

if ($missing.Count -gt 0) { exit 2 }
