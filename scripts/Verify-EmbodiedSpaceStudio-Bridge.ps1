$ErrorActionPreference = "Continue"
$Bridge = Split-Path -Parent $PSScriptRoot
$BridgeName = Split-Path -Leaf $Bridge

$bridgeChecks = @(
  "docker-compose.yaml",
  "Dockerfile",
  ".env",
  "ue_bridge_service/kimodo_ue_service/server.py",
  "kimodo-viser/src/viser/client/build",
  "checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors",
  "checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/model.safetensors.index.json",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/adapter_model.safetensors",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_model.safetensors"
)

$missing = @()
foreach ($rel in $bridgeChecks) {
  $path = Join-Path $Bridge $rel
  $label = "$BridgeName/$rel"
  if (Test-Path -LiteralPath $path) {
    Write-Host "OK      $label"
  } else {
    Write-Host "MISSING $label"
    $missing += $label
  }
}

$outputRoot = Join-Path $Bridge "outputs"
if (Test-Path -LiteralPath $outputRoot) {
  Write-Host "OK      $BridgeName/outputs"
} else {
  Write-Host "INFO    $BridgeName/outputs will be created by the start script or Docker Compose."
}

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:18027/bridge/v1/health" -TimeoutSec 5
  Write-Host "UE bridge health reachable: $($health | ConvertTo-Json -Compress)"
} catch {
  Write-Host "UE bridge health not reachable yet. Start it with scripts/Start-EmbodiedSpaceStudio-Bridge.ps1"
}

if ($missing.Count -gt 0) { exit 2 }