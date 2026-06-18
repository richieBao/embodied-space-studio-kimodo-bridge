$ErrorActionPreference = "Continue"
$Bridge = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $Bridge
$BridgeName = Split-Path -Leaf $Bridge

$checks = @(
  "UnrealProject/EmbodiedSpaceStudio.uproject",
  "UnrealProject/Plugins/KimodoMotionAuthoring/EmbodiedSpaceStudio.uplugin",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Source",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Config",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Content",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py"
)

$bridgeChecks = @(
  "docker-compose.yaml",
  ".env",
  "checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors",
  "checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/model.safetensors.index.json",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/adapter_model.safetensors",
  "text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_model.safetensors"
)

$missing = @()
foreach ($rel in $checks) {
  $path = Join-Path $WorkspaceRoot $rel
  if (Test-Path -LiteralPath $path) {
    Write-Host "OK      $rel"
  } else {
    Write-Host "MISSING $rel"
    $missing += $rel
  }
}

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

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:18027/bridge/v1/health" -TimeoutSec 5
  Write-Host "UE bridge health reachable: $($health | ConvertTo-Json -Compress)"
} catch {
  Write-Host "UE bridge health not reachable yet. Start it with scripts/Start-EmbodiedSpaceStudio-Bridge.ps1"
}

if ($missing.Count -gt 0) { exit 2 }
