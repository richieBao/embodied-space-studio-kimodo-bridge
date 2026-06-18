$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$checks = @(
  "UnrealProject/EmbodiedSpaceStudio.uproject",
  "UnrealProject/Plugins/KimodoMotionAuthoring/EmbodiedSpaceStudio.uplugin",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Source",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Config",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Content",
  "UnrealProject/Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py",
  "KimodoBridge/docker-compose.yaml",
  "KimodoBridge/.env",
  "KimodoBridge/checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors",
  "KimodoBridge/checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml",
  "KimodoBridge/text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/model.safetensors.index.json",
  "KimodoBridge/text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/adapter_model.safetensors",
  "KimodoBridge/text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_model.safetensors"
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

try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:18027/bridge/v1/health" -TimeoutSec 5
  Write-Host "UE bridge health reachable: $($health | ConvertTo-Json -Compress)"
} catch {
  Write-Host "UE bridge health not reachable yet. Start it with Scripts/Start-EmbodiedSpaceStudio-Bridge.ps1"
}

if ($missing.Count -gt 0) { exit 2 }
