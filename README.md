# Embodied Space Studio Kimodo Bridge

This repository contains the Docker runtime used by the **Embodied Space Studio** Unreal Engine plugin to run Kimodo motion generation locally.

The Unreal Engine plugin itself is distributed through Fab and is **not** included in this repository. Install the Fab plugin first, then use this bridge repository to run the local Kimodo services that the plugin connects to.

## What is included

- Dockerfile and Docker Compose configuration for the Kimodo runtime.
- NVIDIA Kimodo source code, licensed under Apache-2.0.
- MotionCorrection source used by Kimodo.
- Helper scripts for starting, stopping, and verifying the bridge.
- Documentation for downloading model weights and configuring `.env`.

## What is not included

This repository intentionally does **not** include:

- The Embodied Space Studio Unreal Engine plugin source code.
- Fab-distributed plugin assets or Unreal project content.
- Kimodo checkpoint weights.
- LLM2Vec / Llama text encoder weights.
- `.env` files with user-specific paths.
- `.venv`, `.cache`, `node_modules`, generated outputs, or any file over GitHub's practical size limits.

## License notes

The Kimodo codebase is licensed under Apache-2.0. See `LICENSE` and `ATTRIBUTIONS.MD`.

Model checkpoints and text encoder weights are licensed separately by their publishers and must be downloaded by the user from the official model pages. Review and accept the relevant model licenses before downloading or using the weights.

Useful references:

- Kimodo upstream repository: <https://github.com/nv-tlabs/kimodo>
- Kimodo-SOMA-RP-v1.1 model: <https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1>
- NVIDIA Open Model License: <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/>
- LLM2Vec base model: <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp>
- LLM2Vec MNTP adapter: <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter>
- LLM2Vec supervised adapter: <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised>

## Prerequisites

1. Windows with Docker Desktop installed and running.
2. WSL2 backend enabled in Docker Desktop.
3. NVIDIA GPU driver and Docker GPU support for local GPU generation.
4. Git installed.
5. The Embodied Space Studio Unreal Engine plugin installed from Fab in an Unreal project.
6. Enough disk space for model downloads. The LLM2Vec text encoder files can require tens of GB.

Confirm Docker is available:

```powershell
docker --version
docker compose version
```

## Folder layout

After cloning this repository and downloading the required models, the bridge folder should look like this:

```text
embodied-space-studio-kimodo-bridge/
  docker-compose.yaml
  Dockerfile
  .env
  checkpoints/
    Kimodo-SOMA-RP-v1.1/
      config.yaml
      model.safetensors
  text-encoders/
    McGill-NLP/
      LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
      LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
      LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
  scripts/
    Start-EmbodiedSpaceStudio-Bridge.ps1
    Stop-EmbodiedSpaceStudio-Bridge.ps1
    Verify-EmbodiedSpaceStudio-Bridge.ps1
```

## Download model weights

Download the Kimodo checkpoint:

- <https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1>

Place the downloaded files under:

```text
checkpoints/Kimodo-SOMA-RP-v1.1/
```

At minimum, the bridge expects:

```text
checkpoints/Kimodo-SOMA-RP-v1.1/model.safetensors
checkpoints/Kimodo-SOMA-RP-v1.1/config.yaml
```

Download the text encoder folders from Hugging Face:

- <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp>
- <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter>
- <https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised>

Place them under:

```text
text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

The base text encoder folder should include files such as `model.safetensors.index.json` and its shard files. The adapter folders should include `adapter_model.safetensors`.

## Configure `.env`

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your machine:

```text
ESS_UNREAL_PROJECT_ROOT=C:/Users/YourName/Documents/Unreal Projects/YourProject
HF_HOME_HOST=./.cache/huggingface
HOST_USER=YourWindowsUserName
SERVER_PORT=7860
TEXT_ENCODER_BASE_MODEL_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
TEXT_ENCODER_ADAPTER_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
TEXT_ENCODER_MNTP_ADAPTER_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter
UE_BRIDGE_HOST_PORT=18027
```

`ESS_UNREAL_PROJECT_ROOT` must point to the Unreal project that has the Fab plugin installed. The bridge runs this file from the Fab plugin:

```text
Plugins/KimodoMotionAuthoring/Resources/Python/kimodo_ue_service/server.py
```

Use forward slashes in Windows paths inside `.env`, for example:

```text
ESS_UNREAL_PROJECT_ROOT=C:/Users/Alice/Documents/Unreal Projects/MyProject
```

## Start the bridge

From PowerShell:

```powershell
cd C:\Path\To\embodied-space-studio-kimodo-bridge
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1
```

This starts the required services:

- `text-encoder`
- `ue-bridge`

To rebuild the Docker image before starting:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -Build
```

To also start the optional Kimodo demo:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -WithDemo
```

The equivalent compose command is:

```powershell
docker compose up -d text-encoder ue-bridge
```

Docker Desktop will show the compose project as `kimodo-main`.

## Verify the bridge

Run:

```powershell
.\scripts\Verify-EmbodiedSpaceStudio-Bridge.ps1
```

Expected local endpoints:

```text
http://127.0.0.1:18027/bridge/v1/health
http://127.0.0.1:18027/bridge/v1/models
```

The expected model key is:

```text
kimodo-soma-rp-v1.1
```

## Stop the bridge

Stop containers but keep them available for faster restart:

```powershell
.\scripts\Stop-EmbodiedSpaceStudio-Bridge.ps1
```

Remove compose containers:

```powershell
.\scripts\Stop-EmbodiedSpaceStudio-Bridge.ps1 -RemoveContainers
```

## Use from Unreal Engine

1. Start Docker Desktop.
2. Start this bridge runtime.
3. Open the Unreal project with the Embodied Space Studio Fab plugin installed.
4. Open the plugin panel.
5. In the Generate tab, check bridge health or click **Connect Docker Bridge**.
6. Refresh models and load `kimodo-soma-rp-v1.1`.

If Unreal cannot connect, first open:

```text
http://127.0.0.1:18027/bridge/v1/health
```

Then check container status:

```powershell
docker ps --filter "name=text-encoder"
docker ps --filter "name=ue-bridge"
docker compose logs -f ue-bridge
```