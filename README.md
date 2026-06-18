# EmbodiedSpaceStudio Kimodo Bridge

This repository is the user-side Docker bridge for running Kimodo motion generation with the EmbodiedSpaceStudio Unreal Engine plugin. It is packaged for local Docker installation and is not a direct copy of the upstream Kimodo README.

The intended boundary is simple:

- Docker runs Kimodo, the text encoder, the optional browser demo, and the EmbodiedSpaceStudio bridge API.
- Unreal Engine runs any project that has the EmbodiedSpaceStudio / KimodoMotionAuthoring plugin installed.
- The UE plugin connects to the local bridge endpoint, usually `http://127.0.0.1:18027/bridge/v1`.
- This repository can be placed in any user-chosen folder. It does not need to live inside or beside a UE project.

## What Is Included

Included in this repo:

- Kimodo Python source used by the Docker image.
- The local `kimodo-viser` fork required by the interactive demo and Kimodo UI.
- The EmbodiedSpaceStudio bridge API service used by the Docker `ue-bridge` container.
- Docker files for `text-encoder`, `demo`, and `ue-bridge`.
- Small placeholder README files for model directories.
- PowerShell helper scripts under `scripts/`.

Intentionally not included:

- Kimodo checkpoint weights under `checkpoints/`.
- LLM2Vec/text encoder model weights under `text-encoders/`.
- Hugging Face cache files, local outputs, virtual environments, Docker caches, and `node_modules`.

## Install

### A. Download This Repository

Download `embodied-space-studio-kimodo-bridge` from GitHub and unzip it to any local folder, for example:

```text
D:/EmbodiedSpaceStudio/KimodoBridge
```

Open a terminal in that folder before running the commands below.

### B. Download Model Files

Put the Kimodo checkpoint under:

```text
checkpoints/
  Kimodo-SOMA-RP-v1.1/
    config.yaml
    model.safetensors
    stats/
```

Put the text encoder folders under:

```text
text-encoders/
  McGill-NLP/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

The large model files are intentionally excluded from GitHub and from the Docker build context.

### C. Configure `.env`

Copy the example file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Recommended settings when models live inside this repository:

```text
HF_HOME_HOST=./.cache/huggingface
TEXT_ENCODER_BASE_MODEL_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
TEXT_ENCODER_MNTP_ADAPTER_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter
TEXT_ENCODER_ADAPTER_HOST=./text-encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
UE_BRIDGE_HOST_PORT=18027
TEXT_ENCODER_DEVICE=auto
KIMODO_OUTPUT_HOST=./outputs
```

`KIMODO_OUTPUT_HOST` is the host folder where generated results are written. For the best UE experience, set it to an absolute path such as:

```text
KIMODO_OUTPUT_HOST=D:/EmbodiedSpaceStudio/KimodoBridge/outputs
```

The helper start script sets `KIMODO_OUTPUT_HOST` to an absolute `outputs` path automatically if it is not already set in the terminal environment.

### D. Build And Start Docker

With Docker Desktop installed and running, use the helper script:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -Build
```

After the first build, start without rebuilding:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1
```

Equivalent Docker commands:

```powershell
docker compose build
docker compose up -d text-encoder ue-bridge
```

To also start the browser-based Kimodo demo:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -WithDemo
```

or:

```powershell
docker compose up -d demo
```

### E. Connect From Unreal Engine

Open any Unreal project that has the EmbodiedSpaceStudio / KimodoMotionAuthoring plugin installed. The plugin should connect to:

```text
http://127.0.0.1:18027/bridge/v1
```

The Docker bridge is independent from the UE project folder. Multiple compatible UE projects can use the same local bridge by connecting to the same endpoint.

## Service URLs

- UE bridge health: `http://127.0.0.1:18027/bridge/v1/health`
- Available models: `http://127.0.0.1:18027/bridge/v1/models`
- Kimodo demo, when enabled: `http://127.0.0.1:7860`
- Text encoder service: `http://127.0.0.1:9550`

## Verify The Installation

```powershell
.\scripts\Verify-EmbodiedSpaceStudio-Bridge.ps1
```

This checks the Docker bridge files, local bridge API service, required checkpoint files, required text encoder files, and whether the bridge health endpoint is reachable.

## Stop The Bridge

```powershell
.\scripts\Stop-EmbodiedSpaceStudio-Bridge.ps1
```

Remove containers:

```powershell
.\scripts\Stop-EmbodiedSpaceStudio-Bridge.ps1 -RemoveContainers
```

## Notes For This Release

This release updates the Docker packaging for user-side installation:

- `kimodo-viser` is vendored locally in this repository.
- Docker no longer clones `https://github.com/nv-tlabs/kimodo-viser.git` during build.
- `kimodo-viser/src/viser/client/build` is included so the runtime does not need to rebuild the web client with npm.
- The `ue-bridge` container runs the bridge API service packaged in this repository and no longer mounts a specific Unreal project.
- Generated outputs are written to `KIMODO_OUTPUT_HOST`, usually this repository's `outputs/` folder.
- `node_modules`, Hugging Face caches, checkpoints, text encoder weights, generated outputs, and local virtual environments are excluded from Git and Docker build context.
- Docker default requirements no longer install `py-soma-x` from GitHub. The optional SOMA layer skin path still requires the SOMA package if you enable that feature manually.
- `MotionCorrection` expects Docker-installed `pybind11-dev` and `libeigen3-dev`; it will not fetch those dependencies from GitHub/GitLab by default.

## Troubleshooting

If Docker cannot find model files:

- Confirm `.env` points to existing host paths.
- Run `.\scripts\Verify-EmbodiedSpaceStudio-Bridge.ps1`.
- Check `docker compose config` to see the resolved bind mount paths.

If Unreal cannot connect:

- Confirm `ue-bridge` is running with `docker compose ps`.
- Open `http://127.0.0.1:18027/bridge/v1/health` in a browser.
- Check logs with `docker compose logs --tail 120 ue-bridge`.

If Unreal cannot open generated results:

- Set `KIMODO_OUTPUT_HOST` in `.env` to an absolute Windows path.
- Restart `ue-bridge` with `docker compose up -d --force-recreate ue-bridge`.

If GPU memory is limited:

- Set `TEXT_ENCODER_DEVICE=cpu` in `.env`.
- Restart `text-encoder` with `docker compose up -d --force-recreate text-encoder`.

## Upstream Credits And License

Kimodo is an NVIDIA motion generation project. This bridge packages the parts needed for EmbodiedSpaceStudio integration and local Docker deployment. See `LICENSE`, `ATTRIBUTIONS.MD`, and the model license pages for the licenses that apply to source code, third-party dependencies, and downloaded model weights.