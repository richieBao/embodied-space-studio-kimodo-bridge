# EmbodiedSpaceStudio Kimodo Bridge

This repository is the user-side Docker bridge for running Kimodo motion generation with the
EmbodiedSpaceStudio Unreal Engine plugin. It is not a standalone mirror of the upstream Kimodo
README. The expected use case is:

- Unreal Engine runs the `EmbodiedSpaceStudio` project and the `KimodoMotionAuthoring` plugin.
- Docker runs the Kimodo Python services, text encoder service, and UE bridge service.
- The UE plugin talks to the local bridge at `127.0.0.1:18027`.

## Expected Folder Layout

Place this repository next to the Unreal project folder:

```text
EmbodiedSpaceStudio/
  UnrealProject/
    EmbodiedSpaceStudio.uproject
    Plugins/
      KimodoMotionAuthoring/
  KimodoBridge/
    docker-compose.yaml
    Dockerfile
    kimodo/
    kimodo-viser/
    scripts/
```

The folder name can differ, but `docker-compose.yaml` mounts `../UnrealProject` into the
container. If your Unreal project is elsewhere, edit the `ue-bridge` volume in
`docker-compose.yaml`.

## What Is Included

Included in this repo:

- Kimodo Python source used by the Docker image.
- The local `kimodo-viser` fork required by the interactive demo and Kimodo UI.
- Docker files for `text-encoder`, `demo`, and `ue-bridge`.
- Small placeholder README files for model directories.
- PowerShell helper scripts under `scripts/`.

Intentionally not included:

- Kimodo checkpoint weights under `checkpoints/`.
- LLM2Vec/text encoder model weights under `text-encoders/`.
- Hugging Face cache files, virtual environments, Docker caches, and `node_modules`.

## First-Time Setup

### 1. Requirements

- Windows with Docker Desktop.
- NVIDIA GPU support for Docker if running generation on GPU.
- The EmbodiedSpaceStudio Unreal project and `KimodoMotionAuthoring` plugin installed beside this
  folder.
- Downloaded Kimodo checkpoint and text encoder model files.

### 2. Create `.env`

From this folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

Adjust the host paths if you changed the folder layout. Relative paths such as
`./checkpoints/...` are recommended when the models live inside this repository.

Important variables:

- `HF_HOME_HOST`: host Hugging Face cache directory.
- `TEXT_ENCODER_BASE_MODEL_HOST`: base LLM2Vec model directory.
- `TEXT_ENCODER_MNTP_ADAPTER_HOST`: MNTP adapter directory.
- `TEXT_ENCODER_ADAPTER_HOST`: supervised adapter directory.
- `UE_BRIDGE_HOST_PORT`: local host port used by Unreal, default `18027`.
- `TEXT_ENCODER_DEVICE`: use `auto`, `cuda`, or `cpu`.

### 3. Put Model Files In Place

Expected checkpoint layout:

```text
checkpoints/
  Kimodo-SOMA-RP-v1.1/
    config.yaml
    model.safetensors
    stats/
```

Expected text encoder layout:

```text
text-encoders/
  McGill-NLP/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

The verification script checks for the key files used by the Docker services.

## Start The Bridge

From this folder:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -Build
```

After the first build, start without rebuilding:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1
```

To also start the browser-based Kimodo demo:

```powershell
.\scripts\Start-EmbodiedSpaceStudio-Bridge.ps1 -WithDemo
```

Equivalent Docker commands:

```powershell
docker compose build
docker compose up -d text-encoder ue-bridge
docker compose up -d demo
```

Service URLs:

- UE bridge health: `http://127.0.0.1:18027/bridge/v1/health`
- Available models: `http://127.0.0.1:18027/bridge/v1/models`
- Kimodo demo, when enabled: `http://127.0.0.1:7860`
- Text encoder service: `http://127.0.0.1:9550`

## Verify The Installation

```powershell
.\scripts\Verify-EmbodiedSpaceStudio-Bridge.ps1
```

This checks:

- The Unreal project and plugin files.
- The Docker bridge files.
- Required checkpoint and text encoder files.
- Whether the UE bridge health endpoint is reachable.

Inside Unreal Engine, open the EmbodiedSpaceStudio project and use the
`KimodoMotionAuthoring` plugin. The plugin should connect to the bridge on
`127.0.0.1:18027` unless you changed `UE_BRIDGE_HOST_PORT`.

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
- `kimodo-viser/src/viser/client/build` is included so the runtime does not need to rebuild the
  web client with npm.
- `node_modules`, Hugging Face caches, checkpoints, text encoder weights, and local virtual
  environments are excluded from Git and Docker build context.
- Docker default requirements no longer install `py-soma-x` from GitHub. The optional SOMA layer
  skin path still requires the SOMA package if you enable that feature manually.
- `MotionCorrection` expects Docker-installed `pybind11-dev` and `libeigen3-dev`; it will not fetch
  those dependencies from GitHub/GitLab by default.

## Troubleshooting

If Docker cannot find model files:

- Confirm `.env` points to existing host paths.
- Run `.\scripts\Verify-EmbodiedSpaceStudio-Bridge.ps1`.
- Check `docker compose config` to see the resolved bind mount paths.

If Unreal cannot connect:

- Confirm `ue-bridge` is running with `docker compose ps`.
- Open `http://127.0.0.1:18027/bridge/v1/health` in a browser.
- Check logs with `docker compose logs --tail 120 ue-bridge`.

If GPU memory is limited:

- Set `TEXT_ENCODER_DEVICE=cpu` in `.env`.
- Restart `text-encoder` with `docker compose up -d --force-recreate text-encoder`.

## Upstream Credits And License

Kimodo is an NVIDIA motion generation project. This bridge packages the parts needed for
EmbodiedSpaceStudio integration and local Docker deployment. See `LICENSE`, `ATTRIBUTIONS.MD`, and
the model license pages for the licenses that apply to source code, third-party dependencies, and
downloaded model weights.
