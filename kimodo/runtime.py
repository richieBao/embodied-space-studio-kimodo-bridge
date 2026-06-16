"""Runtime helpers for selecting safe execution devices."""

from __future__ import annotations

import os

import torch


def _normalize_arch(arch: str) -> str:
    if arch.startswith("sm_") or arch.startswith("compute_"):
        return arch.split("_", 1)[1]
    return arch


def _cuda_arch_supported(device_index: int = 0) -> bool:
    capability = torch.cuda.get_device_capability(device_index)
    capability_token = f"{capability[0]}{capability[1]}"
    supported_archs = {_normalize_arch(arch) for arch in torch.cuda.get_arch_list()}
    return capability_token in supported_archs


def choose_torch_device(default: str = "auto", env_var: str = "KIMODO_DEVICE") -> str:
    """Choose a safe torch device, falling back to CPU when CUDA is unsupported."""
    requested = os.environ.get(env_var, default).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not requested.startswith("cuda"):
        return requested

    if not torch.cuda.is_available():
        print(f"{env_var} requested CUDA but no CUDA device is available. Falling back to CPU.")
        return "cpu"

    try:
        if _cuda_arch_supported(0):
            return requested
        capability = torch.cuda.get_device_capability(0)
        supported = ", ".join(torch.cuda.get_arch_list())
        print(
            f"CUDA device capability sm_{capability[0]}{capability[1]} is not supported by this PyTorch build "
            f"({supported}). Falling back to CPU."
        )
        return "cpu"
    except Exception as error:
        print(f"Could not validate CUDA compatibility ({type(error).__name__}: {error}). Falling back to CPU.")
        return "cpu"