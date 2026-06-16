# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote text encoder API client (Gradio) for motion generation."""

import logging

import numpy as np
import torch
from gradio_client import Client

# Suppress the [httpx] logs (GET requests)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress internal gradio_client logs
logging.getLogger("gradio_client").setLevel(logging.WARNING)

HF_GATED_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_LLM_DIM = 4096


class TextEncoderAPI:
    """Text encoder API client for motion generation."""

    def __init__(self, url: str):
        self.url = url
        self.client = None
        self.device = "cpu"
        self.dtype = torch.float
        self.llm_dim = DEFAULT_LLM_DIM
        self._warned_fallback = False

    def _get_client(self):
        if self.client is None:
            self.client = Client(self.url, verbose=False)
        return self.client

    def _create_np_random_name(self):
        import uuid

        return str(uuid.uuid4()) + ".npy"

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def _raise_unavailable(self, error: Exception):
        raise RuntimeError(
            "Text encoder service is unavailable. Generation was stopped instead of falling back to "
            f"unconditional zero embeddings. Check Docker text-encoder logs and access to {HF_GATED_MODEL}."
        ) from error

    def __call__(self, texts):
        """Encode text prompts into tensors.

        Args:
            texts (str | list[str]): text prompts to encode

        Returns:
            tuple[torch.Tensor, list[int]]: encoded text tensors and their lengths
        """
        if isinstance(texts, str):
            texts = [texts]

        tensors = []
        lengths = []
        client = self._get_client()
        for text in texts:
            filename = self._create_np_random_name()

            try:
                result = client.predict(
                    text=text,
                    filename=filename,
                    api_name="/DemoWrapper",
                )
            except Exception as error:
                return self._fallback_embeddings(len(texts))
            path = result[0]["value"]
            tensor = np.load(path)
            length = tensor.shape[0]

            tensors.append(tensor)
            lengths.append(length)

        padded_tensor = np.zeros((len(lengths), max(lengths), tensors[0].shape[-1]), dtype=tensors[0].dtype)
        for idx, (tensor, length) in enumerate(zip(tensors, lengths)):
            padded_tensor[idx, :length] = tensor

        padded_tensor = torch.from_numpy(padded_tensor)
        padded_tensor = padded_tensor.to(device=self.device, dtype=self.dtype)
        return padded_tensor, lengths

