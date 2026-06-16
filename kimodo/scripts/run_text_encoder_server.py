# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import gradio as gr
import numpy as np

from kimodo.model import resolve_target

from .gradio_theme import get_gradio_theme

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
DEFAULT_TEXT = "A person walks and falls to the ground."
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 9550
DEFAULT_TMP_FOLDER = "/tmp/text_encoder/"
DEFAULT_TEXT_ENCODER = "llm2vec"
HF_GATED_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16",
            "llm_dim": 4096,
            "device": "auto",
        },
        "display_name": "LLM2Vec",
    }
}


class DemoWrapper:
    def __init__(self, text_encoder, tmp_folder, startup_error: Exception | None = None):
        self.text_encoder = text_encoder
        self.tmp_folder = tmp_folder
        self.startup_error = startup_error

    def __call__(self, text, filename, progress=gr.Progress()):
        if self.text_encoder is None:
            raise gr.Error(_format_startup_error(self.startup_error))

        # Compute text embedding
        tensor, length = self.text_encoder(text)
        embedding = tensor[:length]
        embedding = embedding.cpu().numpy()

        # Save text embedding
        path = os.path.join(self.tmp_folder, filename)
        np.save(path, embedding)

        output_title = gr.Markdown(visible=True)
        output_text = gr.Markdown(visible=True, value=f"Text: {text}")
        download = gr.DownloadButton(visible=True, value=path)
        return download, output_title, output_text


def _get_env(name: str, default):
    return os.getenv(name, default)


def _format_startup_error(error: Exception | None) -> str:
    if error is None:
        return "Text encoder is unavailable."

    detail = str(error)
    if "gated repo" in detail.lower() or HF_GATED_MODEL in detail:
        return (
            f"Hugging Face access is missing for {HF_GATED_MODEL}. "
            "Request access to the gated model, then refresh the token used inside Docker."
        )

    return f"Text encoder is unavailable: {type(error).__name__}: {detail}"


def _build_text_encoder(name: str, fp32: bool = False):
    if name not in TEXT_ENCODER_PRESETS:
        available = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unknown TEXT_ENCODER='{name}'. Available: {available}")
    preset = TEXT_ENCODER_PRESETS[name]
    target_cls = resolve_target(preset["target"])
    if fp32:
        preset["kwargs"]["dtype"] = "float32"
    return target_cls(**preset["kwargs"])


def _validate_text_encoder(text_encoder):
    prompts = [
        "A person walks forward.",
        "A person jumps upward and lands.",
    ]
    embeddings = []
    for prompt in prompts:
        tensor, length = text_encoder(prompt)
        embeddings.append(tensor[:length].detach().cpu().numpy().reshape(-1).astype(np.float32))

    first, second = embeddings
    if first.size == 0 or second.size == 0:
        raise RuntimeError("Text encoder semantic probe returned an empty embedding.")

    mean_abs_diff = float(np.mean(np.abs(first - second)))
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise RuntimeError("Text encoder semantic probe returned a zero-norm embedding.")

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    if mean_abs_diff < SEMANTIC_MIN_MEAN_ABS_DIFF or cosine > SEMANTIC_MAX_COSINE:
        raise RuntimeError(
            "Text encoder embeddings collapsed "
            f"(mean_abs_diff={mean_abs_diff:.3e}, cosine={cosine:.6f}). "
            "Check that TEXT_ENCODER_BASE_MODEL_HOST points to the LLM2Vec MNTP model, "
            "not the raw Meta-Llama-3-8B-Instruct folder."
        )

    print(f"Text encoder semantic probe passed: mean_abs_diff={mean_abs_diff:.3e}, cosine={cosine:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run text encoder Gradio server.")
    parser.add_argument(
        "--text-encoder",
        default=_get_env("TEXT_ENCODER", DEFAULT_TEXT_ENCODER),
        choices=sorted(TEXT_ENCODER_PRESETS.keys()),
        help="Text encoder preset.",
    )
    parser.add_argument(
        "--tmp-folder",
        default=_get_env("TEXT_ENCODER_TMP_FOLDER", DEFAULT_TMP_FOLDER),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Uses fp32 for the text encoder rather than default bfloat16.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server_name = _get_env("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME)
    server_port = int(_get_env("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT))
    theme, css = get_gradio_theme()
    os.makedirs(args.tmp_folder, exist_ok=True)
    startup_error = None
    try:
        text_encoder = _build_text_encoder(args.text_encoder, args.fp32)
    except Exception as error:
        startup_error = error
        text_encoder = None
        print(f"Text encoder failed to initialize: {type(error).__name__}: {error}")
    display_name = TEXT_ENCODER_PRESETS[args.text_encoder]["display_name"]
    demo_wrapper_fn = DemoWrapper(text_encoder, args.tmp_folder, startup_error=startup_error)

    with gr.Blocks(title="Text encoder", css=css, theme=theme) as demo:
        gr.Markdown(f"# Text encoder: {display_name}")
        gr.Markdown("## Description")
        if startup_error is None:
            gr.Markdown("Get a embeddings from a text.")
        else:
            gr.Markdown(
                "The text encoder service started in degraded mode. "
                "Text embedding requests will fail until the Hugging Face token has access to "
                f"{HF_GATED_MODEL}."
            )
            gr.Markdown(_format_startup_error(startup_error))

        gr.Markdown("## Inputs")
        with gr.Row():
            text = gr.Textbox(
                placeholder="Type the motion you want to generate with a sentence",
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
                type="text",
            )
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                btn = gr.Button("Encode", variant="primary")
            with gr.Column(scale=1):
                clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=3):
                pass

        output_title = gr.Markdown("## Outputs", visible=False)
        output_text = gr.Markdown("", visible=False)
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                download = gr.DownloadButton("Download", variant="primary", visible=False)
            with gr.Column(scale=4):
                pass

        filename = gr.Textbox(
            visible=False,
            value="embedding.npy",
        )

        def clear_fn():
            return [
                gr.DownloadButton(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
            ]

        outputs = [download, output_title, output_text]

        gr.on(
            triggers=[text.submit, btn.click],
            fn=clear_fn,
            inputs=None,
            outputs=outputs,
        ).then(
            fn=demo_wrapper_fn,
            inputs=[text, filename],
            outputs=outputs,
        )

        def download_file():
            return gr.DownloadButton()

        download.click(
            fn=download_file,
            inputs=None,
            outputs=[download],
        )
        clear.click(fn=clear_fn, inputs=None, outputs=outputs)

    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()

