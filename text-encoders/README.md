# Text encoders

Text encoder model weights are not stored in this Git repository.

Kimodo's LLM2Vec text encoder uses three pieces:

- Gated raw base model: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- MNTP LoRA adapter: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
- Supervised LoRA adapter: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

The bridge maps those host folders into the paths expected by Kimodo. Do not use a separate `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter` Hugging Face repository. In this bridge, `TEXT_ENCODER_MNTP_ADAPTER_HOST` should point to the downloaded `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` adapter folder.

Review and accept the relevant model licenses before downloading.
