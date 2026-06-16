# Text encoders

Text encoder model weights are not stored in this Git repository.

Kimodo's LLM2Vec text encoder uses three pieces:

- Gated raw base model: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- MNTP LoRA adapter source: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
- Supervised LoRA adapter: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

Recommended local layout:

```text
text-encoders/
  meta-llama/
    Meta-Llama-3-8B-Instruct/
  McGill-NLP/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

Download the MNTP adapter from `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`, but save it locally as `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter`. There is no separate Hugging Face repository named `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter`.

Review and accept the relevant model licenses before downloading.
