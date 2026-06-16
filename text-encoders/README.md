# Text encoders

Text encoder model weights are not stored in this Git repository.

Kimodo's LLM2Vec text encoder uses three pieces. The recommended local layout keeps all three folders under `text-encoders/McGill-NLP/` because the Kimodo runtime resolves the base model as `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`.

- Gated raw base model source: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- MNTP LoRA adapter source: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
- Supervised LoRA adapter: https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

Recommended local layout:

```text
text-encoders/
  McGill-NLP/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter/
    LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

Download the gated raw Meta Llama files into `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`.
Download the MNTP adapter from `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`, but save it locally as `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter`. There is no separate Hugging Face repository named `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-adapter`.

Review and accept the relevant model licenses before downloading.
