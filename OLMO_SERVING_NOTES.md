# OLMo-3-32B-Think-DPO Serving Notes

## Setup

- **Model**: `allenai/Olmo-3-32B-Think-DPO` (BF16, Olmo2ForCausalLM architecture)
- **Framework**: vLLM 0.15.1
- **GPU**: NVIDIA H200 (143GB VRAM), model uses ~60 GiB
- **Weights**: downloaded to `/workspace/fuzzy_elicitation/models/`
- **Server**: OpenAI-compatible API on `http://localhost:8000`
- **Served model name**: `olmo-3-32b-think-dpo`

### How to start

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python serve_olmo.py
```

Weights are cached in `models/` after first run, so subsequent starts are fast (~4 min for model load + CUDA graph warmup).

### serve_olmo.py config

- `--dtype bfloat16` (native precision)
- `--max-model-len 16384` (model supports 32K but 16K is plenty for our use case)
- `--download-dir ./models`
- `--trust-remote-code`

## Behavior observations

The model is a "Think" variant (pre-RL). It tends to produce chain-of-thought reasoning in its output, but does not reliably use `<think>`/`</think>` tags on its own — the thinking just comes out as raw text.

### In-context example prompting for think tags

Tested with a system prompt that instructs the model to use `<think>` tags and provides a one-shot example (Aconcagua geography question). Results were mixed:

| Query type | Example | Used tags? | Notes |
|---|---|---|---|
| Arithmetic | 137 * 256 | No | Raw CoT, hit token limit |
| Creative | Write a haiku | No | Got stuck counting syllables |
| Philosophy | Consequentialism pros/cons | No | Rambling reasoning, hit limit |
| Science | Why is sky blue | **Yes** | Clean separation, structured answer |
| Probability | Drawing blue balls | **Yes** | Clean separation, correct answer with \boxed{} |

The model follows the think-tag format for more structured/factual questions but ignores it for open-ended or creative tasks, where it dumps raw chain-of-thought and often exhausts the token budget before producing a final answer.

### Takeaway

vLLM serving works fine. The inconsistent think-tag behavior is a model-level issue (likely pre-RL), not a serving issue. May need to experiment with stronger prompting, higher max_tokens, or just accept the raw CoT and post-process it.
