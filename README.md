# finetune-compress-serve

End-to-end LLM lifecycle project: fine-tuning, compression, and serving —
all on free-tier Google Colab T4 GPU.

## Lifecycle

Base LLM (Llama-3.2-1B/3B)

│
▼

[Stage 1] Fine-Tuning   
Full FT  │  LoRA  │  QLoRA

│
▼ (best method)

[Stage 2] Compression   
Quantization  │  Pruning  │  Distillation

│
▼ (best method)

[Stage 3] Serving   
HF Transformers  │  SDPA/xformers  │  vLLM

## Hardware & Constraints

- **GPU**: NVIDIA T4 (Turing, compute capability 7.5) via Google Colab free tier
- **FlashAttention-2**: NOT supported on Turing — using FlashAttention-1 / SDPA eager / xformers instead
- **vLLM**: runs on T4 but requires `float16` (not `bfloat16`, needs sm_80+)
- **Quantization format**: `bitsandbytes` int4/int8 from training is NOT directly compatible
  with vLLM's AWQ/GPTQ — re-quantization step required before Stage 3 (documented explicitly)

## Environment

```bash
python >= 3.10
pip install -e .
```

## Stage 1 — Fine-Tuning Method Comparison

> In progress

| Method | Perplexity | Task Accuracy | Win Rate | Train Time | Peak VRAM | Checkpoint Size |
|--------|------------|---------------|----------|------------|-----------|-----------------|
| Full FT | - | - | - | - | - | - |
| LoRA | - | - | - | - | - | - |
| QLoRA | - | - | - | - | - | - |

**Winner**: TBD

See full analysis → [`docs/finetune.md`](docs/finetune.md)

## Stage 2 — Compression Method Comparison

> Pending Stage 1

| Method | Perplexity | Size on Disk | Inference Memory | Latency | Quality Δ |
|--------|------------|--------------|-----------------|---------|-----------|
| Quantization (int8) | - | - | - | - | - |
| Quantization (int4) | - | - | - | - | - |
| Pruning | - | - | - | - | - |
| Distillation | - | - | - | - | - |

**Winner**: TBD

See full analysis → [`docs/compress.md`](docs/compress.md)

## Stage 3 — Serving Engine Comparison

> Pending Stage 2

| Engine | TTFT p50 | ITL p50 | ITL p99 | Throughput | Peak Memory | Attention Backend |
|--------|----------|---------|---------|------------|-------------|-------------------|
| HF generate() eager | - | - | - | - | - | - |
| HF generate() SDPA | - | - | - | - | - | - |
| vLLM | - | - | - | - | - | - |

**Winner**: TBD

See full analysis → [`docs/serve.md`](docs/serve.md)

## Key Findings

> Will be populated after all stages complete.

## Limitations

- T4 limits use of FlashAttention-2, bfloat16, and some vLLM optimizations
- What would change on A100/H100: full bfloat16 training, FA2, larger batch sizes,
  no quant format conversion needed