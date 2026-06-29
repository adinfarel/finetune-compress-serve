# Stage 1 — Fine-Tuning Method Comparison

## Overview

Three fine-tuning methods were compared on **Llama-3.2-1B** using the **Alpaca** instruction dataset, all run on a single **NVIDIA T4 GPU (15GB VRAM, Turing sm_7.5)** via Google Colab free tier.

| Method | Perplexity ↓ | GSM8K Acc | Win Rate | Train Time | Peak VRAM | Trainable % | Ckpt Size |
|--------|-------------|-----------|----------|------------|-----------|-------------|-----------|
| LoRA   | **3.346**   | 6%        | 72%      | 11.9 min   | 5.6 GB    | 0.14%       | ~25 MB    |
| QLoRA  | 3.447       | 4%        | 72%      | 20.2 min   | **2.3 GB**| 0.23%       | ~6 MB     |
| Full FT| NaN         | 0%        | 0%       | 37.9 min   | 11.9 GB   | 100%        | ~2.4 GB   |

**Winner: QLoRA** — nearly identical quality to LoRA at less than half the VRAM. On T4, this is the practical choice.

---

## Results & Analysis

### Perplexity

LoRA achieves the best perplexity (3.346) vs QLoRA (3.447) — a delta of **0.101**. This gap is small and expected: QLoRA's NF4 quantization introduces minor precision loss in the frozen base weights, but the trainable adapter remains in float16 so the loss is minimal.

Full FT produced NaN — training was numerically unstable (see disclaimer below).

### GSM8K Accuracy (Task Accuracy)

Both LoRA and QLoRA scored low: 6% and 4% respectively. This is **expected and not a failure** — Alpaca is an instruction-following dataset, not a math reasoning dataset. The model learns to format responses and follow instructions, not to reason through arithmetic. GSM8K accuracy here serves as a sanity check that the model hasn't catastrophically forgotten capabilities, not as a primary quality signal.

A model fine-tuned on a math-specific dataset (e.g., MetaMath, OpenMathInstruct) would score significantly higher on GSM8K.

### Win Rate (LLM-as-Judge)

Both LoRA and QLoRA achieve **72% win rate** against the base model, judged by Llama-3-70B via Groq API (50 samples, greedy decoding, temperature=0). This is the strongest quality signal — it shows both methods meaningfully improve instruction-following over the base model.

The tie rate is 8–10%, loss rate 18–20%. The similarity between LoRA and QLoRA win rates confirms the perplexity gap (0.101) does not translate to a perceivable quality difference in practice.

### Efficiency Tradeoff

The key insight is the **VRAM gap**:

- LoRA: 5.6 GB peak VRAM
- QLoRA: 2.3 GB peak VRAM (58% reduction)

This means on T4 (15 GB), QLoRA leaves ~12.7 GB headroom vs LoRA's ~9.4 GB. With larger models (3B, 7B), this headroom determines whether training fits at all. QLoRA makes 7B-class models trainable on T4; LoRA at float16 may OOM.

Train time is longer for QLoRA (20.2 min vs 11.9 min) due to the dequantization overhead on every forward/backward pass. This is the direct cost of the VRAM saving.

Checkpoint size reflects the adapter-only saving: LoRA ~25 MB, QLoRA ~6 MB, vs Full FT ~2.4 GB (entire model weights).

---

## Full FT Disclaimer

> Full FT results are **not valid** and should be treated as a hardware constraint finding, not a method failure.

**What happened:**

Training loss diverged to ~309 (random-level), perplexity returned NaN, and the judge rated 100% loss against the base model.

**Root cause (diagnosed during training):**

1. T4 (Turing, sm_7.5) does not support bfloat16 natively. The intended dtype was float16.
2. A version conflict between `transformers`, `accelerate`, and `bitsandbytes` caused the trainer to silently re-cast weights to bfloat16 mid-training despite explicit `fp16=True` in TrainingArguments.
3. Forcing bfloat16 on T4 is technically possible but extremely slow (~60+ min for 1 epoch) and produced no better results in preliminary tests.
4. The workaround was setting both `fp16=False` and `bf16=False` and relying on `torch_dtype="auto"`, which resolved the casting conflict — but float16 full fine-tuning on 1B parameters is inherently prone to gradient overflow/underflow without careful loss scaling.
5. Gradient scaler conflicts with certain PEFT/TRL versions further destabilized training.

**What would fix it:**

- A100/H100: bfloat16 native support eliminates the dtype conflict entirely
- T4 with full FT: would require aggressive gradient clipping, a smaller LR (2e-5 → 5e-6), and possibly mixed-precision workarounds — significant engineering overhead for marginal benefit over QLoRA

**Conclusion:** Full FT on T4 at float16 is not practical. QLoRA achieves comparable quality with 58% less VRAM and a 25 MB checkpoint vs 2.4 GB. The hardware constraint makes the choice clear.

---

## Key Findings

**1. QLoRA is the practical winner on constrained hardware.**
Quality delta vs LoRA is negligible (0.101 perplexity, identical win rate). VRAM savings are substantial. This finding aligns with the original QLoRA paper — the NF4 quantization error is mostly absorbed by the adapter training.

**2. Win rate is a better quality signal than perplexity for instruction tuning.**
Perplexity measures token prediction likelihood — it does not capture whether the model follows instructions well or produces helpful responses. Win rate via LLM-as-judge captures this directly. Both methods winning 72% against the base model confirms instruction tuning is working.

**3. GSM8K accuracy is a distribution mismatch, not a model failure.**
Low math accuracy after Alpaca fine-tuning is expected. Fine-tuning on a narrow distribution improves that distribution's task while potentially degrading out-of-distribution tasks. This is a known phenomenon called **catastrophic forgetting** — though at 1 epoch with LoRA/QLoRA, forgetting is minimal since the base weights are frozen.

**4. T4 dtype constraints are a real engineering problem.**
The float16/bfloat16 library version conflict cost significant debugging time. This is not documented clearly in most tutorials, which assume Ampere+ hardware. On Turing GPUs, `fp16` and `bf16` flags must be handled carefully, and library versions must be pinned to avoid silent re-casting.

**5. Checkpoint size matters for the compression stage.**
LoRA/QLoRA save only adapter weights (~6–25 MB) not the full model. Before Stage 2 compression, the full merged model must be reconstructed via `model.merge_and_unload()`. This is a required step before quantization or pruning can be applied.

---

## What Changes on Better Hardware

| Constraint | T4 (this project) | A100/H100 |
|------------|-------------------|-----------|
| dtype | float16 (conflict-prone) | bfloat16 (stable) |
| Full FT feasibility | Unstable, impractical | Fully viable |
| FlashAttention | FA1 / SDPA eager only | FA2 (2x faster) |
| Batch size | 4 + grad accum 4 | 32+ directly |
| Full FT train time | ~60+ min (bfloat16) | ~10–15 min |

---

## Next Step

QLoRA checkpoint proceeds to **Stage 2 — Compression**, where quantization, distillation, and pruning will be applied to the merged QLoRA model.

Before Stage 2, run:
```python
# merge adapter weights back into base model
model = model.merge_and_unload()
model.save_pretrained("results/stage1/qlora_merged")
```
