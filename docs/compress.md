# Stage 2 — Compression Method Comparison

## Overview

Four compression methods were applied to the **Stage 1 QLoRA merged checkpoint** (Llama-3.2-1B, fine-tuned on Alpaca), evaluated on the same test split and benchmark as Stage 1 for direct comparability. All runs on NVIDIA T4 GPU.

| Method | Perplexity ↓ | GSM8K Acc | Win Rate vs S1 | Size (MB) | Peak Mem (MB) | TTFT p50 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|--------|-------------|-----------|---------------|-----------|--------------|--------------|-------------|-------------------|
| BnB int8 | **6.446** | 4.2% | **50%** | ~1300* | 539 | 107.4 | 77.3 | 12.9 |
| BnB int4 ★ | 6.653 | 4.2% | 45% | ~700* | 502 | 58.8 | **45.7** | **21.9** |
| AWQ int4 | 6.637 | 0% | 35% | **983** | **488** | **48.5** | 98.8 | 10.1 |
| Depth Prune | 1299.8    | 0% | 0% | 1893 | 575 | 26.3 | 18.4 | 54.4 |
| Distillation | — | — | — | — | — | — | — | — |

**Winner: BnB int4** — best latency-throughput at acceptable quality cost.

> *BnB int8/int4 are load-time quantization — no separate checkpoint is saved to disk. Size estimates based on theoretical bit-width reduction from fp16 baseline (~2.4 GB).
> ★ Winner proceeding to Stage 3.

---

## Results & Analysis

### Quantization: int8 vs int4 vs AWQ

**int8** preserves quality best — perplexity 6.446, win rate 50% vs Stage 1. The 50% win rate means compressed int8 and Stage 1 QLoRA are roughly equivalent in perceived quality at half the precision. The cost: ITL 77ms and throughput 12.9 tok/s are the slowest among quantized methods.

**int4 bitsandbytes** is the practical winner. Perplexity degradation vs int8 is small (6.65 vs 6.45). Win rate drops modestly to 45%. But latency improves dramatically: TTFT 58ms vs 107ms, ITL 45ms vs 77ms, throughput 21.9 vs 12.9 tok/s. For serving workloads, this 2x throughput gain matters more than the marginal quality difference.

**AWQ int4** is interesting but underperforms on T4 specifically. TTFT is the fastest (48ms) but ITL is the worst among quantized methods (98ms) — producing the lowest throughput (10.1 tok/s). This is a known issue: AWQ's GEMM kernels are optimized for Ampere+ (sm_80+) architecture. On Turing (sm_75), the kernel falls back to a less optimal path, making decode phase slower than expected. Win rate also lowest at 35%.

> **Note on AWQ**: despite underperforming on T4 inference, the AWQ checkpoint (983 MB on disk) is the only format natively supported by vLLM without conversion. It will be carried into Stage 3 for vLLM compatibility testing — this is a real-world finding about format lock-in in the compression → serving pipeline.

**Quality degradation vs Stage 1**: all quantized methods show perplexity jump from ~3.4 to ~6.4–6.7. This is expected — these are post-training quantization methods with no re-training. The quality loss reflects the precision reduction in weight representation. With quantization-aware training (QAT) or GPTQ calibration on domain-specific data, this gap could be reduced.

### Depth Pruning — Model Collapse

Pruning 4 of 16 layers (25% of depth) caused **catastrophic quality collapse**: perplexity 1299, 0% GSM8K, 0% win rate. The model output is incoherent.

**Root cause**: magnitude-based scoring is too naive for identifying unimportant layers. The scorer removed layers 1, 2, 5, 6 — all early-to-mid layers with scores 43.0–43.6. But early transformer layers are critical for building token representations. The score distribution across layers was very tight (43.0–49.2), meaning magnitude alone cannot discriminate importance reliably.

Layer scores show a clear pattern in hindsight: scores increase monotonically toward later layers (layer 15 scores 49.2 vs layer 6 at 43.0). A better heuristic would be to protect the first N and last N layers from pruning entirely, only considering middle layers as candidates.

**What would work better**:
- Activation-based scoring with calibration data (implemented but not used here due to time)
- Angular distance pruning (ShortGPT method) — measures cosine similarity of layer input/output; near-identical = redundant
- Protection of first 4 and last 4 layers from pruning candidates
- Iterative pruning with re-evaluation after each layer removal rather than batch removal

Depth pruning throughput (54.4 tok/s) shows the theoretical upside — fewer layers = proportionally faster decode. The architecture change is real and beneficial; the scoring method is what failed.

### Distillation — Skipped

Sequence-level distillation was attempted but encountered NaN gradients during training, consistent with the float16 instability seen in Stage 1 Full FT. Even without NaN issues, estimated training time on T4 would be 30–60 minutes with uncertain quality improvement.

> **Note**: knowledge distillation to a smaller model is a legitimate and often highly effective compression technique. The skip here is a resource constraint, not a methodological rejection. On A100/H100 with bfloat16, distillation to a 0.5B student from a 1B teacher would likely achieve better size reduction than quantization with competitive quality. The technique is worth revisiting with better hardware.

---

## Key Findings

**1. int4 quantization hits the practical sweet spot on T4.**
2x throughput vs int8 with marginal quality degradation. For instruction-following workloads where exact token probability doesn't matter (win rate matters more than perplexity), int4 is the right choice.

**2. AWQ kernel performance is architecture-dependent.**
AWQ on T4 shows fast TTFT but slow ITL — a signature of suboptimal decode kernels on Turing. The same checkpoint on A100 would perform significantly better due to optimized GEMM paths. This highlights that compression method benchmarks are only valid for the specific hardware they're run on.

**3. Magnitude-based depth pruning is unreliable without layer protection heuristics.**
The score range was too narrow (43–49) to reliably separate important from unimportant layers. Aggressive pruning (25% layers) without re-training or calibration causes collapse. A safer approach: prune ≤10% layers, protect boundaries, use activation-based scoring.

**4. bitsandbytes and vLLM are not compatible — format conversion is a real pipeline step.**
BnB int4 winner cannot be directly served via vLLM in Stage 3. AWQ checkpoint must be used instead. This conversion friction is a genuine MLOps concern: compression decisions affect downstream serving options. Documenting this explicitly is one of the key practical contributions of this project.

**5. Peak inference memory is surprisingly similar across quantized methods.**
int8: 539 MB, int4: 502 MB, AWQ: 488 MB. The differences are smaller than expected because peak memory during a single inference run is dominated by KV cache and activations (which are always fp16), not weight storage alone. Compression reduces model load size more than it reduces runtime peak memory for short sequences.

---

## What Proceeds to Stage 3

Two artifacts enter Stage 3:

1. **BnB int4** (HF transformers serving) — quality-latency winner, served via `generate()` with SDPA
2. **AWQ int4** (vLLM serving) — only vLLM-compatible format, carried forward despite T4 limitations

Stage 3 will compare these against the fp16 baseline using TTFT, ITL, and throughput under realistic serving conditions.

---

## What Changes on Better Hardware

| Constraint | T4 (this project) | A100/H100 |
|------------|-------------------|-----------|
| AWQ kernel performance | Suboptimal (Turing fallback) | Native GEMM optimized |
| Distillation | NaN gradients, ~60 min | Stable bfloat16, ~15 min |
| Pruning recovery | No re-training (time constraint) | Fine-tune after prune |
| GPTQ quantization | Feasible but not tested | Standard practice |
