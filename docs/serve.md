# Stage 3 — Serving Engine Comparison

## Overview

Five serving configurations were benchmarked on **NVIDIA T4 GPU**, comparing HF transformers (eager/SDPA attention) against vLLM (PagedAttention + continuous batching) across three model variants: fp16 baseline, AWQ int4, and the (collapsed) pruned model from Stage 2.

This stage was designed to test specific hypotheses from Stage 2 findings, not to run an unstructured sweep.

| Engine | Model | TTFT/ITL p50 (N=1) | Throughput @N=1 | Throughput @N=16 | Peak Mem |
|--------|-------|---------------------|-----------------|-------------------|----------|
| HF eager | fp16 | 26.4 ms | 37.8 tok/s | — (no batching) | 2366 MB |
| HF SDPA | fp16 | 28.7 ms | 34.9 tok/s | — (no batching) | 2366 MB |
| vLLM fp16 ★ | fp16 | 12.6 ms | **77.5 tok/s** | **1127.8 tok/s** | n/a* |
| vLLM AWQ | int4 | 40.9 ms | 24.2 tok/s | 381.8 tok/s | n/a* |
| vLLM Pruned| fp16 (collapsed) | 9.9 ms | 102.1 tok/s | 1412.2 tok/s | n/a* |

**Winner: vLLM fp16** — best throughput at full quality, no compression artifacts needed.

> *vLLM manages GPU memory outside the standard PyTorch allocator (its own memory pool for PagedAttention). `torch.cuda.max_memory_allocated()` does not capture this correctly — measured as 0 across all vLLM runs. This is a measurement limitation, not an actual zero-memory claim. See note below.

---

## Hypothesis Testing — Results

### Hypothesis 1: Pure serving engine effect (fp16, no compression)

**Confirmed.** Same model, same precision, same hardware — only the serving engine changed.

- HF eager: 37.8 tok/s
- vLLM fp16: 77.5 tok/s

**2.05x throughput improvement from the serving engine alone**, with zero changes to model quality. This isolates exactly what PagedAttention and continuous batching contribute even at N=1 — likely from more efficient KV cache memory layout and kernel scheduling, not batching benefit yet (that shows up at higher N).

### Hypothesis 2: Does vLLM's optimized AWQ kernel fix the ITL regression seen in Stage 2?

**Partially confirmed — kernel helps, but underlying inefficiency remains.**

Stage 2 (HF naive AWQ loading): ITL 98.8 ms
Stage 3 (vLLM AWQ, proper GEMM kernel): ITL 40.9 ms

vLLM's AWQ kernel is **2.4x faster** than the naive HF loading path — confirming that part of the Stage 2 regression was due to HF's unoptimized AWQ integration, not AWQ itself.

However, vLLM AWQ ITL (40.9 ms) is still **3.2x slower** than vLLM fp16 ITL (12.6 ms). The remaining gap is attributable to AWQ's GEMM kernel being designed for Ampere+ tensor cores; on Turing (sm_75), it still falls back to a less optimal compute path. **The hypothesis is confirmed directionally** (proper kernel reduces regression) **but not eliminated** (architecture mismatch persists).

### Hypothesis 3: Does the pruned model's throughput advantage hold under continuous batching?

**Confirmed — and the gap is consistent across all concurrency levels.**

| Concurrency | fp16 | Pruned | Pruned advantage |
|-------------|------|--------|-------------------|
| N=1  | 77.5 tok/s   | 102.1 tok/s  | +32% |
| N=4  | 322.3 tok/s  | 402.6 tok/s  | +25% |
| N=8  | 605.5 tok/s  | 781.6 tok/s  | +29% |
| N=16 | 1127.8 tok/s | 1412.2 tok/s | +25% |

The compute advantage (fewer layers = less FLOPs per token) holds consistently at ~25-32% across all batch sizes. Continuous batching does not erase or amplify this gap meaningfully — it's a fixed architectural speedup.

**Critical caveat**: this model has a perplexity of 1299 (vs 3.4 for fp16) — it is non-functional. This result demonstrates that throughput and continuous batching efficiency are completely orthogonal to model quality. A broken model can have excellent serving metrics. **Throughput numbers must never be read without the corresponding quality numbers.**

---

## Concurrency Scaling Analysis

All three vLLM configurations show near-linear throughput scaling from N=1 to N=16:

- fp16: 77.5 → 1127.8 tok/s (14.5x scaling factor)
- AWQ: 24.2 → 381.8 tok/s (15.8x scaling factor)
- Pruned: 102.1 → 1412.2 tok/s (13.8x scaling factor)

This is the expected signature of continuous batching working correctly — GPU utilization improves as more requests share the same forward pass, and throughput scales close to linearly with batch size until compute or memory bandwidth saturates (not yet reached at N=16 on T4 for a 1B model).

**Important finding**: AWQ's kernel inefficiency is preserved across all concurrency levels — it never "catches up" to fp16 even with batching. At N=16, AWQ delivers 381.8 tok/s vs fp16's 1127.8 tok/s — still a 3x gap. **Continuous batching amplifies absolute throughput for all methods proportionally; it does not compensate for kernel-level inefficiency.**

---

## Counter-Intuitive Finding: HF SDPA Underperforms Eager

SDPA (Scaled Dot Product Attention, PyTorch's built-in optimized attention) is generally expected to match or beat eager attention. Here:

- HF eager: 37.8 tok/s
- HF SDPA: 34.9 tok/s (8% slower)

SDPA's p99 latency is also notably worse (110.9 ms vs 45.3 ms for eager) — suggesting higher variance, not just lower average throughput.

**Likely explanation**: for short sequences and small batch size (N=1) on a 1B parameter model, SDPA's kernel dispatch overhead (deciding between flash/memory-efficient/math backends) may outweigh its computational benefit. SDPA's advantages typically materialize at longer sequences or larger batch sizes where the optimized memory access pattern matters more. On T4 specifically, SDPA cannot use the flash-attention backend (Turing unsupported) and falls back to the memory-efficient or math backend, which may have selection overhead not present in straightforward eager execution.

This finding suggests **attention backend choice should be benchmarked, not assumed**, especially on older hardware with limited backend support.

---

## Verified Attention Backend (vLLM)

vLLM was confirmed via initialization logs to **not use FlashAttention-2** on T4, consistent with the documented Turing constraint. vLLM selected its fallback backend automatically (XFormers memory-efficient attention path), as expected for compute capability 7.5 hardware. This confirms the project's hardware constraint assumptions were correctly anticipated rather than discovered as a surprise failure.

---

## Measurement Limitation: vLLM Peak Memory

`peak_mem_mb` reads as 0.0 across all vLLM experiments. This is **not an actual measurement of zero memory usage** — it's an artifact of vLLM managing its own GPU memory pool (`gpu_memory_utilization=0.90` pre-allocates a fixed pool for PagedAttention) outside PyTorch's standard caching allocator. `torch.cuda.max_memory_allocated()` only tracks PyTorch-allocated memory, which vLLM bypasses.

**Correct approach for future work**: use `nvidia-smi` polling or vLLM's internal `LLM.llm_engine.model_executor.driver_worker.model_runner` memory stats, or simply report the configured `gpu_memory_utilization` percentage of total VRAM as the operating envelope (in this case, 90% of T4's 15GB ≈ 13.5GB reserved regardless of actual usage).

---

## Key Findings

**1. Serving engine choice matters as much as compression.**
vLLM fp16 (no compression at all) outperforms every HF configuration by 2x+ throughput. Before reaching for quantization or pruning, switching serving engines is the highest-leverage, zero-quality-cost optimization available.

**2. Optimized kernels matter more than the compression format itself.**
The AWQ ITL regression seen in Stage 2 was 60% attributable to HF's naive integration (98.8ms → 40.9ms when moved to vLLM's proper kernel). The remaining gap is genuine hardware/architecture mismatch (Turing vs Ampere+ GEMM optimization), not a flaw in AWQ as a method.

**3. Continuous batching scales throughput proportionally, not correctively.**
A model with poor per-token efficiency (AWQ on T4) stays proportionally behind across all concurrency levels. Batching is not a substitute for kernel-level optimization — it multiplies whatever efficiency you start with.

**4. Throughput must always be reported with quality, never alone.**
The pruned model is empirically the fastest configuration in this entire project (1412 tok/s @ N=16) and also the most useless (perplexity 1299). This is the clearest demonstration in the whole pipeline of why Stage 2's "quality vs efficiency tradeoff" framing matters — efficiency-only leaderboards are actively misleading.

**5. Backend assumptions should be verified, not trusted.**
SDPA underperforming eager attention on T4 at small batch sizes is counter-intuitive and would have gone unnoticed without direct benchmarking. The project's principle of "verify which attention backend is actually used, don't assume" (stated upfront in the original constraints) paid off here.

---

## Final Recommendation

For deployment on T4-class hardware with this model and workload profile:

**Use vLLM with the fp16 merged checkpoint.** It delivers the best throughput-quality combination without any compression complexity, format conversion, or kernel mismatch risk. Compression (int4/AWQ) only becomes worthwhile when VRAM capacity — not throughput — is the binding constraint (e.g., serving a larger model than fp16 would allow on the same GPU).

---

## What Changes on Better Hardware

| Constraint | T4 (this project) | A100/H100 |
|------------|--------------------|-----------|
| AWQ kernel performance | Turing fallback, 3.2x slower than fp16 | Native GEMM, near-parity with fp16 |
| FlashAttention | Not available (FA1/SDPA/XFormers only) | FA2 available, ~2x decode speedup |
| SDPA backend selection | Limited backend options on Turing | Full backend selection including FA2 |
| Max practical batch size | Limited by 15GB VRAM | Much higher with 40-80GB VRAM |
