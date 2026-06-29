import time
import json
import aysncio #type: ignore
import numpy as np
from typing import Optional
from pathlib import Path

from fcs.serve.config import ServeConfig

def load_vllm_engine(cfg: ServeConfig, model_path: str):
    try:
        from vllm import LLM, SamplingParams #type: ignore
    except ImportError as e:
        raise ImportError("vLLM not installed. Run: pip install vllm") from e
    
    vcfg = cfg.vllm
    
    print(f"Initializing vLLM engine: {model_path}")
    print(f"quantization: {vcfg.quantization}, dtype: {vcfg.dtype}")
    
    llm = LLM(
        model=model_path,
        dtype=vcfg.dtype,
        quantization=vcfg.quantization,
        gpu_memory_utilization=vcfg.gpu_memory_utilization,
        max_model_len=vcfg.max_model_run,
        trust_remote_code=True,
        # encforce_eager=True
        # NOTE: uncomment if CUDA graph OOM in T4
    )
    
    sampling_params = SamplingParams(
        temperature=vcfg.temperature,
        max_tokens=vcfg.max_tokens,
    )
    
    return llm, sampling_params

def benchmark_concurrency(
    llm,
    sampling_params,
    prompts: list[str],
    concurrency: int,
    n_rounds: int = 10,
) -> dict:
    from vllm import SamplingParams #type: ignore
    
    batch_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    
    e2e_latencies = []
    ttfts = []
    itls = []
    
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        t_total = (time.perf_counter() - t0) * 1000  # ms
        
        e2e_latencies.append(t_total)
        
        for out in outputs:
            
            if hasattr(out, "metrics") and out.metrics is not None:
                m = out.metrics
                if m.first_token_time is not None and m.arrival_time is not None:
                    ttfts.append((m.first_token_time - m.arrival_time) * 1000)
                if m.finished_time is not None and m.first_token_time is not None:
                    n_tok = len(out.outputs[0].token_ids)
                    if n_tok > 1:
                        decode_ms = (m.finished_time - m.first_token_time) * 1000
                        itls.append(decode_ms / (n_tok - 1))
            
            else:
                n_tok = len(out.outputs[0].token_ids)
                if n_tok > 0:
                    itls.append(t_total / n_tok / concurrency)
    
    e2e_arr = np.array(e2e_latencies)
    throughput_tok = (
        np.mean([len(o.outputs[0].token_ids) for o in outputs]) * concurrency / (np.mean(e2e_latencies) / 1000)
    )
    
    result = {
        "concurrency": concurrency,
        "e2e_p50_ms": round(float(np.percentile(e2e_arr, 50)), 3),
        "e2e_p99_ms": round(float(np.percentile(e2e_arr, 99)), 3),
        "throughput_tok_per_sec": round(throughput_tok, 2),
    }
    
    if ttfts:
        ttfts_arr = np.array(ttfts)
        result["ttft_p50_ms"] = round(float(np.percentile(ttfts_arr, 50)), 3)
        result["ttft_p99_ms"] = round(float(np.percentile(ttfts_arr, 99)), 3)

    if itls:
        itls_arr = np.array(itls)
        result["itl_p50_ms"] = round(float(np.percentile(itls_arr, 50)), 3)
        result["itl_p99_ms"] = round(float(np.percentile(itls_arr, 99)), 3)

    return result

def run_vllm_benchmark(cfg: ServeConfig) -> dict:
    experiment = cfg.experiment
    output_dir = cfg.output_dir
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    if experiment == "vllm_fp16":
        model_path = cfg.model.fp16_model_path
    elif experiment == "vllm_awq":
        model_path = cfg.model.awq_model_path
    elif experiment == "vllm_pruned":
        model_path = cfg.model.pruned_model_path
    else:
        raise ValueError(f"Unknown vLLM experiment: {experiment}")
    
    llm, sampling_params = load_vllm_engine(cfg, model_path)
    
    prompts = cfg.benchmark.prompts
    concurrency_levels = cfg.benchmark.concurrency_levels
    
    print(f"\nBenchmarking concurrency levels: {concurrency_levels}")
    
    concurrency_results = []
    for n in concurrency_levels:
        print(f"\n  Concurrency N={n}...")
        r = benchmark_concurrency(
            llm=llm,
            sampling_params=sampling_params,
            prompts=prompts,
            concurrency=n,
            n_rounds=10,
        )
        concurrency_results.append(r)
        print(f"    E2E p50: {r['e2e_p50_ms']:.1f} ms | "
              f"Throughput: {r['throughput_tok_per_sec']:.1f} tok/s")
    
    import torch
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    results = {
        "experiment": experiment,
        "model_path": model_path,
        "quantization": cfg.vllm.quantization,
        "peak_mem_mb": round(peak_mem_mb, 2),
        "concurrency_sweep": concurrency_results,
        # N=1 metrics for apple-to-apple vs HF single-request
        "n1_metrics": next(
            (r for r in concurrency_results if r["concurrency"] == 1), {}
        ),
    }
    
    out_path = Path(output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {str(out_path)!r}")
    return results