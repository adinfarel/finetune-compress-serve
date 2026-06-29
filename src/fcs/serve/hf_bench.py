'''
HF benchmark - Eager & SDPA
Single-request latency: TTFT and ITL
'''

import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from fcs.serve.config import ServeConfig

def load_hf_model(
    model_path: str,
    attn_implementation: str,
    dtype=torch.float16
):
    """Load model with different attn implmnttn ["Eager", "SDPA"]"""
    print("Loading model: ", model_path)
    print("Attention backend: ", attn_implementation)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    print(f"Actual attention implementation in config {actual_attn}")
    
    return model, tokenizer

@torch.no_grad()
def measure_single_request(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda"
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_len = inputs["input_ids"].shape[1]
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    total_dcd_ms = (time.perf_counter() - t1) * 1000 # ms
    
    n_generated = output.shape[1] - inputs["input_ids"].shape[1]
    if n_generated > 1:
        t_itl_total = total_dcd_ms - ttft_ms
        itl_ms = (t_itl_total / (n_generated - 1))

    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "n_generated": n_generated,
        "total_decode_ms": t_itl_total
    }

def run_hf_benchmark(cfg: ServeConfig) -> dict:
    experiment = cfg.experiment
    attn_impl = "eager" if experiment == "hf_eager" else "sdpa"
    model_path = cfg.model.fp16_model_path
    output_dir = cfg.output_dir
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_hf_model(
        model_path=model_path,
        attn_implementation=attn_impl,
        dtype=torch.float16
    )
    
    prompts = cfg.benchmark.prompts
    max_new_tokens = cfg.benchmark.max_new_tokens
    warmup_runs = cfg.benchmark.warmup_runs
    benchmark_runs = cfg.benchmark.benchmark_runs
    
    print(f"\nWarmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        measure_single_request(model, tokenizer, prompts[0], max_new_tokens, device)
    
    # benchmark
    print(f"Benchmarking ({benchmark_runs} runs x {len(prompts)} prompts)...")
    all_ttfts, all_itls = [], []
    per_prompt_results = []
    
    for prompt in prompts:
        ttfts, itls = [], []
        for _ in range(benchmark_runs):
            r = measure_single_request(model, tokenizer, prompt, max_new_tokens, device)
            ttfts.append(r['ttft_ms'])
            itls.append(r['itl_ms'])
            
        all_itls.append(ttfts)
        all_ttfts.append(ttfts)
        per_prompt_results.append({
            "prompt": prompt[:60] + "...",
            "ttft_p50": round(float(np.percentile(ttfts, 50)), 3),
            "itl_p50": round(float(np.percentile(itls, 50)), 3),
        })
    
    peak_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2) # mb
    
    ttfts_arr = np.array(all_ttfts)
    itls_arr = np.array(all_itls)
    
    results = {
        "experiment": experiment,
        "attn_implementation": attn_impl,
        "model_path": model_path,
        "concurrency": 1,
        "ttft_p50_ms": round(float(np.percentile(ttfts_arr, 50)), 3),
        "ttft_p99_ms": round(float(np.percentile(ttfts_arr, 99)), 3),
        "itl_p50_ms": round(float(np.percentile(itls_arr, 50)), 3),
        "itl_p99_ms": round(float(np.percentile(itls_arr, 99)), 3),
        "throughput_tok_per_sec": round(1000 / float(np.percentile(itls_arr, 50)), 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "per_prompt": per_prompt_results,
    }
    
    out_path = Path(output_dir) / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved: {str(out_path)!r}")
    return results