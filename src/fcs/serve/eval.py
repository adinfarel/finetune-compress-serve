import json
import numpy as np
from pathlib import Path

EXPERIMENTS = {
    "hf_eager":    "results/stage3/hf_eager/results.json",
    "hf_sdpa":     "results/stage3/hf_sdpa/results.json",
    "vllm_fp16":   "results/stage3/vllm_fp16/results.json",
    "vllm_awq":    "results/stage3/vllm_awq/results.json",
    "vllm_pruned": "results/stage3/vllm_pruned/results.json",
}

def load_all_results() -> dict:
    results = {}
    for name, path in EXPERIMENTS.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                results[name] = json.load(f)
        else:
            print(f"  WARNING: {path} not found yet")
    return results

def extract_n1_metrics(r: dict) -> dict:
    if "concurrency_sweep" in r:
        return r.get("n1_metrics", {})
    return r

def print_comparison_table(results: dict):
    cols = ["ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms",
            "throughput_tok_per_sec", "peak_mem_mb"]
    
    header = f"{'Experiment':<18}" + "".join(f"{c[:16]:>18}" for c in cols)
    print("\n" + "="*len(header))
    print("  Stage 3 — Serving Comparison (N=1)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    
    for name, r in results.items():
        m = extract_n1_metrics(r)
        row = f"{name:<18}"
        for c in cols:
            val = m.get(c, r.get(c, "-"))
            row += f"{str(val):>18}"
        print(row)

    print("="*len(header))

def print_concurrency_table(results: dict):
    """
    Print throughput scaling across concurrency levels untuk vLLM experiments.
    """
    vllm_exps = {k: v for k, v in results.items() if "vllm" in k and "concurrency_sweep" in v}
    if not vllm_exps:
        return

    print("\n" + "="*70)
    print("  vLLM Throughput Scaling — Concurrency Sweep")
    print("="*70)
    print(f"{'Experiment':<18} {'N=1':>12} {'N=4':>12} {'N=8':>12} {'N=16':>12}")
    print("-"*70)

    for name, r in vllm_exps.items():
        sweep = {s["concurrency"]: s for s in r["concurrency_sweep"]}
        row = f"{name:<18}"
        for n in [1, 4, 8, 16]:
            s = sweep.get(n, {})
            tok_s = s.get("throughput_tok_per_sec", "-")
            row += f"{str(tok_s):>12}"
        print(row)

    print("="*70)

def save_summary(results: dict, output_dir: str = "results/stage3"):
    """Save aggregated summary JSON."""
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "n1": extract_n1_metrics(r),
            "peak_mem_mb": r.get("peak_mem_mb"),
            "concurrency_sweep": r.get("concurrency_sweep", []),
        }

    out = Path(output_dir) / "stage3_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out}")
    return summary