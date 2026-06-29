import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcs.serve.config import ServeConfig
from fcs.serve.eval import load_all_results, print_comparison_table, print_concurrency_table, save_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Run one serving experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip running experiment",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = ServeConfig.from_yaml(args.config)
    experiment = cfg.experiment
    
    if args.summary_only:
        results = load_all_results()
        print_comparison_table(results)
        print_concurrency_table(results)
        save_summary(results)
        return

    results = {}

    if experiment in ("hf_eager", "hf_sdpa"):
        from fcs.serve.hf_bench import run_hf_benchmark
        results = run_hf_benchmark(cfg)
    
    elif experiment in ("vllm_fp16", "vllm_awq", "vllm_pruned"):
        from fcs.serve.vllm_bench import run_vllm_benchmark
        results = run_vllm_benchmark(cfg)
    
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    print(f"\n{'='*55}")
    print(f"  RESULTS: {experiment}")
    print(f"{'='*55}")
    for k, v in results.items():
        if k not in ("per_prompt", "concurrency_sweep"):
            print(f"  {k:<40} {v}")
    
    if "concurrency_sweep" in results:
        print(f"\n  Concurrency sweep:")
        for s in results["concurrency_sweep"]:
            print(f"    N={s['concurrency']:<4} | "
                  f"E2E p50: {s.get('e2e_p50_ms', '-'):>8} ms | "
                  f"Throughput: {s.get('throughput_tok_per_sec', '-'):>8} tok/s")

    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()