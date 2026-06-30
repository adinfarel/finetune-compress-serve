import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results() -> dict:
    paths = {
        "hf_eager":    "results/stage3/hf_eager/results.json",
        "hf_sdpa":     "results/stage3/hf_sdpa/results.json",
        "vllm_fp16":   "results/stage3/vllm_fp16/results.json",
        "vllm_awq":    "results/stage3/vllm_awq/results.json",
        "vllm_pruned": "results/stage3/vllm_pruned/results.json",
    }
    results = {}
    for name, p in paths.items():
        path = Path(p)
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"WARNING: {p} not found")
    return results


COLORS = {
    "hf_eager":    "#9B9B9B",
    "hf_sdpa":     "#6B6B6B",
    "vllm_fp16":   "#4C9BE8",
    "vllm_awq":    "#9B7FE8",
    "vllm_pruned": "#E87C4C",
}

LABELS = {
    "hf_eager":    "HF eager",
    "hf_sdpa":     "HF SDPA",
    "vllm_fp16":   "vLLM fp16 ★",
    "vllm_awq":    "vLLM AWQ",
    "vllm_pruned": "vLLM Pruned*",
}


def get_n1_metrics(r: dict) -> dict:
    if "concurrency_sweep" in r:
        return r.get("n1_metrics", {})
    return r


def plot_n1_throughput(results: dict, ax: plt.Axes): #type: ignore
    """Bar chart: throughput di N=1 — engine effect isolation."""
    methods = list(results.keys())
    vals, colors, labels = [], [], []

    for m in methods:
        metrics = get_n1_metrics(results[m])
        tp = metrics.get("throughput_tok_per_sec")
        if tp is None:
            continue
        vals.append(float(tp))
        colors.append(COLORS[m])
        labels.append(LABELS[m])

    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("Throughput @ N=1 (engine isolation)", fontweight="bold", pad=12)
    ax.set_ylabel("tok/sec")
    ax.set_ylim(0, max(vals) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)


def plot_itl_comparison(results: dict, ax: plt.Axes): #type: ignore
    """ITL p50 bar chart — decode speed per token."""
    methods = list(results.keys())
    vals, colors, labels = [], [], []

    for m in methods:
        metrics = get_n1_metrics(results[m])
        itl = metrics.get("itl_p50_ms")
        if itl is None:
            continue
        vals.append(float(itl))
        colors.append(COLORS[m])
        labels.append(LABELS[m])

    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("ITL p50 — decode latency (↓ better)", fontweight="bold", pad=12)
    ax.set_ylabel("ms / token")
    ax.set_ylim(0, max(vals) * 1.25)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)


def plot_concurrency_scaling(results: dict, ax: plt.Axes): #type: ignore
    """Line plot: throughput scaling across concurrency levels — vLLM only."""
    vllm_methods = [m for m in results if "vllm" in m and "concurrency_sweep" in results[m]]

    for m in vllm_methods:
        sweep = results[m]["concurrency_sweep"]
        ns = [s["concurrency"] for s in sweep]
        tps = [s["throughput_tok_per_sec"] for s in sweep]
        ax.plot(ns, tps, marker="o", linewidth=2.5, markersize=8,
                color=COLORS[m], label=LABELS[m])

    ax.set_title("Throughput Scaling — Continuous Batching", fontweight="bold", pad=12)
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel("Throughput (tok/sec)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)


def plot_quality_vs_speed_tradeoff(ax: plt.Axes): #type: ignore
    """
    Scatter: throughput @ N=16 vs quality (perplexity dari stage 2).
    Manual data dari stage 1/2 karena tidak ada di stage3 results.
    """
    data = {
        "vLLM fp16":   {"throughput": 1127.77, "perplexity": 3.447, "color": COLORS["vllm_fp16"]},
        "vLLM AWQ":    {"throughput": 381.78,  "perplexity": 6.637, "color": COLORS["vllm_awq"]},
        "vLLM Pruned": {"throughput": 1412.22, "perplexity": 1299.76, "color": COLORS["vllm_pruned"]},
    }

    for name, d in data.items():
        ax.scatter(d["throughput"], d["perplexity"], color=d["color"], s=220,
                   zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(name, (d["throughput"], d["perplexity"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_title("Throughput vs Quality @ N=16\n(pruned is fast but unusable)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Throughput @ N=16 (tok/sec)")
    ax.set_ylabel("Perplexity (log scale, ↓ better)")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("← fast but broken", xy=(0.55, 0.85), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic")


def main():
    results = load_results()
    if not results:
        print("No results found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Stage 3 — Serving Engine Comparison\n"
        "Llama-3.2-1B variants · T4 GPU",
        fontsize=14, fontweight="bold", y=1.02
    )

    plot_n1_throughput(results, axes[0][0])
    plot_itl_comparison(results, axes[0][1])
    plot_concurrency_scaling(results, axes[1][0])
    plot_quality_vs_speed_tradeoff(axes[1][1])

    fig.text(
        0.5, -0.02,
        "* vLLM Pruned: highest throughput but model collapsed (perplexity 1299, see Stage 2). "
        "Speed without quality is not a valid serving choice.",
        ha="center", fontsize=9, color="gray", style="italic"
    )

    plt.tight_layout()
    out = Path("results/stage3/stage3_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()