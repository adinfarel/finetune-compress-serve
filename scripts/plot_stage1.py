import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def load_results() -> dict:
    methods = ["lora", "qlora", "full_ft"]
    results = {}
    
    for m in methods:
        path = Path(f"results/stage1/{m}/results.json")
        if path.exists():
            with open(path) as f:
                results[m] = json.load(f)
    
    return results

COLORS = {
    "lora":    "#4C9BE8",
    "qlora":   "#56C596",
    "full_ft": "#E87C4C",
}

LABELS = {
    "lora":    "LoRA",
    "qlora":   "QLoRA",
    "full_ft": "Full FT*",
}

def plot_quality_comparison(results: dict, ax: plt.Axes): #type: ignore
    """Perplexity bar chart — lower is better."""
    methods = [m for m in results if results[m].get("test_perplexity") not in (None, float("nan"))]
    values = [results[m]["test_perplexity"] for m in methods]
    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_title("Perplexity (↓ better)", fontweight="bold", pad=12)
    ax.set_ylabel("Perplexity")
    ax.set_ylim(0, max(values) * 1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)


def plot_vram_vs_quality(results: dict, ax: plt.Axes): #type: ignore
    """Scatter: VRAM vs Perplexity — ideal = bottom-left."""
    for method, r in results.items():
        ppl = r.get("test_perplexity")
        vram = r.get("peak_vram_gb")
        if ppl is None or ppl != ppl:  # nan check
            continue
        ax.scatter(vram, ppl, color=COLORS[method], s=200, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(LABELS[method], (vram, ppl),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_title("VRAM vs Quality tradeoff", fontweight="bold", pad=12)
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Perplexity (↓ better)")
    ax.spines[["top", "right"]].set_visible(False)

    ax.annotate("← ideal", xy=(0.05, 0.05), xycoords="axes fraction",
                fontsize=9, color="gray", style="italic")

def plot_win_rate(results: dict, ax: plt.Axes): #type: ignore
    """Stacked bar: win/tie/loss rates."""
    methods = list(results.keys())
    wins   = [results[m].get("judge_win_rate", 0) * 100 for m in methods]
    ties   = [results[m].get("judge_tie_rate", 0) * 100 for m in methods]
    losses = [results[m].get("judge_loss_rate", 0) * 100 for m in methods]
    labels = [LABELS[m] for m in methods]
    x = np.arange(len(methods))

    ax.bar(x, wins,   color="#56C596", label="Win",  width=0.5)
    ax.bar(x, ties,   color="#A8DADC", label="Tie",  width=0.5, bottom=wins)
    ax.bar(x, losses, color="#E87C4C", label="Loss", width=0.5,
           bottom=[w + t for w, t in zip(wins, ties)])

    ax.set_title("LLM-as-Judge Win Rate", fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    for i, w in enumerate(wins):
        ax.text(i, w/2, f"{w:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

def plot_efficiency(results: dict, ax: plt.Axes): #type: ignore
    """Train time + trainable param % — dual axis."""
    methods = list(results.keys())
    labels  = [LABELS[m] for m in methods]
    times   = [results[m].get("train_time_seconds", 0) / 60 for m in methods]  # ke menit
    pct     = [results[m].get("trainable_param_pct", 0) for m in methods]
    x = np.arange(len(methods))

    ax2 = ax.twinx()

    bars = ax.bar(x - 0.2, times, width=0.35, color=[COLORS[m] for m in methods],
                  alpha=0.85, label="Train Time (min)")
    ax2.bar(x + 0.2, pct, width=0.35, color=[COLORS[m] for m in methods],
            alpha=0.4, label="Trainable Params (%)")

    ax.set_title("Training Efficiency", fontweight="bold", pad=12)
    ax.set_ylabel("Train Time (min)")
    ax2.set_ylabel("Trainable Params (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    p1 = mpatches.Patch(color="gray", alpha=0.85, label="Train Time (min)")
    p2 = mpatches.Patch(color="gray", alpha=0.4,  label="Trainable Params (%)")
    ax.legend(handles=[p1, p2], fontsize=9, loc="upper left")

def main():
    results = load_results()

    if not results:
        print("ERROR: There's result not found at results/stage1/")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Stage 1 — Fine-Tuning Method Comparison\nLlama-3.2-1B · Alpaca · T4 GPU",
        fontsize=14, fontweight="bold", y=1.02
    )

    plot_quality_comparison(results, axes[0][0])
    plot_vram_vs_quality(results, axes[0][1])
    plot_win_rate(results, axes[1][0])
    plot_efficiency(results, axes[1][1])

    # footnote full FT disclaimer
    fig.text(0.5, -0.02,
             "* Full FT: training numerically unstable (float16 gradient overflow on T4). "
             "Results excluded from quality plots.",
             ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    out = Path("results/stage1/stage1_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()