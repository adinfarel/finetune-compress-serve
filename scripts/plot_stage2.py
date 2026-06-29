import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path



def load_results() -> dict:
    methods = {
        "int8":        "results/stage2/quant_int8/results.json",
        "int4_bnb":    "results/stage2/quant_int4/results.json",
        "awq":         "results/stage2/quant_awq/results.json",
        "prune_depth": "results/stage2/prune_depth/results.json",
    }
    results = {}
    for name, path in methods.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                results[name] = json.load(f)
        else:
            print(f"WARNING: {path} not found, skipping")
    return results


COLORS = {
    "int8":        "#4C9BE8",
    "int4_bnb":    "#56C596",
    "awq":         "#9B7FE8",
    "prune_depth": "#E87C4C",
}

LABELS = {
    "int8":        "BnB int8",
    "int4_bnb":    "BnB int4 ★",
    "awq":         "AWQ int4",
    "prune_depth": "Depth Prune*",
}

VIABLE = ["int8", "int4_bnb", "awq"]


def plot_perplexity(results: dict, ax: plt.Axes): #type: ignore
    methods = [m for m in VIABLE if m in results]
    vals = [results[m]["perplexity"] for m in methods]
    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    # stage 1 reference line
    ax.axhline(y=3.447, color="#E8C84C", linewidth=1.5, linestyle="--", label="Stage 1 QLoRA (3.447)")
    ax.legend(fontsize=8)

    ax.set_title("Perplexity (↓ better)", fontweight="bold", pad=12)
    ax.set_ylabel("Perplexity")
    ax.set_ylim(0, max(vals) * 1.25)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)


def plot_latency(results: dict, ax: plt.Axes): #type: ignore
    """TTFT vs ITL scatter — ideal = bottom left."""
    for m in results:
        r = results[m]
        ttft = r.get("ttft_p50_ms")
        itl = r.get("itl_p50_ms")
        if ttft is None or itl is None:
            continue
        ax.scatter(ttft, itl, color=COLORS[m], s=200, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(LABELS[m], (ttft, itl),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_title("TTFT vs ITL tradeoff (↓ better both)", fontweight="bold", pad=12)
    ax.set_xlabel("TTFT p50 (ms)")
    ax.set_ylabel("ITL p50 (ms)")
    ax.annotate("← ideal", xy=(0.05, 0.05), xycoords="axes fraction",
                fontsize=9, color="gray", style="italic")
    ax.spines[["top", "right"]].set_visible(False)


def plot_win_rate(results: dict, ax: plt.Axes): #type: ignore
    methods = list(results.keys())
    labels = [LABELS[m] for m in methods]
    wins   = [results[m].get("win_rate_vs_stage1", 0) * 100 for m in methods]
    ties   = [results[m].get("tie_rate_vs_stage1", 0) * 100 for m in methods]
    losses = [results[m].get("loss_rate_vs_stage1", 0) * 100 for m in methods]
    x = np.arange(len(methods))

    ax.bar(x, wins,   color="#56C596", label="Win",  width=0.5)
    ax.bar(x, ties,   color="#A8DADC", label="Tie",  width=0.5, bottom=wins)
    ax.bar(x, losses, color="#E87C4C", label="Loss", width=0.5,
           bottom=[w+t for w, t in zip(wins, ties)])

    ax.set_title("Win Rate vs Stage 1 QLoRA", fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    for i, w in enumerate(wins):
        if w > 0:
            ax.text(i, w/2, f"{w:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")


def plot_size_vs_quality(results: dict, ax: plt.Axes): #type: ignore
    """Size on disk vs perplexity — ideal = bottom left."""
    ax.scatter(
        [2400], [3.447],
        color="#E8C84C", s=150, zorder=5,
        edgecolors="white", linewidth=1.5, marker="*"
    )
    ax.annotate("Stage 1 QLoRA\n(full fp16)", (2400, 3.447),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color="#E8C84C")

    for m in VIABLE:
        if m not in results:
            continue
        r = results[m]
        size = r.get("size_mb", 0)
        ppl = r.get("perplexity")
        if not ppl:
            continue

        if size == 0:
            size = 1300 if m == "int8" else 700  

        ax.scatter(size, ppl, color=COLORS[m], s=200, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(LABELS[m], (size, ppl),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_title("Model Size vs Quality", fontweight="bold", pad=12)
    ax.set_xlabel("Size on disk (MB, estimated for BnB)")
    ax.set_ylabel("Perplexity (↓ better)")
    ax.annotate("← ideal", xy=(0.05, 0.05), xycoords="axes fraction",
                fontsize=9, color="gray", style="italic")
    ax.spines[["top", "right"]].set_visible(False)

def main():
    results = load_results()
    if not results:
        print("No results found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Stage 2 — Compression Method Comparison\n"
        "Llama-3.2-1B QLoRA merged · T4 GPU",
        fontsize=14, fontweight="bold", y=1.02
    )

    plot_perplexity(results, axes[0][0])
    plot_latency(results, axes[0][1])
    plot_win_rate(results, axes[1][0])
    plot_size_vs_quality(results, axes[1][1])

    fig.text(
        0.5, -0.02,
        "* Depth Pruning: model collapse (perplexity 1299) — excluded from quality plots. "
        "Distillation: skipped (NaN gradients + training time not viable on T4).",
        ha="center", fontsize=9, color="gray", style="italic"
    )

    plt.tight_layout()
    out = Path("results/stage2/stage2_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()