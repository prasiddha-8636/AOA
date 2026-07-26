"""
Turns results/colab_benchmark_results.json into the comparative plots main.tex
claims exist. Run standalone after run_colab_benchmark.py, or call
plot_all(results) directly from within it.
"""

import os
import json
import matplotlib.pyplot as plt


def plot_perplexity_curves(all_results: dict, out_path: str):
    plt.figure(figsize=(7, 5))
    for key, res in all_results.items():
        lengths, ppls = [], []
        for L, v in sorted(res["perplexity"].items(), key=lambda kv: int(kv[0])):
            if isinstance(v, (int, float)):
                lengths.append(int(L))
                ppls.append(v)
        if lengths:
            plt.plot(lengths, ppls, marker="o", label=res["model_name"])
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Perplexity")
    plt.xscale("log", base=2)
    plt.title("Perplexity vs. Sequence Length")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_needle_heatmap(all_results: dict, out_dir: str):
    for key, res in all_results.items():
        lengths = sorted(res["needle_accuracy"].keys(), key=lambda x: int(x))
        depths = None
        matrix = []
        for L in lengths:
            row = res["needle_accuracy"][L]
            if depths is None:
                depths = sorted(row.keys(), key=lambda x: float(x))
            matrix.append([row[d] if row[d] is not None else float("nan") for d in depths])

        if not matrix:
            continue

        plt.figure(figsize=(6, 4))
        plt.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(label="Retrieval accuracy")
        plt.xticks(range(len(depths)), [f"{float(d)*100:.0f}%" for d in depths])
        plt.yticks(range(len(lengths)), [str(L) for L in lengths])
        plt.xlabel("Needle depth")
        plt.ylabel("Sequence length")
        plt.title(f"Needle-in-a-Haystack: {res['model_name']}")
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"needle_{key}.png"), dpi=150)
        plt.close()


def plot_overhead(all_results: dict, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, res in all_results.items():
        lengths, vram, latency = [], [], []
        for L, v in sorted(res["overhead"].items(), key=lambda kv: int(kv[0])):
            if v.get("vram_mb", 0) > 0:
                lengths.append(int(L))
                vram.append(v["vram_mb"])
                latency.append(v["latency_ms"])
        if lengths:
            ax1.plot(lengths, vram, marker="o", label=res["model_name"])
            ax2.plot(lengths, latency, marker="o", label=res["model_name"])

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Peak VRAM (MB)")
    ax1.set_title("Memory Overhead")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Latency (ms/token)")
    ax2.set_title("Inference Latency")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_all(all_results: dict, out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    plot_perplexity_curves(all_results, os.path.join(out_dir, "perplexity_vs_length.png"))
    plot_needle_heatmap(all_results, out_dir)
    plot_overhead(all_results, os.path.join(out_dir, "overhead.png"))
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    with open("results/colab_benchmark_results.json") as f:
        all_results = json.load(f)
    plot_all(all_results)
