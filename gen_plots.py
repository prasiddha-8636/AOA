import json, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/admin/AOAPaper/results/colab_benchmark_results.json") as f:
    results = json.load(f)

os.makedirs("/home/admin/AOAPaper/results/plots", exist_ok=True)

# PPL curves
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"learned_absolute": "#e74c3c", "rope": "#3498db", "alibi": "#2ecc71"}
for mk, res in results.items():
    xs, ys = [], []
    for L, v in sorted(res["perplexity"].items(), key=lambda kv: int(kv[0])):
        if isinstance(v, (int, float)):
            xs.append(int(L))
            ys.append(v)
    if xs:
        ax.plot(
            xs,
            ys,
            marker="o",
            label=res["model_name"],
            color=colors.get(mk),
            linewidth=2,
        )
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Perplexity")
ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
ax.set_title("Zero-Shot Perplexity vs. Sequence Length")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/admin/AOAPaper/results/plots/ppl_curves.pdf", dpi=150)
plt.savefig("/home/admin/AOAPaper/results/plots/ppl_curves.png", dpi=150)
plt.close()
print("Saved ppl_curves")

# Needle heatmap
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (mk, res) in enumerate(results.items()):
    lengths = sorted(res["needle_accuracy"].keys(), key=lambda x: int(x))
    depths = ["0.25", "0.5", "0.75", "0.9"]
    matrix = []
    for L in lengths:
        row = res["needle_accuracy"][L]
        if row is None:
            matrix.append([float("nan")] * len(depths))
        else:
            matrix.append(
                [row[d] if row[d] is not None else float("nan") for d in depths]
            )
    ax = axes[i]
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(["25%", "50%", "75%", "90%"])
    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels([str(int(L)) for L in lengths])
    ax.set_xlabel("Insertion Depth")
    ax.set_ylabel("Sequence Length")
    ax.set_title(res["model_name"].split("(")[0].strip(), fontsize=10)
plt.colorbar(im, ax=axes, label="Retrieval Accuracy", fraction=0.02, pad=0.04)
plt.tight_layout()
plt.savefig("/home/admin/AOAPaper/results/plots/needle_heatmaps.pdf", dpi=150)
plt.savefig("/home/admin/AOAPaper/results/plots/needle_heatmaps.png", dpi=150)
plt.close()
print("Saved needle_heatmaps")

# Latency comparison
fig, ax = plt.subplots(figsize=(7, 4))
for mk, res in results.items():
    xs, ys = [], []
    for L, v in sorted(res["overhead"].items(), key=lambda kv: int(kv[0])):
        if isinstance(v, dict) and v.get("latency_ms", 0) > 0:
            xs.append(int(L))
            ys.append(v["latency_ms"])
    if xs:
        ax.plot(
            xs,
            ys,
            marker="s",
            label=res["model_name"],
            color=colors.get(mk),
            linewidth=2,
        )
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Latency (ms/token)")
ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
ax.set_title("Inference Latency vs. Sequence Length")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/admin/AOAPaper/results/plots/latency_curves.pdf", dpi=150)
plt.savefig("/home/admin/AOAPaper/results/plots/latency_curves.png", dpi=150)
plt.close()
print("Saved latency_curves")
print("All plots saved to results/plots/")
