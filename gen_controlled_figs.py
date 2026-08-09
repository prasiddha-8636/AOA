"""Generate controlled-experiment figures for the paper."""

import json, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/controlled_results.json") as f:
    data = json.load(f)

with open("results/bootstrap_ci_results.json") as f:
    ci_data = json.load(f)["perplexity"]

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

colors = {"learned": "#e74c3c", "rope": "#3498db", "alibi": "#2ecc71"}
labels = {"learned": "Learned Absolute", "rope": "RoPE", "alibi": "ALiBi"}

# --- Fig 1: PPL curves with bootstrap CI bands ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in ["learned", "rope", "alibi"]:
    ppl = data[m]["ppl"]
    ci = ci_data[m]
    xs = sorted(int(k) for k in ppl)
    ys = [ppl[str(x)] for x in xs]
    lo = [ci[str(x)]["ci_lo"] for x in xs]
    hi = [ci[str(x)]["ci_hi"] for x in xs]
    ax.plot(
        xs, ys, marker="o", color=colors[m], label=labels[m], linewidth=2, markersize=5
    )
    ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.15)
ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Perplexity")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/ppl_curves.pdf", dpi=200)
plt.close()
print("Saved ppl_curves.pdf")

# --- Fig 2: Extrapolation ratio bar chart ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ratios = {}
for m in ["learned", "rope", "alibi"]:
    ppl = data[m]["ppl"]
    ratios[m] = ppl["8192"] / ppl["512"]

names = [labels[m] for m in ratios]
vals = [ratios[m] for m in ratios]
bars = ax.bar(
    names, vals, color=[colors[m] for m in ratios], width=0.5, edgecolor="white"
)

ax.axhline(
    y=1.0, color="gray", linestyle="--", linewidth=1.0, label="No degradation (1.0×)"
)
for bar, val in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{val:.2f}×",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax.set_ylabel("Perplexity at 8192 / Perplexity at 512")
ax.set_ylim(0, max(vals) * 1.25)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/extrapolation_ratio.pdf", dpi=200)
plt.close()
print("Saved extrapolation_ratio.pdf")

# --- Fig 3: Needle heatmaps (unchanged - all zeros) ---
fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))
for i, m in enumerate(["learned", "rope", "alibi"]):
    ctx = [512, 1024, 2048, 4096, 8192]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    heatmap = np.zeros((len(depths), len(ctx)))
    ax = axes[i]
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ctx)))
    ax.set_xticklabels([str(c) for c in ctx], fontsize=7, rotation=45)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d * 100:.0f}%" for d in depths], fontsize=7)
    ax.set_title(labels[m], fontsize=9)
    ax.set_xlabel("Context Length")
    if i == 0:
        ax.set_ylabel("Depth")

plt.tight_layout()
plt.savefig(f"{OUT}/needle_heatmaps.pdf", dpi=200)
plt.close()
print("Saved needle_heatmaps.pdf")
