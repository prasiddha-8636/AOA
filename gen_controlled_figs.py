"""Generate controlled-experiment figures for the paper."""

import json, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/controlled_results.json") as f:
    data = json.load(f)

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

palette = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#c0392b",
    "#16a085",
]
labels = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
    "alibi": "ALiBi",
    "kerple": "KERPLE",
    "cable": "CABLE",
    "nope": "NoPE",
    "position_interpolation": "PI",
    "yarn": "YaRN",
}
methods = [m for m in labels if m in data]
colors = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

# --- Fig 1: PPL curves ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in methods:
    ppl = data[m]["ppl"]
    xs = sorted(int(k) for k in ppl if isinstance(ppl[k], (int, float)))
    ys = [ppl[str(x)] for x in xs]
    ax.plot(
        xs, ys, marker="o", color=colors[m], label=labels[m], linewidth=2, markersize=5
    )
ax.set_xscale("log", base=2)
ax.set_xticks([512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
ax.set_xlabel("Context Length", fontsize=11)
ax.set_ylabel("Perplexity", fontsize=11)
ax.set_title("Perplexity vs Context Length", fontsize=12)
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/ppl_curves.pdf", bbox_inches="tight")
plt.close(fig)

# --- Fig 2: Extrapolation ratio ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ratios = {}
for m in methods:
    ppl = data[m]["ppl"]
    p512 = ppl.get("512", ppl.get(512))
    p8192 = ppl.get("8192", ppl.get(8192))
    if p512 and p8192:
        ratios[m] = p8192 / p512
# Sort by ratio
sorted_methods = sorted(ratios, key=ratios.get)
bar_colors = [colors[m] for m in sorted_methods]
bar_labels = [labels[m] for m in sorted_methods]
bar_values = [ratios[m] for m in sorted_methods]
bars = ax.barh(bar_labels, bar_values, color=bar_colors, edgecolor="white", height=0.6)
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="No change")
ax.set_xlabel("PPL$_{8192}$ / PPL$_{512}$", fontsize=11)
ax.set_title("Extrapolation Ratio", fontsize=12)
ax.grid(True, axis="x", alpha=0.3)
# Cap sinusoidal for readability
for bar, val in zip(bars, bar_values):
    if val > 100:
        ax.text(
            bar.get_width() - 10,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}$\\times$",
            ha="right",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    else:
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}$\\times$",
            ha="left",
            va="center",
            fontsize=9,
        )
plt.tight_layout()
fig.savefig(f"{OUT}/extrapolation_ratio.pdf", bbox_inches="tight")
plt.close(fig)

# --- Fig 3: Needle heatmap (0% everywhere, show one representative) ---
fig, ax = plt.subplots(figsize=(4, 3))
depths = ["0%", "25%", "50%", "75%", "100%"]
lengths = ["512", "1024", "2048", "4096", "8192"]
heatmap = np.zeros((len(depths), len(lengths)))
im = ax.imshow(heatmap, cmap="Reds", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(lengths)))
ax.set_xticklabels(lengths, fontsize=9)
ax.set_yticks(range(len(depths)))
ax.set_yticklabels(depths, fontsize=9)
ax.set_xlabel("Context Length", fontsize=10)
ax.set_ylabel("Needle Depth", fontsize=10)
ax.set_title("Needle Retrieval (0\\% all methods)", fontsize=10)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Accuracy", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}/needle_heatmaps.pdf", bbox_inches="tight")
plt.close(fig)

print("Figures saved to", OUT)
