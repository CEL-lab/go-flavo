"""
Regenerate the F1-score difference heatmap for the manuscript.

Shows "Difference from Best" (in percentage points) for each of the 5
taxon-specific models across 4 F1 averaging strategies.

Data source: current_results.md / experiment JSON files.
Output: v2/paper/images/Heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ── Current v2 results (from current_results.md) ──────────────────────────
# Using the corrected 70/15/15 split results for all models.

models = ["XGBoost", "DNN", "GAT", "GCN", "GraphSAGE"]

# Micro F1, Macro F1, Weighted F1, Samples F1
results = {
    "XGBoost":   [0.8952, 0.7999, 0.8883, 0.8505],
    "DNN":       [0.9298, 0.8887, 0.9304, 0.8926],
    "GAT":       [0.7756, 0.7836, 0.8384, 0.7911],
    "GCN":       [0.7740, 0.7860, 0.8357, 0.7851],
    "GraphSAGE": [0.8112, 0.8098, 0.8476, 0.8053],
}

metrics = [
    "Difference from Best\nMicro F1 (%)",
    "Difference from Best\nMacro F1 (%)",
    "Difference from Best\nWeighted F1 (%)",
    "Difference from Best\nSamples F1 (%)",
]

# Build matrix: difference from best (percentage points)
data = np.array([results[m] for m in models])          # (5, 4)
best_per_metric = data.max(axis=0)                       # (4,)
diff = (best_per_metric - data) * 100                    # percentage points

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    diff,
    annot=True,
    fmt=".1f",
    cmap="RdBu_r",          # same diverging palette as the original
    vmin=0,
    vmax=max(diff.max(), 20),  # scale to data range
    xticklabels=[m.replace("Difference from Best\n", "") for m in metrics],
    yticklabels=models,
    linewidths=1,
    linecolor="white",
    cbar_kws={"label": "Percentage points below best"},
    ax=ax,
)

ax.set_xlabel("Metric")
ax.set_ylabel("Model")
ax.set_xticklabels(
    [
        "Difference from Best\nMicro F1 (%)",
        "Difference from Best\nMacro F1 (%)",
        "Difference from Best\nWeighted F1 (%)",
        "Difference from Best\nSamples F1 (%)",
    ],
    rotation=45,
    ha="right",
)

plt.tight_layout()

# Save
out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "images")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Heatmap.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also print the data for verification
print("\nDifference from best (percentage points):")
print(f"{'Model':<12}", end="")
for m in ["Micro F1", "Macro F1", "Weighted F1", "Samples F1"]:
    print(f"{m:>14}", end="")
print()
for i, model in enumerate(models):
    print(f"{model:<12}", end="")
    for j in range(4):
        print(f"{diff[i, j]:>14.1f}", end="")
    print()

plt.close()
