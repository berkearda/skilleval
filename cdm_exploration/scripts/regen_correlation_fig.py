"""Regenerate presentation-friendly skill correlation histogram with built-in takeaway cards."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "report"

mastery_df = pd.read_csv(DATA_DIR / "skill_mastery_profiles_hac50.csv", index_col=0)
mastery_matrix = mastery_df.values
corr_matrix = np.corrcoef(mastery_matrix.T)
upper_tri = corr_matrix[np.triu_indices(50, k=1)]

# ── Layout: histogram on top, three cards on bottom ──
fig = plt.figure(figsize=(10, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)

ax = fig.add_subplot(gs[0])

# Histogram
bins = np.linspace(-0.2, 1.0, 50)
n, bin_edges, patches = ax.hist(upper_tri, bins=bins, edgecolor="white",
                                 linewidth=0.5, color="#3B6CB5", alpha=0.85)

for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge < 0:
        patch.set_facecolor("#B71C1C")
        patch.set_alpha(0.7)
    elif left_edge >= 0.7:
        patch.set_facecolor("#E8A817")
        patch.set_alpha(0.85)

# Mean line
mean_val = np.mean(upper_tri)
ax.axvline(mean_val, color="#1F407A", linewidth=2.2, linestyle="--", zorder=5)
ax.annotate(f"mean = {mean_val:.2f}",
            xy=(mean_val, ax.get_ylim()[1] * 0.92),
            xytext=(mean_val + 0.13, ax.get_ylim()[1] * 0.92),
            fontsize=12, fontweight="bold", color="#1F407A",
            arrowprops=dict(arrowstyle="->,head_width=0.2", color="#1F407A", lw=1.5),
            va="center")

ax.annotate("near-duplicate\nskill pairs",
            xy=(0.88, 5), fontsize=8.5, color="#8B6914", ha="center", style="italic")
ax.annotate("anti-correlated\npairs",
            xy=(-0.08, 8), fontsize=8.5, color="#B71C1C", ha="center", style="italic")

ax.set_xlabel("Pearson correlation coefficient (r)", fontsize=11, labelpad=6)
ax.set_ylabel("Number of skill pairs", fontsize=11, labelpad=6)
ax.set_xlim(-0.25, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)
ax.text(0.98, 0.95, "1,225 pairwise correlations\n(50 skills, 235 LLMs)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray", style="italic")

# ── Bottom strip: three takeaway cards ──
ax_cards = fig.add_subplot(gs[1])
ax_cards.set_xlim(0, 3)
ax_cards.set_ylim(0, 1)
ax_cards.axis("off")

card_data = [
    {"x": 0.5, "bg": "#D6E3F5", "border": "#3B6CB5", "title": "Moderate positive",
     "value": r"$\bar{r}$ = 0.32", "sub": "Skills are correlated", "vcolor": "#1F407A"},
    {"x": 1.5, "bg": "#FFF3D0", "border": "#E8A817", "title": "Near-duplicate pairs",
     "value": "r > 0.9",  "sub": "Redundant clusters", "vcolor": "#8B6914"},
    {"x": 2.5, "bg": "#FADEDE", "border": "#B71C1C", "title": "Anti-correlated pairs",
     "value": "r < 0",    "sub": "Genuine trade-offs", "vcolor": "#B71C1C"},
]

for c in card_data:
    rect = mpatches.FancyBboxPatch(
        (c["x"] - 0.44, 0.08), 0.88, 0.84,
        boxstyle="round,pad=0.04",
        facecolor=c["bg"], edgecolor=c["border"], linewidth=1.5)
    ax_cards.add_patch(rect)
    ax_cards.text(c["x"], 0.78, c["title"], ha="center", va="center",
                  fontsize=9, color="#444444")
    ax_cards.text(c["x"], 0.46, c["value"], ha="center", va="center",
                  fontsize=15, fontweight="bold", color=c["vcolor"])
    ax_cards.text(c["x"], 0.18, c["sub"], ha="center", va="center",
                  fontsize=8.5, color="#888888", style="italic")

path_out = FIG_DIR / "fig_skill_correlation_hist.pdf"
fig.savefig(path_out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {path_out}")
