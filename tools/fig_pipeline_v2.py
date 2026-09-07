"""Render the finalized SkillEval taxonomy pipeline as a professional PNG.

Color-codes every operation by the kind of call it makes:
  LLM call / embedding+vector search / plain compute / human check.

    python tools/fig_pipeline_v2.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Okabe-Ito palette, one colour per call type
LLM = "#D55E00"   # vermillion  : LLM call
VEC = "#0072B2"   # blue        : embedding / vector search
CPU = "#009E73"   # green       : plain compute, no model
HUM = "#CC79A7"   # purple      : human check
INK = "#1a1a1a"
MUTE = "#6b6b6b"
BANDBG = "#f5f5f2"
BANDED = "#3a3a3a"

TAGWORD = {"LLM": "LLM", "VEC": "SEARCH", "CPU": "COMPUTE", "HUM": "HUMAN"}
TAGCOL = {"LLM": LLM, "VEC": VEC, "CPU": CPU, "HUM": HUM}
# (kind, label, column_x, row_y_offset) laid out as a 2x2 grid
LEGEND = [("LLM", "LLM call", 12, 0), ("VEC", "candidate net (recall)", 60, 0),
          ("CPU", "plain compute", 12, 1), ("HUM", "human check", 60, 1)]

W, H = 120.0, 220.0
fig, ax = plt.subplots(figsize=(12.5, 23.0))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

LEFT, RIGHT = 8.0, 112.0
CW = RIGHT - LEFT
CHIPW, CHIPH = 17.0, 4.4
DESC_X = LEFT + 8 + CHIPW + 2  # where substep text starts


def rbox(x, y, w, h, fc, ec, lw=1.4, z=1, rs=2.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                       mutation_aspect=1, fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def chip(cx, cy, kind):
    rbox(cx, cy - CHIPH / 2, CHIPW, CHIPH, TAGCOL[kind], TAGCOL[kind], lw=0, z=3, rs=1.4)
    ax.text(cx + CHIPW / 2, cy, TAGWORD[kind], color="white", fontsize=7.4,
            fontweight="bold", ha="center", va="center", zorder=4)


def varrow(x, y0, y1, label=None):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=18, lw=2.2, color="#333333", zorder=2))
    if label:
        ax.text(x + 3, (y0 + y1) / 2, label, fontsize=8.6, style="italic",
                color=MUTE, ha="left", va="center")


def band(y_top, num, title, subs):
    """Draw one numbered step band; return its bottom y."""
    h = 9.0 + len(subs) * 5.2 + 2.5
    y_bot = y_top - h
    rbox(LEFT, y_bot, CW, h, BANDBG, BANDED, lw=1.6, z=1)
    # number badge + title
    bcx, bcy = LEFT + 5.5, y_top - 5.5
    ax.add_patch(Circle((bcx, bcy), 3.3, fc=INK, ec="none", zorder=3))
    ax.text(bcx, bcy, str(num), color="white", fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=4)
    ax.text(LEFT + 11, y_top - 5.5, title, fontsize=12.5, fontweight="bold",
            color=INK, ha="left", va="center", zorder=3)
    # substeps
    sy = y_top - 12.5
    for kind, desc in subs:
        chip(LEFT + 8, sy, kind)
        ax.text(DESC_X, sy, desc, fontsize=9.6, color=INK, ha="left",
                va="center", zorder=3)
        sy -= 5.2
    return y_bot


# ---- title + legend ----
ax.text(W / 2, H - 5, "SkillEval: skill-taxonomy construction pipeline",
        fontsize=17, fontweight="bold", ha="center", va="center", color=INK)
ax.text(W / 2, H - 9.8,
        "general, data-driven by default; inputs are only the questions (everything else optional)",
        fontsize=10, style="italic", ha="center", va="center", color=MUTE)
ax.text(W / 2, H - 13.0,
        "embeddings only narrow candidates; an LLM makes every reuse and same/different decision",
        fontsize=10, style="italic", ha="center", va="center", color="#7a3b16")
lrow = [H - 17.0, H - 21.0]
for kind, lab, lx, lr in LEGEND:
    chip(lx, lrow[lr], kind)
    ax.text(lx + CHIPW + 1.5, lrow[lr], lab, fontsize=8.8, color=INK,
            ha="left", va="center")

# ---- inputs ----
iy_top = H - 25.5
ih = 11
rbox(LEFT, iy_top - ih, CW, ih, "#eef3f7", "#7f9bb0", lw=1.4, z=1)
ax.text(LEFT + 4, iy_top - 4.2, "INPUTS", fontsize=10, fontweight="bold",
        color="#33597a", ha="left", va="center")
ax.text(LEFT + 4, iy_top - 8.2,
        "Questions (required)        Answers, seed taxonomy, worked solutions (all optional)",
        fontsize=9.6, color=INK, ha="left", va="center")
cursor = iy_top - ih

GAP = 7.0
varrow(W / 2, cursor, cursor - GAP)
cursor -= GAP

cursor = band(cursor, 1, "Solve and label, step by step", [
    ("LLM", "Solve each item; label each step with a fine skill + one broad skill"),
    ("CPU", "Use real solutions when available (optional grounding for hard items)"),
    ("CPU", "Gold-answer gate: correctly solved items only build the bank"),
])
varrow(W / 2, cursor, cursor - GAP, "per-question skills + definitions")
cursor -= GAP

cursor = band(cursor, 2, "Build the skill bank as a searchable index  (running memory)", [
    ("VEC", "Coarse net: modern embedding + keyword search -> top 20-30 candidates"),
    ("CPU", "Exact / near-exact match: auto-reuse.    Empty net: auto-create."),
    ("LLM", "LLM reads the candidates and decides reuse vs create, not a threshold"),
    ("CPU", "Run in parallel chunks, then combine the local banks"),
])
varrow(W / 2, cursor, cursor - GAP, "skill bank")
cursor -= GAP

cursor = band(cursor, 3, "Merge duplicates, then build the two-level hierarchy", [
    ("VEC", "Coarse net nominates look-alike pairs (embedding + keyword)"),
    ("LLM", "LLM reads both definitions + example items, decides same or not"),
    ("CPU", "Meaning wins; a behavior difference is reported, never used to split"),
    ("LLM", "Nest each fine skill under exactly one broad parent skill"),
])
varrow(W / 2, cursor, cursor - GAP, "two-level taxonomy (fine nested under broad)")
cursor -= GAP

cursor = band(cursor, 4, "Label every question against the final bank", [
    ("VEC", "Coarse net retrieves candidate skills per item (constant-size prompt)"),
    ("LLM", "LLM assigns skills from the bank; parallel; order-independent"),
    ("LLM", "Every item labeled; failed-extraction ones via a stronger model + flag"),
])
varrow(W / 2, cursor, cursor - GAP, "Q-matrix (items x skills)")
cursor -= GAP

cursor = band(cursor, 5, "Evaluate, two separate parts", [
    ("LLM", "Part A, good skills (by reading): real skill, right items, size, coverage, recurs"),
    ("HUM", "Part A anchor: a human checks a sample, so the model does not only grade itself"),
    ("CPU", "Part B, diagnostic value (by scores): held-out split, CDM fit, reported separately"),
])

# ---- checks footer ----
fy_top = cursor - GAP
varrow(W / 2, cursor, fy_top)
fh = 18
rbox(LEFT, fy_top - fh, CW, fh, "#f0efe9", "#9a957f", lw=1.3, z=1)
ax.text(LEFT + 4, fy_top - 4, "BUILT-IN CHECKS (compute)", fontsize=9.5,
        fontweight="bold", color="#6b6442", ha="left", va="center")
ax.text(LEFT + 4, fy_top - 8.0,
        "beat free category labels   .   beat old cluster version   .   duplicate reduction",
        fontsize=8.8, color=INK, ha="left", va="center")
ax.text(LEFT + 4, fy_top - 11.2,
        "bank convergence (growth curve)   .   order robustness   .   cutoff insensitivity",
        fontsize=8.8, color=INK, ha="left", va="center")
ax.text(LEFT + 4, fy_top - 14.4,
        "embedder chosen by recall@k   .   rare-skill merge/drop   .   cost estimate   .   generality (MMLU)",
        fontsize=8.8, color=INK, ha="left", va="center")

out = "skilleval_pipeline_v2.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
