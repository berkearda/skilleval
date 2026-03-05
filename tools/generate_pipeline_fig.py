#!/usr/bin/env python
"""Generate the CDMEval pipeline diagram (assets/pipeline.{png,pdf}).

Standalone script — no Hydra, no data dependencies.
Run:  python tools/generate_pipeline_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cdmeval.utils.visualization import setup_style, SAVE_KW

# ── Colours ───────────────────────────────────────────────────────
PRIMARY = "#4C72B0"
PROCESS_BG = "#D6E3F5"
IO_BG = "#F0F0F0"
EDGE_PROCESS = "#4C72B0"
EDGE_IO = "#AAAAAA"

# ── Layout ────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 5.8
BOX_W, BOX_H = 2.3, 0.85
CONTENT_L = 2.15
CONTENT_R = FIG_W - 0.15
BANNER_X, BANNER_W = 0.1, 1.7
LAYER_YS = [4.6, 2.9, 1.2]
BAND_H = 1.3

# ── Layer specs ───────────────────────────────────────────────────
LAYERS = [
    {
        "label": "Layer 1\nQ-Matrix\nConstruction",
        "boxes": [
            ("2,643 Math Problems\n(GSM8K + MATH)", "input"),
            ("LLM Metacognition\n\u2192 7,723 Skill Labels", "process"),
            ("SBERT\nEmbedding", "process"),
            ("HAC Clustering\n(K = 50)", "process"),
            ("Binary Q-Matrix\n(2,643 \u00d7 50)", "output"),
        ],
    },
    {
        "label": "Layer 2\nSkill\nProfiling",
        "boxes": [
            ("Response Matrix\n(235 \u00d7 2,643)\n+ Q-Matrix", "input"),
            ("NCDM\nTraining", "process"),
            ("Skill Mastery Profiles\n(235 \u00d7 50)", "output"),
        ],
    },
    {
        "label": "Layer 3\nRouting",
        "boxes": [
            ("New Question", "input"),
            ("SBERT\nEmbedding", "process"),
            ("Text-Conditioned\nNCDM", "process"),
            ("P(correct)\nper LLM", "process"),
            ("Route to\nBest LLM", "output"),
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────

def _box_centres(n: int) -> list[float]:
    """Return *n* evenly-spaced x-centres within the content strip."""
    first = CONTENT_L + BOX_W / 2
    last = CONTENT_R - BOX_W / 2
    if n == 1:
        return [(first + last) / 2]
    step = (last - first) / (n - 1)
    return [first + i * step for i in range(n)]


def _draw_box(ax, cx: float, cy: float, text: str, kind: str) -> None:
    fc = PROCESS_BG if kind == "process" else IO_BG
    ec = EDGE_PROCESS if kind == "process" else EDGE_IO
    patch = mpatches.FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor=ec, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=8.5, linespacing=1.3)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, **kw) -> None:
    patch = mpatches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=0.8,head_width=0.4",
        mutation_scale=15, color="black", linewidth=1.3, **kw,
    )
    ax.add_patch(patch)


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    for layer, yc in zip(LAYERS, LAYER_YS):
        # Layer banner
        banner = mpatches.FancyBboxPatch(
            (BANNER_X, yc - BAND_H / 2), BANNER_W, BAND_H,
            boxstyle="round,pad=0.06",
            facecolor=PRIMARY, edgecolor="none",
        )
        ax.add_patch(banner)
        ax.text(
            BANNER_X + BANNER_W / 2, yc, layer["label"],
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white", linespacing=1.35,
        )

        # Boxes
        n = len(layer["boxes"])
        xs = _box_centres(n)
        for (text, kind), cx in zip(layer["boxes"], xs):
            _draw_box(ax, cx, yc, text, kind)

        # Horizontal arrows between consecutive boxes
        for i in range(n - 1):
            _arrow(ax,
                   xs[i] + BOX_W / 2 + 0.05, yc,
                   xs[i + 1] - BOX_W / 2 - 0.05, yc)

    # Downward arrows between layer banners
    bx = BANNER_X + BANNER_W / 2
    for i in range(len(LAYER_YS) - 1):
        _arrow(ax,
               bx, LAYER_YS[i] - BAND_H / 2 - 0.02,
               bx, LAYER_YS[i + 1] + BAND_H / 2 + 0.02)

    # Save
    out_dir = ROOT / "assets"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"pipeline.{ext}"
        fig.savefig(path, **SAVE_KW)
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
