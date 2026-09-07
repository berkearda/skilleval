"""Reliability diagram for fig:calib (gold-tier batch-2 revision).

Renders a 2-panel reliability diagram comparing model calibration on
training items (well-calibrated, ECE=0.028) vs held-out test items
(mildly overconfident, ECE=0.108). Style anchored to
NEURIPS_FIGURE_CHECKLIST.md Part J (LOCKED Tailwind-700 palette, no orange).

Differences vs batch 1 (`fig_calibration_reliability_revised.py`):
  - palette swapped to LOCKED Part J: blue-700 / pink-700 / slate-600
    (no orange / no vermillion anywhere)
  - no figure-level suptitle (finding lives in LaTeX caption only)
  - tighter margins (less wasted space without the suptitle)
  - bold reserved for panel letters only; ECE/Brier text boxes regular weight
  - over-confidence annotation uses regular-weight slate text + slate leader

Data provenance:
  - All ECE / Brier / N values are read from the experiment-log-verified
    JSON at cdm_exploration/experiments/v2_calibration_analysis.json
    (entry name=calibration_analysis, verified=True).
  - Per-bin reliability data (mean predicted P, observed fraction, count)
    is read from the cached sidecar at
    cdm_exploration/experiments/v2_calibration_reliability_bins.json
    (built once by tools/fig_calibration_reliability_revised.py from the
    same checkpoint + 80/20 split that tools/run_calibration_analysis.py
    uses; seed_everything(42), test_size=0.2, random_state=42).

Output:
  cdm_exploration/figures/review_batch2/fig_calibration_reliability_REVISED.pdf
  cdm_exploration/figures/review_batch2/fig_calibration_reliability_REVISED.png  (200 DPI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
OUT_DIR = REPO / "cdm_exploration" / "figures" / "review_batch2"
JSON_PATH = EXP / "v2_calibration_analysis.json"
BINS_CACHE = EXP / "v2_calibration_reliability_bins.json"

# LOCKED Part J palette (Tailwind-700, NO ORANGE)
COLOR_TRAIN = "#1D4ED8"   # blue-700  (in-distribution / training items)
COLOR_TEST  = "#BE185D"   # pink-700  (out-of-distribution / held-out items)
COLOR_REF   = "#475569"   # slate-600 (y=x reference + neutral annotation)
N_BINS = 20


def _require_bins_cache() -> dict:
    """Read the cached per-bin reliability data. Cache build is owned by
    the batch-1 script; this v2 renderer is read-only for speed."""
    if not BINS_CACHE.exists():
        raise FileNotFoundError(
            f"Missing bins cache at {BINS_CACHE}. "
            f"Run tools/fig_calibration_reliability_revised.py once to "
            f"build it (one-time, then this script can render in <5 s)."
        )
    return json.loads(BINS_CACHE.read_text())


def _draw_panel(ax_main, ax_hist, bins, ece, brier, n_items,
                title, color, panel_letter):
    """Reliability diagram on top axis + prediction histogram below."""
    mp = np.array([b["mean_pred"] for b in bins])
    ma = np.array([b["mean_actual"] for b in bins])
    cnt = np.array([b["count"] for b in bins], dtype=float)
    frac = cnt / cnt.sum() if cnt.sum() > 0 else cnt
    centers = np.array([(b["lo"] + b["hi"]) / 2 for b in bins])
    keep = cnt > 0

    # Diagonal reference y=x (perfect calibration)
    ax_main.plot([0, 1], [0, 1], ls="--", lw=0.9, color=COLOR_REF,
                 alpha=0.7, zorder=1)

    # Gap shading between curve and diagonal — directly visualises miscalib
    ax_main.fill_between(mp[keep], mp[keep], ma[keep],
                         color=color, alpha=0.15, linewidth=0, zorder=2)

    # Reliability line + markers
    ax_main.plot(mp[keep], ma[keep], "-", color=color, lw=2.0, zorder=4)
    ax_main.plot(mp[keep], ma[keep], "o", color=color, markersize=5.5,
                 markerfacecolor=color, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=5)

    # ECE / Brier inline label, top-left, regular weight (no bold per J6.5)
    ax_main.text(0.04, 0.96,
                 f"ECE = {ece:.3f}\nBrier = {brier:.3f}",
                 transform=ax_main.transAxes,
                 fontsize=9.5, va="top", ha="left",
                 color=color,
                 bbox=dict(facecolor="white", edgecolor=color, lw=0.6,
                          pad=3.5, alpha=0.95))

    # Concise panel sub-title (regular weight); panel letter is the only
    # bold element, placed above the axis flush left.
    ax_main.set_title(f"{title}  ($n_{{items}}={n_items:,}$)",
                      fontsize=9.5, pad=4, loc="center")
    ax_main.text(-0.08, 1.06, panel_letter, transform=ax_main.transAxes,
                 fontsize=12, fontweight="bold", va="bottom", ha="left",
                 color="#222222")

    # Axes cosmetics
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_aspect("equal")
    ax_main.set_xticks(np.arange(0, 1.01, 0.2))
    ax_main.set_yticks(np.arange(0, 1.01, 0.2))
    for sp in ("top", "right"):
        ax_main.spines[sp].set_visible(False)
    ax_main.grid(True, alpha=0.18, lw=0.5, zorder=0)
    ax_main.set_axisbelow(True)
    ax_main.tick_params(labelbottom=False)

    # Histogram of prediction distribution (alpha 0.5 per spec)
    ax_hist.bar(centers, frac, width=(1.0 / N_BINS) * 0.92,
                color=color, alpha=0.5, edgecolor="white", linewidth=0.5)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylim(0, max(0.18, float(frac.max()) * 1.1))
    ax_hist.set_xticks(np.arange(0, 1.01, 0.2))
    ax_hist.set_yticks([0.0, 0.1])
    for sp in ("top", "right"):
        ax_hist.spines[sp].set_visible(False)
    ax_hist.grid(True, axis="y", alpha=0.15, lw=0.5)
    ax_hist.set_axisbelow(True)
    ax_hist.set_xlabel("Predicted P(correct)", fontsize=10)


def main() -> None:
    summary = json.loads(JSON_PATH.read_text())
    train_ece = summary["train_test"]["train_ece"]
    train_brier = summary["train_test"]["train_brier"]
    test_ece = summary["train_test"]["test_ece"]
    test_brier = summary["train_test"]["test_brier"]

    cache = _require_bins_cache()
    n_train_items = cache["split"]["n_train_items"]
    n_test_items = cache["split"]["n_test_items"]
    train_bins = cache["train"]["bins"]
    test_bins = cache["test"]["bins"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    # No suptitle -> reclaim vertical space; tighter overall figure
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[3.4, 1.0],
        hspace=0.06, wspace=0.24,
        left=0.085, right=0.985, top=0.92, bottom=0.12,
    )
    ax_train = fig.add_subplot(gs[0, 0])
    ax_train_h = fig.add_subplot(gs[1, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    ax_test_h = fig.add_subplot(gs[1, 1])

    # Shared y-label semantics; left column carries y tick labels (J8/E9)
    ax_train.set_ylabel("Observed fraction correct", fontsize=10)
    ax_train_h.set_ylabel("Frac. of\npredictions", fontsize=8.5)
    ax_test.tick_params(labelleft=False)
    ax_test_h.tick_params(labelleft=False)

    _draw_panel(ax_train, ax_train_h, train_bins, train_ece, train_brier,
                n_train_items,
                title="Training items (in-distribution)",
                color=COLOR_TRAIN, panel_letter="a")
    _draw_panel(ax_test, ax_test_h, test_bins, test_ece, test_brier,
                n_test_items,
                title="Held-out test items (out-of-distribution)",
                color=COLOR_TEST, panel_letter="b")

    # Over-confidence annotation on the test panel: regular weight, slate
    # leader (per spec). Points at the widest miscalibration gap (P~0.85).
    ax_test.annotate(
        "model is over-confident\n($\\Delta$ECE $\\approx$ +0.08)",
        xy=(0.85, 0.65), xytext=(0.40, 0.30),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=9, color=COLOR_REF,
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor=COLOR_REF, lw=0.6,
                  pad=2.5, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=COLOR_REF, lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_calibration_reliability_REVISED.pdf"
    out_png = OUT_DIR / "fig_calibration_reliability_REVISED.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
