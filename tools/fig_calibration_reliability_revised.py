"""Revised reliability diagram for fig:calib (gold-tier pass).

Renders a 2-panel reliability diagram comparing model calibration on
training items (well-calibrated, ECE=0.028) vs held-out test items
(mildly overconfident, ECE=0.108). Style anchored to
NEURIPS_FIGURE_CHECKLIST.md Part J.

Data provenance:
  - All ECE / Brier / N values are read from the experiment-log-verified
    JSON at cdm_exploration/experiments/v2_calibration_analysis.json
    (entry name=calibration_analysis, verified=True).
  - Per-bin reliability data (mean predicted P, observed fraction, count)
    is held in a sidecar cache at
    cdm_exploration/experiments/v2_calibration_reliability_bins.json.
    If the sidecar is missing, this script computes it once from the
    same checkpoint + data + 80/20 split that
    tools/run_calibration_analysis.py uses (seed_everything(42),
    test_size=0.2, random_state=42), writes the cache, then renders.
    On every subsequent run only the cache is read, so the render is
    pure and fast (<5 s).

Output:
  cdm_exploration/figures/review_batch1/fig_calibration_reliability_REVISED.pdf
  cdm_exploration/figures/review_batch1/fig_calibration_reliability_REVISED.png  (200 DPI)
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
OUT_DIR = REPO / "cdm_exploration" / "figures" / "review_batch1"
JSON_PATH = EXP / "v2_calibration_analysis.json"
BINS_CACHE = EXP / "v2_calibration_reliability_bins.json"

# Part J palette (Okabe-Ito): protagonist blue + a single contrasting hue
# for the held-out test panel. Vermillion is reserved for IrtNet in the
# routing figures, but here it functions as "out-of-distribution warning"
# rather than a method label, so the role does not collide.
COLOR_TRAIN = "#0072B2"   # Okabe-Ito blue (well-calibrated, in-dist)
COLOR_TEST  = "#D55E00"   # Okabe-Ito vermillion (overconfident, OOD)
COLOR_REF   = "#888888"   # diagonal reference
COLOR_GAP   = "#D55E00"   # gap shading (test only)
N_BINS = 20


def compute_bins(preds: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Return per-bin reliability stats (equal-width). Mirrors
    tools/run_calibration_analysis.py exactly."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == 0:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds > lo) & (preds <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"lo": float(lo), "hi": float(hi), "count": 0,
                        "mean_pred": float((lo + hi) / 2),
                        "mean_actual": 0.0, "frac": 0.0})
            continue
        out.append({"lo": float(lo), "hi": float(hi), "count": n,
                    "mean_pred": float(preds[mask].mean()),
                    "mean_actual": float(labels[mask].mean()),
                    "frac": float(n / len(preds))})
    return out


def _build_bins_cache() -> dict:
    """One-time per-bin cache build. Re-uses the exact pipeline of
    tools/run_calibration_analysis.py (same checkpoint, same 80/20 split,
    seed=42). Produces ~36M predictions on MPS/CPU; takes a few minutes.
    Subsequent figure renders read the cached JSON only."""
    print("[cache miss] Building per-bin reliability cache (one-time)...",
          flush=True)
    sys.path.insert(0, str(REPO))
    import torch
    from sklearn.model_selection import train_test_split
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint

    seed_everything(42)
    data_dir = REPO / "cdm_exploration" / "data" / "cdm_ready"
    if torch.backends.mps.is_available():
        device = resolve_device("mps")
    elif torch.cuda.is_available():
        device = resolve_device("cuda")
    else:
        device = "cpu"
    print(f"  device: {device}", flush=True)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, device,
    )
    net.to(device)
    net.eval()

    ITEM_BATCH = 200
    all_preds = np.zeros((n_llms, n_items), dtype=np.float32)
    for start in range(0, n_items, ITEM_BATCH):
        end = min(start + ITEM_BATCH, n_items)
        idx = np.arange(start, end)
        te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            k_d = torch.sigmoid(net.k_difficulty_proj(te))
            e_d = torch.sigmoid(net.e_difficulty_proj(te))
            theta_sig = torch.sigmoid(net.student_emb.weight)
            stat = theta_sig.unsqueeze(1)
            x = e_d.unsqueeze(0) * (stat - k_d.unsqueeze(0)) * qr.unsqueeze(0)
            x_flat = x.reshape(n_llms * len(idx), K)
            h1 = torch.sigmoid(net.prednet_full1(x_flat))
            h2 = torch.sigmoid(net.prednet_full2(h1))
            pred = torch.sigmoid(net.prednet_full3(h2)).reshape(n_llms, len(idx))
            all_preds[:, start:end] = pred.cpu().numpy()
        if (start // ITEM_BATCH) % 10 == 0:
            print(f"  items {start}-{end}/{n_items}", flush=True)

    train_idx, test_idx = train_test_split(
        np.arange(n_items), test_size=0.2, random_state=42)
    n_train, n_test = len(train_idx), len(test_idx)

    train_p = all_preds[:, train_idx].ravel()
    train_l = R[:, train_idx].ravel().astype(np.float32)
    test_p = all_preds[:, test_idx].ravel()
    test_l = R[:, test_idx].ravel().astype(np.float32)

    cache = {
        "source_json": str(JSON_PATH.relative_to(REPO)),
        "checkpoint": "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        "n_bins": N_BINS,
        "split": {"random_state": 42, "test_size": 0.2,
                  "n_items": int(n_items), "n_train_items": int(n_train),
                  "n_test_items": int(n_test), "n_llms": int(n_llms)},
        "train": {
            "bins": compute_bins(train_p, train_l, N_BINS),
            "n_pred": int(train_p.size),
        },
        "test": {
            "bins": compute_bins(test_p, test_l, N_BINS),
            "n_pred": int(test_p.size),
        },
    }
    BINS_CACHE.write_text(json.dumps(cache, indent=2))
    print(f"  wrote {BINS_CACHE}", flush=True)
    return cache


def _load_or_build_bins() -> dict:
    if BINS_CACHE.exists():
        return json.loads(BINS_CACHE.read_text())
    return _build_bins_cache()


def _draw_panel(ax_main, ax_hist, bins, ece, brier, n_pred, n_items,
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
                         color=COLOR_GAP if color == COLOR_TEST else color,
                         alpha=0.15, linewidth=0, zorder=2)

    # Reliability line + markers
    ax_main.plot(mp[keep], ma[keep], "-", color=color, lw=2.0, zorder=4)
    ax_main.plot(mp[keep], ma[keep], "o", color=color, markersize=5.5,
                 markerfacecolor=color, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=5)

    # ECE / Brier inline label, top-left, with halo box (white) for legibility
    ax_main.text(0.04, 0.96,
                 f"ECE = {ece:.3f}\nBrier = {brier:.3f}",
                 transform=ax_main.transAxes,
                 fontsize=9.5, va="top", ha="left", fontweight="bold",
                 color=color,
                 bbox=dict(facecolor="white", edgecolor=color, lw=0.6,
                          pad=3.5, alpha=0.95))

    # Panel sub-title (concise). Panel letter goes ABOVE the axis to the
    # far left so it never collides with curves or the ECE label box.
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

    # Histogram of prediction distribution
    ax_hist.bar(centers, frac, width=(1.0 / N_BINS) * 0.92,
                color=color, alpha=0.55, edgecolor="white", linewidth=0.5)
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

    cache = _load_or_build_bins()
    n_train_items = cache["split"]["n_train_items"]
    n_test_items = cache["split"]["n_test_items"]
    train_bins = cache["train"]["bins"]
    test_bins = cache["test"]["bins"]
    train_n = cache["train"]["n_pred"]
    test_n = cache["test"]["n_pred"]

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

    fig = plt.figure(figsize=(7.4, 4.7))
    # 2 columns x 2 rows; bottom row = small histogram strip
    gs = fig.add_gridspec(
        2, 2, height_ratios=[3.4, 1.0],
        hspace=0.06, wspace=0.28,
        left=0.085, right=0.985, top=0.80, bottom=0.10,
    )
    ax_train = fig.add_subplot(gs[0, 0])
    ax_train_h = fig.add_subplot(gs[1, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    ax_test_h = fig.add_subplot(gs[1, 1])

    # Shared y-label for the reliability axes (left only) per Part J
    ax_train.set_ylabel("Observed fraction correct", fontsize=10)
    ax_train_h.set_ylabel("Frac. of\npredictions", fontsize=8.5)
    # Suppress duplicate y tick labels on right panel (E9 / J8)
    ax_test.tick_params(labelleft=False)
    ax_test_h.tick_params(labelleft=False)

    _draw_panel(ax_train, ax_train_h, train_bins, train_ece, train_brier,
                train_n, n_train_items,
                title="Training items (in-distribution)",
                color=COLOR_TRAIN, panel_letter="a")
    _draw_panel(ax_test, ax_test_h, test_bins, test_ece, test_brier,
                test_n, n_test_items,
                title="Held-out test items (out-of-distribution)",
                color=COLOR_TEST, panel_letter="b")

    # Inline finding annotation on the test panel: highlight the
    # over-confidence gap with one short sentence. Place it BELOW the
    # diagonal (where the gap shading lives) and aim the leader at the
    # widest miscalibration bin (predicted P~0.85).
    ax_test.annotate(
        "model is over-confident\n($\\Delta$ECE $\\approx$ +0.08)",
        xy=(0.85, 0.65), xytext=(0.40, 0.30),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=9, color=COLOR_TEST, fontweight="bold",
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor=COLOR_TEST, lw=0.6,
                  pad=2.5, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=COLOR_TEST, lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )

    # Top suptitle = finding-first one-liner; two lines so it does not
    # collide with the per-panel sub-titles below.
    fig.suptitle(
        "Calibration is tight on training items (ECE $=$ 0.028)\n"
        "but mildly over-confident on held-out items (ECE $=$ 0.108)",
        fontsize=11, fontweight="bold", y=0.99, x=0.085, ha="left",
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
