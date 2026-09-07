"""Benchmark-level prediction scatter (fig:benchpred), batch 2 revision.

Renders true vs predicted per-benchmark mean accuracy for every LLM, with
SkillEval and IRT 2PL overlaid on a 2x3 grid (5 benchmarks + Overall).

Style anchored to NEURIPS_FIGURE_CHECKLIST.md Part J (LOCKED palette,
no orange anywhere). Differences vs batch 1 (revised.py):
  - Tailwind-700 palette: SkillEval blue-700 #1D4ED8, IRT pink-700 #BE185D,
    reference slate-600 #475569 (NO orange / vermillion / yellow stars)
  - NO suptitle / figure-level header (finding lives in LaTeX caption only)
  - Tighter compact layout (~5.5" target with 2x3 grid -> small panels)
  - Less bold: bold reserved for panel letters; inline labels regular weight

Numbers traced to cdm_exploration/experiments/v2_benchmark_prediction.json
(per-benchmark Pearson r, paper checkpoint).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Part J1 (LOCKED) — Tailwind-700 / Option K, NO orange
SKILL_COLOR = "#1D4ED8"   # blue-700, protagonist (SkillEval)
IRT_COLOR   = "#BE185D"   # pink-700, comparator (IRT 2PL)
REF_COLOR   = "#475569"   # slate-600, y=x reference (dotted)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint

    seed_everything(42)

    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "cdm_exploration" / "figures" / "review_batch2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig_benchmark_prediction_REVISED.pdf"
    out_png = out_dir / "fig_benchmark_prediction_REVISED.png"

    print("Loading data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    benchmarks = [it["benchmark"] for it in items_data]
    bench_names = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

    all_items = np.arange(n_items)
    _, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  {n_llms} LLMs, {n_items} items, {len(test_items)} test items",
          flush=True)

    print("Loading SkillEval (text-conditioned CDM)...", flush=True)
    cdm = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
        cdm, device,
    )
    cdm = cdm.to(device); cdm.eval()

    print("Loading IRT 2PL...", flush=True)
    irt = IRT2PL(n_llms, n_items)
    load_checkpoint(
        Path("cdm_exploration/checkpoints/expanded/irt_2pl_v2.pt"), irt, device,
    )
    irt = irt.to(device); irt.eval()

    print("Predicting SkillEval...", flush=True)
    te_test = torch.tensor(text_embs[test_items], dtype=torch.float32, device=device)
    qr_test = torch.tensor(q_matrix[test_items], dtype=torch.float32, device=device)
    with torch.no_grad():
        stat = torch.sigmoid(cdm.student_emb(torch.arange(n_llms, device=device)))
        k_d = torch.sigmoid(cdm.k_difficulty_proj(te_test))
        e_d = torch.sigmoid(cdm.e_difficulty_proj(te_test))
        cdm_pred = np.zeros((n_llms, len(test_items)), dtype=np.float32)
        chunk = 256
        for i in range(0, n_llms, chunk):
            s = stat[i:i + chunk].unsqueeze(1)
            x = e_d.unsqueeze(0) * (s - k_d.unsqueeze(0)) * qr_test.unsqueeze(0)
            h1 = torch.sigmoid(cdm.prednet_full1(x))
            h2 = torch.sigmoid(cdm.prednet_full2(h1))
            p = torch.sigmoid(cdm.prednet_full3(h2)).squeeze(-1)
            cdm_pred[i:i + chunk] = p.cpu().numpy()

    print("Predicting IRT 2PL...", flush=True)
    with torch.no_grad():
        theta = irt.theta.weight.squeeze(-1)
        alpha = torch.exp(irt.alpha.weight[test_items].squeeze(-1))
        beta = irt.beta.weight[test_items].squeeze(-1)
        logit = theta.unsqueeze(1) * alpha.unsqueeze(0) - beta.unsqueeze(0)
        irt_pred = torch.sigmoid(logit).cpu().numpy()

    test_bench = np.array([benchmarks[i] for i in test_items])
    R_test = R[:, test_items]

    panels = {}
    for b in bench_names:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        panels[b] = (
            R_test[:, mask].mean(axis=1),
            cdm_pred[:, mask].mean(axis=1),
            irt_pred[:, mask].mean(axis=1),
        )
    panels["Overall"] = (
        R_test.mean(axis=1),
        cdm_pred.mean(axis=1),
        irt_pred.mean(axis=1),
    )

    # ─── Style (Part J: tight typography hierarchy, sans-serif, no bold) ───
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    # 2x3 compact grid, full text width 5.5", roughly square panels
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.95),
                             sharex=True, sharey=True)
    order = ["MATH", "BBH", "GPQA", "MuSR", "IFEval", "Overall"]
    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, (ax, name) in enumerate(zip(axes.flat, order)):
        true_acc, cdm_acc, irt_acc = panels[name]
        r_cdm, _ = pearsonr(true_acc, cdm_acc)
        r_irt, _ = pearsonr(true_acc, irt_acc)

        lo, hi = 0.0, 1.0

        # y=x reference line — slate-600 dotted (Part J: reference dotted)
        ax.plot([lo, hi], [lo, hi], ":", color=REF_COLOR, lw=0.9,
                alpha=0.85, zorder=1)

        # IRT first (background, square markers — redundant encoding G7)
        ax.scatter(true_acc, irt_acc, s=5, c=IRT_COLOR, alpha=0.32,
                   marker="s", linewidths=0, rasterized=True, zorder=2)
        # SkillEval on top (filled circle markers)
        ax.scatter(true_acc, cdm_acc, s=5, c=SKILL_COLOR, alpha=0.65,
                   marker="o", linewidths=0, rasterized=True, zorder=3)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.10, linewidth=0.4)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        # Panel letter (bold) + benchmark name (regular), upper-left
        ax.text(0.04, 0.965, panel_letters[idx],
                transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                va="top", ha="left")
        ax.text(0.18, 0.965, name,
                transform=ax.transAxes, fontsize=8.5, fontweight="regular",
                va="top", ha="left")

        # Compact r values in lower-right (regular weight, no monospace bbox)
        delta = r_cdm - r_irt
        sign = "+" if delta >= 0 else ""
        box_text = (f"SkillEval r={r_cdm:.3f}\n"
                    f"IRT 2PL  r={r_irt:.3f}\n"
                    f"$\\Delta r$ {sign}{delta:.3f}")
        ax.text(0.96, 0.04, box_text,
                transform=ax.transAxes, fontsize=6.8, va="bottom", ha="right",
                fontweight="regular",
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.22", alpha=0.9,
                          linewidth=0.4))

    # Shared axis labels (outer only)
    for ax in axes[1, :]:
        ax.set_xlabel("True benchmark accuracy", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted accuracy", fontsize=8.5)

    # Inline series labels in (f) Overall panel — no figure-level legend
    overall_ax = axes[1, 2]
    overall_ax.text(0.04, 0.83, "SkillEval", transform=overall_ax.transAxes,
                    fontsize=7.5, color=SKILL_COLOR,
                    fontweight="regular", ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.8, alpha=0.85))
    overall_ax.text(0.04, 0.74, "IRT 2PL", transform=overall_ax.transAxes,
                    fontsize=7.5, color=IRT_COLOR,
                    fontweight="regular", ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.8, alpha=0.85))

    # NO suptitle (per new constraint 1) — finding lives in LaTeX caption only

    plt.tight_layout(pad=0.3, w_pad=0.35, h_pad=0.5)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close()
    print(f"Saved: {out_pdf}", flush=True)
    print(f"Saved: {out_png}", flush=True)


if __name__ == "__main__":
    main()
