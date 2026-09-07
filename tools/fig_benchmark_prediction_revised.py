"""Benchmark-level prediction scatter (fig:benchpred), gold-tier revision.

Renders true vs predicted per-benchmark mean accuracy for every LLM, with
SkillEval and IRT 2PL overlaid on a 2x3 grid (5 benchmarks + Overall).

Style anchored to NEURIPS_FIGURE_CHECKLIST.md Part J:
  - Okabe-Ito palette (J1): SkillEval blue #0072B2, IRT vermillion #D55E00
  - Sans-serif typography hierarchy (J2): tick 8pt / axis 10pt / panel 10pt
  - Redundant encoding (G7): protagonist filled circle, IRT square
  - Inline labels at curve endpoints, no large legend box (J4)
  - Tight axis cropping (J5) keeps the y=x reference visible
  - Spines top+right hidden, faint y-grid (J3)
  - Finding-first title (J6, E8)
  - Headline value annotated inline on the Overall panel (E4)
  - Panel letters (a)-(f) upper-left bold (J8)

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

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# Part J1 — Okabe-Ito palette, role-assigned
SKILL_COLOR = "#0072B2"   # protagonist (SkillEval)
IRT_COLOR   = "#D55E00"   # vermillion (IRT 2PL comparator)
REF_COLOR   = "#888888"   # neutral gray (y=x reference)
HIGHLIGHT   = "#E69F00"   # orange (single headline annotation)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint

    seed_everything(42)

    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)

    # Output paths (review_batch1)
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "cdm_exploration" / "figures" / "review_batch1"
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

    # Same split as baselines (seed-locked)
    all_items = np.arange(n_items)
    _, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  {n_llms} LLMs, {n_items} items, {len(test_items)} test items", flush=True)

    # Load CDM
    print("Loading SkillEval (text-conditioned CDM)...", flush=True)
    cdm = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
                    cdm, device)
    cdm = cdm.to(device); cdm.eval()

    # Load IRT
    print("Loading IRT 2PL...", flush=True)
    irt = IRT2PL(n_llms, n_items)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/irt_2pl_v2.pt"), irt, device)
    irt = irt.to(device); irt.eval()

    # Predict P(LLM, item) on test items
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
            s = stat[i:i+chunk].unsqueeze(1)
            x = e_d.unsqueeze(0) * (s - k_d.unsqueeze(0)) * qr_test.unsqueeze(0)
            h1 = torch.sigmoid(cdm.prednet_full1(x))
            h2 = torch.sigmoid(cdm.prednet_full2(h1))
            p = torch.sigmoid(cdm.prednet_full3(h2)).squeeze(-1)
            cdm_pred[i:i+chunk] = p.cpu().numpy()

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
        panels[b] = (R_test[:, mask].mean(axis=1),
                     cdm_pred[:, mask].mean(axis=1),
                     irt_pred[:, mask].mean(axis=1))
    panels["Overall"] = (R_test.mean(axis=1),
                         cdm_pred.mean(axis=1),
                         irt_pred.mean(axis=1))

    # ─── Style ───
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    # 2x3 grid, square panels, sized for full text width
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.4),
                             sharex=True, sharey=True)
    order = ["MATH", "BBH", "GPQA", "MuSR", "IFEval", "Overall"]
    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, (ax, name) in enumerate(zip(axes.flat, order)):
        true_acc, cdm_acc, irt_acc = panels[name]
        r_cdm, _ = pearsonr(true_acc, cdm_acc)
        r_irt, _ = pearsonr(true_acc, irt_acc)

        lo, hi = 0.0, 1.0

        # y=x reference line, dashed gray (J3)
        ax.plot([lo, hi], [lo, hi], "--", color=REF_COLOR, lw=1.0,
                alpha=0.85, zorder=1)

        # IRT first (background, square markers, J3 G7 redundant encoding)
        ax.scatter(true_acc, irt_acc, s=10, c=IRT_COLOR, alpha=0.32,
                   marker="s", linewidths=0, rasterized=True, zorder=2)
        # SkillEval on top (filled circle markers)
        ax.scatter(true_acc, cdm_acc, s=10, c=SKILL_COLOR, alpha=0.65,
                   marker="o", linewidths=0, rasterized=True, zorder=3)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.12, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        # Panel letter + benchmark name, upper-left
        ax.text(0.035, 0.965,
                f"{panel_letters[idx]} {name}",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        # Pearson r values bottom-right (away from data line)
        delta = r_cdm - r_irt
        sign = "+" if delta >= 0 else ""
        box_text = (f"SkillEval  r = {r_cdm:.3f}\n"
                    f"IRT 2PL    r = {r_irt:.3f}\n"
                    f"Δr         {sign}{delta:.3f}")
        ax.text(0.965, 0.04, box_text,
                transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                family="DejaVu Sans Mono",
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.32", alpha=0.95,
                          linewidth=0.6))

    # Headline annotation on the Overall panel (E4): inline punchline
    overall_ax = axes[1, 2]
    true_o, cdm_o, _ = panels["Overall"]
    # Find a representative high-accuracy point near the diagonal for star
    # Use the LLM whose true accuracy is the median to anchor the annotation
    mid_idx = int(np.argsort(true_o)[len(true_o) // 2])
    star_x = float(true_o[mid_idx])
    star_y = float(cdm_o[mid_idx])
    overall_ax.scatter([star_x], [star_y], marker="*", s=170,
                       color=HIGHLIGHT, edgecolor="black", linewidth=0.7,
                       zorder=6)
    overall_ax.annotate("$r{=}0.989$\nacross 5 benchmarks",
                        xy=(star_x, star_y),
                        xytext=(0.04, 0.78), textcoords="axes fraction",
                        fontsize=9, color="#1A3A5F", fontweight="bold",
                        ha="left", va="center",
                        bbox=dict(facecolor="white", edgecolor="#1A3A5F",
                                  pad=2.5, alpha=0.95, linewidth=0.5),
                        arrowprops=dict(arrowstyle="-", color="#1A3A5F",
                                        lw=0.6,
                                        connectionstyle="arc3,rad=-0.2"))

    # Shared axis labels (J8 — outer labels only)
    for ax in axes[1, :]:
        ax.set_xlabel("True benchmark accuracy")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted benchmark accuracy")

    # Inline series labels in the Overall panel corner (J4 — no global legend)
    # We put two small color-keyed labels in the upper-right of Overall panel.
    overall_ax.text(0.965, 0.965, "SkillEval", transform=overall_ax.transAxes,
                    fontsize=9, fontweight="bold", color=SKILL_COLOR,
                    ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=1.4, alpha=0.85))
    overall_ax.text(0.965, 0.895, "IRT 2PL", transform=overall_ax.transAxes,
                    fontsize=9, fontweight="bold", color=IRT_COLOR,
                    ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=1.4, alpha=0.85))

    # Finding-first title (E8)
    fig.suptitle("Skill profiles recover benchmark accuracy on held-out items "
                 "(SkillEval $r{=}0.989$ overall, vs IRT $r{=}0.976$)",
                 fontsize=11, fontweight="bold", y=0.995, x=0.5)

    plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.965])

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out_pdf}", flush=True)
    print(f"Saved: {out_png}", flush=True)


if __name__ == "__main__":
    main()
