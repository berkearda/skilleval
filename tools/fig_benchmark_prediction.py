"""Benchmark-level prediction scatter (Figure 3).

Per-benchmark mean accuracy: true vs predicted, CDM and IRT overlaid.
6 panels = 5 benchmarks + Overall. Pearson r in each title.
"""
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

CDM_COLOR = "#1E40AF"   # deep blue (authoritative)
IRT_COLOR = "#FCA5A5"   # soft coral (de-emphasized)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    setup_style()

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    device = resolve_device(cfg.device)

    print("Loading...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    benchmarks = [it["benchmark"] for it in items_data]
    bench_names = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

    # Same split as baselines
    all_items = np.arange(n_items)
    _, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    test_set = set(test_items.tolist())
    print(f"  {n_llms} LLMs, {n_items} items, {len(test_items)} test items", flush=True)

    # ── Load CDM ──
    print("Loading CDM...", flush=True)
    cdm = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
                    cdm, device)
    cdm = cdm.to(device); cdm.eval()

    # ── Load IRT ──
    print("Loading IRT...", flush=True)
    irt = IRT2PL(n_llms, n_items)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/irt_2pl_v2.pt"), irt, device)
    irt = irt.to(device); irt.eval()

    # ── Predict P(LLM, item) for ALL test items at once ──
    print("Predicting CDM...", flush=True)
    te_test = torch.tensor(text_embs[test_items], dtype=torch.float32, device=device)
    qr_test = torch.tensor(q_matrix[test_items], dtype=torch.float32, device=device)
    with torch.no_grad():
        stat = torch.sigmoid(cdm.student_emb(torch.arange(n_llms, device=device)))  # (S, K)
        k_d = torch.sigmoid(cdm.k_difficulty_proj(te_test))  # (T, K)
        e_d = torch.sigmoid(cdm.e_difficulty_proj(te_test))  # (T, K)
        cdm_pred = np.zeros((n_llms, len(test_items)), dtype=np.float32)
        chunk = 256
        for i in range(0, n_llms, chunk):
            s = stat[i:i+chunk].unsqueeze(1)            # (b, 1, K)
            x = e_d.unsqueeze(0) * (s - k_d.unsqueeze(0)) * qr_test.unsqueeze(0)
            h1 = torch.sigmoid(cdm.prednet_full1(x))
            h2 = torch.sigmoid(cdm.prednet_full2(h1))
            p = torch.sigmoid(cdm.prednet_full3(h2)).squeeze(-1)
            cdm_pred[i:i+chunk] = p.cpu().numpy()

    print("Predicting IRT...", flush=True)
    with torch.no_grad():
        theta = irt.theta.weight.squeeze(-1)            # (S,)
        alpha = torch.exp(irt.alpha.weight[test_items].squeeze(-1))  # (T,)
        beta = irt.beta.weight[test_items].squeeze(-1)               # (T,)
        logit = theta.unsqueeze(1) * alpha.unsqueeze(0) - beta.unsqueeze(0)
        irt_pred = torch.sigmoid(logit).cpu().numpy()

    # ── Aggregate per (LLM, benchmark) over TEST items only ──
    test_bench = np.array([benchmarks[i] for i in test_items])
    R_test = R[:, test_items]  # (S, T) ground truth

    panels = {}
    for b in bench_names:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        true_acc = R_test[:, mask].mean(axis=1)
        cdm_acc = cdm_pred[:, mask].mean(axis=1)
        irt_acc = irt_pred[:, mask].mean(axis=1)
        panels[b] = (true_acc, cdm_acc, irt_acc)

    # All benchmarks combined
    panels["All benchmarks"] = (R_test.mean(axis=1),
                                cdm_pred.mean(axis=1),
                                irt_pred.mean(axis=1))

    # ── Figure: 2x3 grid ──
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    order = ["MATH", "BBH", "GPQA", "MuSR", "IFEval", "All benchmarks"]

    for ax, name in zip(axes.flat, order):
        true_acc, cdm_acc, irt_acc = panels[name]
        r_cdm, _ = pearsonr(true_acc, cdm_acc)
        r_irt, _ = pearsonr(true_acc, irt_acc)
        d_r = r_cdm - r_irt

        lo, hi = 0.0, 1.0
        # Solid dark diagonal reference
        ax.plot([lo, hi], [lo, hi], "-", color="#333333", lw=1.0,
                alpha=0.85, zorder=1)

        # IRT first (de-emphasized), CDM on top (authoritative)
        ax.scatter(true_acc, irt_acc, s=5, c=IRT_COLOR, alpha=0.30,
                   linewidths=0, rasterized=True, zorder=2)
        ax.scatter(true_acc, cdm_acc, s=5, c=CDM_COLOR, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=3)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, linewidth=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        # Panel name top-left
        ax.text(0.04, 0.96, name, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")
        # r values bottom-right (out of data path)
        box_text = (f"CDM  {r_cdm:.3f}\n"
                    f"IRT  {r_irt:.3f}\n"
                    f"Δr  {d_r:+.3f}")
        ax.text(0.96, 0.04, box_text,
                transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.35", alpha=0.95, linewidth=0.7))

        if ax in axes[:, 0]:
            ax.set_ylabel("Predicted accuracy", fontsize=11)
        if ax in axes[1, :]:
            ax.set_xlabel("True accuracy", fontsize=11)

    # Shared legend at top
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="CDMEval",
               markerfacecolor=CDM_COLOR, markersize=8),
        Line2D([0], [0], marker="o", color="w", label="IRT 2PL",
               markerfacecolor=IRT_COLOR, markersize=8),
        Line2D([0], [0], color="#333333", lw=1.0, label="perfect prediction"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.96])

    out_pdf = fig_dir / "main_ready" / "fig_benchmark_prediction.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, **SAVE_KW)
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved: {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
