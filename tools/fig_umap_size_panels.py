"""UMAP of mastery profiles, faceted by family, colored by parameter size.

Replaces fig_umap_profiles.pdf. Story: within each family, do skill profiles
trace a clean trajectory with parameter count?
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


PHI_SIZES = [
    ("phi-3.5-mini", 3.8), ("phi-3.5-moe", 6.6), ("phi-3-mini", 3.8),
    ("phi-3-small", 7.0), ("phi-3-medium", 14.0), ("phi-3", 3.8),
    ("phi-2", 2.7), ("phi-1_5", 1.3), ("phi-1.5", 1.3), ("phi-1", 1.3),
    ("phi2", 2.7), ("phi4", 14.0), ("phi-4", 14.0),
]


def parse_family_and_size(name):
    low = name.lower()
    rules = [("llama", "Llama"), ("qwen", "Qwen"), ("mistral", "Mistral"),
             ("gemma", "Gemma"), ("phi-", "Phi"), ("phi2", "Phi"), ("phi3", "Phi"),
             ("phi4", "Phi")]
    family = "Other"
    for kw, fam in rules:
        if kw in low:
            family = fam
            break
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return family, float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return family, float(m.group(1))
    if family == "Phi":
        for kw, sz in PHI_SIZES:
            if kw in low:
                return family, sz
    return family, None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
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
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms = R.shape[0]
    K = q_matrix.shape[1]

    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device); net.eval()
    with torch.no_grad():
        raw = net.student_emb(torch.arange(n_llms, device=device)).cpu().numpy()
    mastery = 1.0 / (1.0 + np.exp(-raw))

    families, sizes = [], []
    for name in llm_names:
        f, s = parse_family_and_size(name)
        families.append(f); sizes.append(s)
    families = np.array(families)
    sizes = np.array([s if s is not None else np.nan for s in sizes])
    has_size = ~np.isnan(sizes)
    print(f"  parsed sizes for {has_size.sum()}/{n_llms} ({has_size.mean()*100:.1f}%)", flush=True)

    print("UMAP...", flush=True)
    import umap
    coords = umap.UMAP(n_components=2, metric="cosine", random_state=42,
                       n_neighbors=30, min_dist=0.1).fit_transform(mastery)

    top5 = ["Llama", "Qwen", "Gemma", "Mistral", "Phi"]

    # Global axis limits and color norm
    xlim = (coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
    ylim = (coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)
    s_valid = sizes[has_size]
    norm = LogNorm(vmin=max(0.3, np.percentile(s_valid, 1)),
                   vmax=np.percentile(s_valid, 99))

    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharex=True, sharey=True)

    rho_per_family = {}
    for ax, fam in zip(axes, top5):
        bg = (families != fam)
        ax.scatter(coords[bg, 0], coords[bg, 1], c="#f2f2f2", s=3,
                   alpha=0.5, linewidths=0, rasterized=True)

        m = (families == fam) & has_size
        n_total = (families == fam).sum()
        n_sized = m.sum()
        sc = ax.scatter(coords[m, 0], coords[m, 1], c=sizes[m],
                        cmap="plasma", norm=norm, s=16, alpha=0.9,
                        edgecolors="white", linewidths=0.2, rasterized=True)

        # Spearman ρ between log(size) and the better of UMAP-1 / UMAP-2
        if n_sized >= 5:
            log_s = np.log(sizes[m])
            r1, _ = spearmanr(log_s, coords[m, 0])
            r2, _ = spearmanr(log_s, coords[m, 1])
            rho = r1 if abs(r1) >= abs(r2) else r2
            axis_used = "U1" if abs(r1) >= abs(r2) else "U2"
        else:
            rho, axis_used = float("nan"), "—"
        rho_per_family[fam] = rho

        # Subtle PAGA-style binned centroid path (under data, thin grey)
        # Skip Mistral and Phi: size range too narrow for a meaningful path
        if n_sized >= 12 and fam not in ("Mistral", "Phi"):
            from matplotlib.patches import FancyArrowPatch
            s_vals = sizes[m]
            pts = coords[m]
            log_s = np.log(s_vals)
            n_bins = 5
            edges = np.quantile(log_s, np.linspace(0, 1, n_bins + 1))
            edges[-1] += 1e-9
            cents, bin_means = [], []
            for b in range(n_bins):
                mask_b = (log_s >= edges[b]) & (log_s < edges[b + 1])
                if mask_b.sum() >= 2:
                    cents.append(pts[mask_b].mean(axis=0))
                    bin_means.append(s_vals[mask_b].mean())
            cents = np.array(cents)
            # Only draw the path if it spans a meaningful distance and has ≥3 nodes
            span_ok = (len(cents) >= 3 and
                       np.linalg.norm(cents[-1] - cents[0]) >
                       0.18 * (xlim[1] - xlim[0]))
            if span_ok:
                # Thin grey connecting line, drawn under everything except bg
                if len(cents) >= 3:
                    ax.plot(cents[:-1, 0], cents[:-1, 1], "-",
                            color="#444444", lw=1.0, alpha=0.85,
                            solid_capstyle="round", zorder=2.5)
                # Subtle terminal arrow
                arrow = FancyArrowPatch(
                    posA=cents[-2], posB=cents[-1],
                    arrowstyle="-|>,head_length=4,head_width=2.5",
                    color="#444444", lw=1.0, mutation_scale=1, zorder=2.6)
                ax.add_patch(arrow)
                # Small filled centroid dots with white halo for contrast
                ax.scatter(cents[:, 0], cents[:, 1], s=46,
                           facecolor="white", edgecolor="white",
                           linewidths=0, zorder=2.65)
                ax.scatter(cents[:, 0], cents[:, 1], s=24,
                           c=bin_means, cmap="plasma", norm=norm,
                           edgecolor="#222222", linewidths=0.7, zorder=2.7)
        if fam in ("Mistral", "Phi"):
            # Document why no overlay was drawn
            ax.text(0.5, 0.08, "narrow size range\n(colormap only)",
                    transform=ax.transAxes, fontsize=9, color="#222222",
                    ha="center", va="center", style="italic",
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#888888",
                              boxstyle="round,pad=0.3", alpha=0.95,
                              linewidth=0.8))

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        rho_str = f"ρ={rho:+.2f}" if not np.isnan(rho) else "ρ=—"
        ax.text(0.02, 0.98, f"{fam}\n{rho_str}",
                transform=ax.transAxes, fontsize=11, va="top", ha="left",
                fontweight="bold")
        ax.text(0.98, 0.98, f"n={n_sized}/{n_total}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="#555555")

    # UMAP axis hint on first panel
    axes[0].annotate("", xy=(0.18, 0.04), xytext=(0.04, 0.04),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    axes[0].annotate("", xy=(0.04, 0.18), xytext=(0.04, 0.04),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    axes[0].text(0.20, 0.04, "UMAP-1", transform=axes[0].transAxes,
                 fontsize=8, color="#333", va="center")
    axes[0].text(0.05, 0.20, "UMAP-2", transform=axes[0].transAxes,
                 fontsize=8, color="#333")

    print("Spearman rho (log size vs UMAP):", rho_per_family, flush=True)

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01,
                        ticks=[0.5, 1, 3, 7, 13, 34, 70])
    cbar.ax.set_yticklabels(["0.5B", "1B", "3B", "7B", "13B", "34B", "70B"])
    cbar.set_label("Parameters", fontsize=10)
    cbar.outline.set_visible(False)

    out_main = fig_dir / "fig_umap_size_panels.pdf"
    out_polish = fig_dir / "appendix_ready" / "fig_umap_size_panels.pdf"
    out_polish.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_main, **SAVE_KW)
    fig.savefig(out_polish, **SAVE_KW)
    fig.savefig(str(out_polish).replace(".pdf", ".png"), dpi=180,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_main}\nSaved: {out_polish}", flush=True)


if __name__ == "__main__":
    main()
