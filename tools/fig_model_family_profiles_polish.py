"""Polished family profiles figure (Llama vs Qwen)."""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

PHI_SIZES = [("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
             ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3)]


def parse_family_and_size(name):
    low = name.lower()
    fam = "Other"
    for kw, f in [("llama", "Llama"), ("qwen", "Qwen"), ("mistral", "Mistral"),
                  ("gemma", "Gemma"), ("phi-", "Phi"), ("phi2", "Phi"), ("phi3", "Phi")]:
        if kw in low:
            fam = f
            break
    suffix = name.split("__")[-1] if "__" in name else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return fam, float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return fam, float(m.group(1))
    if fam == "Phi":
        for kw, sz in PHI_SIZES:
            if kw in low:
                return fam, sz
    return fam, None


SHORT = {
    "Applying Quadratic Formula To Find Roots": "Quadratic formula",
    "Applying Trigonometric Identities To Solve Equatio": "Trig. identities",
    "Calculating Decay Length From Energy And Mass": "Decay length",
    "Identifying Structural Isomers In Organic Compound": "Organic isomers",
    "Recognizing Symmetry In Two Dimensional Figures": "2D symmetry",
    "Applying Boolean Logic To Combinations": "Boolean logic",
    "Solving Logical Word Puzzles": "Logical puzzles",
    "Following Multi-Step Instructions": "Multi-step instr.",
    "Reading Tables And Extracting Data": "Table reading",
    "Reasoning About Object Locations": "Object location",
}


def short(name, n=22):
    if name in SHORT:
        return SHORT[name]
    s = name
    for p in ("Applying ", "Identifying ", "Calculating ", "Solving ",
              "Working with ", "Recognizing ", "Reasoning About ",
              "Following ", "Reading "):
        if s.startswith(p):
            s = s[len(p):]
            break
    return s if len(s) <= n else s[:n - 1] + "…"


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

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        cluster_labels = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    skill_names = [cluster_labels[str(i)] for i in range(K)]

    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
                    net, device)
    net = net.to(device); net.eval()
    with torch.no_grad():
        raw = net.student_emb(torch.arange(n_llms, device=device)).cpu().numpy()
    mastery = 1.0 / (1.0 + np.exp(-raw))

    families, sizes = [], []
    for n in llm_names:
        f, s = parse_family_and_size(n)
        families.append(f); sizes.append(s)
    families = np.array(families)
    sizes = np.array([s if s is not None else np.nan for s in sizes])

    # Skill benchmark assignment
    skill_bm = {}
    for si in range(K):
        idx = np.where(q_matrix[:, si] > 0)[0]
        if len(idx) == 0:
            skill_bm[si] = "?"; continue
        bc = Counter()
        for it in idx[:100]:
            bc[items_data[int(it)].get("benchmark", "?")] += 1
        skill_bm[si] = bc.most_common(1)[0][0]

    # Pick 8 high-variance skills, 2 per major benchmark
    var = mastery.var(axis=0)
    selected, used = [], set()
    for tgt, k in [("MATH", 2), ("BBH", 2), ("GPQA", 2), ("MuSR", 1), ("IFEval", 1)]:
        cands = sorted([(si, var[si]) for si in range(K) if skill_bm[si] == tgt],
                       key=lambda x: -x[1])
        for si, _ in cands:
            sn = short(skill_names[si])
            if sn not in used:
                selected.append(si); used.add(sn)
                if sum(1 for s in selected if skill_bm[s] == tgt) >= k:
                    break
    if len(selected) < 8:
        for si in np.argsort(-var):
            if si not in selected:
                selected.append(si)
            if len(selected) >= 8:
                break
    top8 = np.array(selected[:8])

    size_buckets = [(0, 3, "<3B"), (3, 10, "3-10B"),
                    (10, 35, "10-35B"), (35, 200, "≥35B")]
    # Viridis-like sequential colormap (4 stops)
    bucket_colors = [cm.viridis(x) for x in [0.15, 0.40, 0.65, 0.88]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)

    for ax, fam, panel_letter in zip(axes, ["Llama", "Qwen"], ["a", "b"]):
        fam_mask = families == fam
        fam_sizes = sizes[fam_mask]
        fam_mastery = mastery[fam_mask]
        n_panel = int(fam_mask.sum())

        y_pos = np.arange(len(top8))
        bar_h = 0.20
        bucket_ns = []
        for bi, (lo, hi, label) in enumerate(size_buckets):
            bm_msk = (fam_sizes >= lo) & (fam_sizes < hi)
            n_b = int(bm_msk.sum())
            bucket_ns.append(n_b)
            if n_b == 0:
                continue
            mean_m = fam_mastery[bm_msk].mean(axis=0)
            ax.barh(y_pos + bi * bar_h, mean_m[top8], height=bar_h,
                    color=bucket_colors[bi], edgecolor="white", linewidth=0.4)

        ax.set_yticks(y_pos + 1.5 * bar_h)
        ax.set_yticklabels([short(skill_names[i]) for i in top8], fontsize=10)
        ax.set_xlabel("Mean mastery", fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.15, lw=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Panel label (no title)
        ax.text(0.0, 1.02, f"({panel_letter}) {fam}  ·  {n_panel} models",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")

    # Shared legend at top center
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor=bucket_colors[i], label=size_buckets[i][2],
              edgecolor="none") for i in range(len(size_buckets))
    ]
    fig.legend(handles=leg_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               fontsize=10, title="Parameters", title_fontsize=10,
               handlelength=1.2, columnspacing=2.0)

    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.92])
    out = fig_dir / "main_ready" / "fig_model_family_profiles.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE_KW)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
