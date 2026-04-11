"""Per-skill alignment tax: base vs instruct/chat model comparison.

Compares theta profiles of base models vs their instruction-tuned
versions to discover which skills alignment improves and which it hurts.

Usage:
    python tools/run_alignment_tax.py device=cpu
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def find_pairs(names):
    """Find base-instruct pairs by stripping alignment suffixes."""
    name_to_idx = {n: i for i, n in enumerate(names)}

    def norm(n):
        return n.replace("__", "/").lower()

    suffixes = ["-instruct", "_instruct", "-chat", "-it", ".instruct", "-rlhf"]
    pairs = []
    seen = set()

    for idx, name in enumerate(names):
        nl = norm(name)
        for suf in suffixes:
            if suf in nl:
                pos = nl.rfind(suf)
                base_nl = nl[:pos] + nl[pos + len(suf):]
                for idx2, name2 in enumerate(names):
                    if idx2 == idx:
                        continue
                    nl2 = norm(name2)
                    if nl2 == base_nl or nl2.replace("-", "") == base_nl.replace("-", ""):
                        key = tuple(sorted([idx, idx2]))
                        if key not in seen:
                            seen.add(key)
                            pairs.append({"base_idx": idx2, "inst_idx": idx,
                                          "base_name": name2, "inst_name": name})
    return pairs


def detect_family(name):
    """Heuristic family detection from model name."""
    nl = name.lower()
    for fam in ["llama", "qwen", "mistral", "gemma", "phi", "falcon",
                 "yi", "solar", "olmo", "stablelm", "granite"]:
        if fam in nl:
            return fam
    return "other"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import setup_style
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        skill_labels = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)
    skill_names = [skill_labels[str(i)] for i in range(K)]

    # ── Load theta ──
    print("\nLoading trained theta...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, "cpu",
    )
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).numpy()  # (n_llms, K)

    # ════════════════════════════════════════════════════════════════
    # 1. FIND BASE-INSTRUCT PAIRS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("1. FINDING BASE-INSTRUCT PAIRS", flush=True)
    print(f"{'='*60}", flush=True)
    pairs = find_pairs(llm_names)
    print(f"  Found {len(pairs)} pairs", flush=True)

    if len(pairs) < 10:
        print("  WARNING: fewer than 10 pairs — results may be unreliable", flush=True)

    # Annotate families
    for p in pairs:
        p["family"] = detect_family(p["base_name"])

    family_counts = defaultdict(int)
    for p in pairs:
        family_counts[p["family"]] += 1
    print(f"  Pairs per family:", flush=True)
    for fam, c in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {fam:<10}: {c}", flush=True)

    # Sanity: instruct models generally have higher overall accuracy
    acc_base = np.array([R[p["base_idx"]].mean() for p in pairs])
    acc_inst = np.array([R[p["inst_idx"]].mean() for p in pairs])
    n_inst_better = (acc_inst > acc_base).sum()
    print(f"\n  Instruct accuracy > base: {n_inst_better}/{len(pairs)} pairs "
          f"({100*n_inst_better/len(pairs):.0f}%)", flush=True)
    print(f"  Mean accuracy: base={acc_base.mean():.3f}, instruct={acc_inst.mean():.3f}",
          flush=True)

    # ════════════════════════════════════════════════════════════════
    # 2. PER-SKILL ALIGNMENT DELTA
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("2. PER-SKILL ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    deltas = np.zeros((len(pairs), K))
    for i, p in enumerate(pairs):
        deltas[i] = theta[p["inst_idx"]] - theta[p["base_idx"]]

    mean_delta = deltas.mean(axis=0)  # (K,)
    std_delta = deltas.std(axis=0)
    n_improved = (deltas > 0).sum(axis=0)   # per skill, how many pairs improved
    n_degraded = (deltas < 0).sum(axis=0)

    # Overall stats
    n_skills_improved = (mean_delta > 0.01).sum()
    n_skills_degraded = (mean_delta < -0.01).sum()
    n_skills_neutral = K - n_skills_improved - n_skills_degraded
    print(f"  Skills improved (mean delta > 0.01): {n_skills_improved}", flush=True)
    print(f"  Skills degraded (mean delta < -0.01): {n_skills_degraded}", flush=True)
    print(f"  Skills neutral: {n_skills_neutral}", flush=True)
    print(f"  Overall mean delta: {mean_delta.mean():.4f}", flush=True)

    # Top improved
    sorted_idx = np.argsort(-mean_delta)
    print(f"\n  Top 10 IMPROVED skills (alignment helps):", flush=True)
    for rank, k in enumerate(sorted_idx[:10]):
        print(f"    +{mean_delta[k]:.4f} ({n_improved[k]}/{len(pairs)} pairs): "
              f"{skill_names[k][:55]}", flush=True)

    # Top degraded
    print(f"\n  Top 10 DEGRADED skills (alignment tax):", flush=True)
    for rank, k in enumerate(sorted_idx[-10:][::-1]):
        print(f"    {mean_delta[k]:+.4f} ({n_degraded[k]}/{len(pairs)} pairs): "
              f"{skill_names[k][:55]}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. PER-BENCHMARK ALIGNMENT TAX
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("3. PER-BENCHMARK ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    benchmarks = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
    bench_skills = {}
    for b in benchmarks:
        b_items = [i for i, it in enumerate(items_data) if it.get("benchmark") == b]
        Q_b = q_matrix[b_items]
        skills = set(np.where(Q_b.sum(axis=0) > 0)[0])
        bench_skills[b] = skills

    bench_delta = {}
    for b in benchmarks:
        skills = list(bench_skills[b])
        d = mean_delta[skills].mean()
        bench_delta[b] = float(d)
        n_up = (mean_delta[skills] > 0.01).sum()
        n_down = (mean_delta[skills] < -0.01).sum()
        print(f"  {b:<8}: mean delta={d:+.4f}, skills improved={n_up}, degraded={n_down}",
              flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. PER-FAMILY ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("4. PER-FAMILY ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    family_deltas = defaultdict(list)
    for i, p in enumerate(pairs):
        family_deltas[p["family"]].append(deltas[i])

    per_family = {}
    for fam in sorted(family_deltas.keys()):
        fd = np.array(family_deltas[fam])
        fm = fd.mean(axis=0)
        n_up = (fm > 0.01).sum()
        n_down = (fm < -0.01).sum()
        per_family[fam] = {
            "n_pairs": len(family_deltas[fam]),
            "mean_delta": float(fm.mean()),
            "skills_improved": int(n_up),
            "skills_degraded": int(n_down),
            "mean_delta_per_skill": fm.tolist(),
        }
        print(f"  {fam:<10} ({len(family_deltas[fam]):>2} pairs): "
              f"mean={fm.mean():+.4f}, up={n_up}, down={n_down}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 5. CONNECT TO PREREQUISITE DAG
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("5. ALIGNMENT TAX BY PREREQUISITE DEPTH", flush=True)
    print(f"{'='*60}", flush=True)

    prereq_path = Path("cdm_exploration/experiments/v2_skill_prerequisites.json")
    depth_delta = {}
    if prereq_path.exists():
        prereq = json.load(open(prereq_path))
        depth_map = prereq.get("depth_per_skill", {})
        depth_groups = defaultdict(list)
        for k in range(K):
            d = depth_map.get(str(k), 0)
            depth_groups[d].append(k)

        for d in sorted(depth_groups.keys()):
            skills = depth_groups[d]
            dm = mean_delta[skills].mean()
            depth_delta[d] = float(dm)
            print(f"  Depth {d} ({len(skills):>2} skills): mean delta = {dm:+.4f}",
                  flush=True)
    else:
        print("  Prerequisite data not found, skipping", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Fig 1: Top improved + degraded skills ──
    fig, ax = plt.subplots(figsize=(10, 7))
    n_show = 15
    top_up = sorted_idx[:n_show]
    top_down = sorted_idx[-n_show:][::-1]
    show_idx = np.concatenate([top_down, top_up])
    show_delta = mean_delta[show_idx]
    show_names = [skill_names[k][:40] for k in show_idx]
    colors = ["#C44E52" if d < 0 else "#55A868" for d in show_delta]

    y = np.arange(len(show_idx))
    ax.barh(y, show_delta, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(show_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean mastery change (instruct - base)", fontsize=11)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out1 = fig_dir / "fig_alignment_tax_skills.pdf"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}", flush=True)

    # ── Fig 2: Family heatmap ──
    fam_order = [f for f in sorted(per_family.keys()) if per_family[f]["n_pairs"] >= 3]
    if fam_order:
        heat = np.zeros((len(fam_order), K))
        for fi, fam in enumerate(fam_order):
            heat[fi] = np.array(per_family[fam]["mean_delta_per_skill"])

        # Sort skills by overall delta for visual clarity
        skill_order = np.argsort(mean_delta)
        heat = heat[:, skill_order]

        fig, ax = plt.subplots(figsize=(14, max(3, len(fam_order) * 0.8)))
        vmax = max(abs(heat.min()), abs(heat.max()), 0.05)
        im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(len(fam_order)))
        ax.set_yticklabels([f"{f} ({per_family[f]['n_pairs']})" for f in fam_order],
                           fontsize=9)
        ax.set_xlabel("Skills (sorted by mean delta)", fontsize=11)
        plt.colorbar(im, ax=ax, label="Mastery delta (instruct - base)", shrink=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough families", ha="center", va="center")
    out2 = fig_dir / "fig_alignment_tax_families.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Fig 3: Depth vs delta scatter ──
    fig, ax = plt.subplots(figsize=(7, 5))
    if depth_delta:
        depths = [depth_map.get(str(k), 0) for k in range(K)]
        ax.scatter(depths, mean_delta, s=30, alpha=0.5, color="#4C72B0")
        # Mean per depth
        for d in sorted(depth_delta.keys()):
            ax.scatter(d, depth_delta[d], s=150, color="#C44E52", zorder=5,
                       edgecolors="white", linewidths=1.5,
                       label=f"Depth {d} mean" if d == 0 else "")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Skill depth in prerequisite DAG", fontsize=12)
        ax.set_ylabel("Mean alignment delta", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No prerequisite data", ha="center", va="center")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out3 = fig_dir / "fig_alignment_tax_depth.pdf"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "alignment_tax",
        "n_llms": int(n_llms), "K": int(K),
        "n_pairs": len(pairs),
        "pairs": [{"base": p["base_name"], "instruct": p["inst_name"],
                    "family": p["family"]} for p in pairs],
        "instruct_better_pct": float(100 * n_inst_better / len(pairs)),
        "mean_acc_base": float(acc_base.mean()),
        "mean_acc_instruct": float(acc_inst.mean()),
        "overall_mean_delta": float(mean_delta.mean()),
        "skills_improved": int(n_skills_improved),
        "skills_degraded": int(n_skills_degraded),
        "skills_neutral": int(n_skills_neutral),
        "per_skill": {str(k): {"name": skill_names[k],
                                 "mean_delta": float(mean_delta[k]),
                                 "std_delta": float(std_delta[k]),
                                 "n_improved": int(n_improved[k]),
                                 "n_degraded": int(n_degraded[k])}
                       for k in range(K)},
        "top_improved": [{"skill": skill_names[int(k)],
                           "delta": float(mean_delta[k])}
                          for k in sorted_idx[:15]],
        "top_degraded": [{"skill": skill_names[int(k)],
                           "delta": float(mean_delta[k])}
                          for k in sorted_idx[-15:][::-1]],
        "per_benchmark": bench_delta,
        "per_family": {f: {"n_pairs": v["n_pairs"],
                            "mean_delta": v["mean_delta"],
                            "skills_improved": v["skills_improved"],
                            "skills_degraded": v["skills_degraded"]}
                        for f, v in per_family.items()},
        "per_depth": {str(k): v for k, v in depth_delta.items()},
    }

    out_json = Path("cdm_exploration/experiments/v2_alignment_tax.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="alignment_tax",
        config={"K": K, "n_pairs": len(pairs), "device": device},
        results={"n_pairs": len(pairs), "mean_delta": float(mean_delta.mean()),
                 "skills_improved": int(n_skills_improved),
                 "skills_degraded": int(n_skills_degraded),
                 "per_benchmark": bench_delta},
        split_info={"n_llms": n_llms, "K": K},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
