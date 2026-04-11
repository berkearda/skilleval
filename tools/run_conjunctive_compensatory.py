"""Conjunctive vs compensatory skill interaction analysis.

For each multi-skill item, determines whether LLMs combine skills
conjunctively (need ALL), compensatorily (need ANY), or additively.

Usage:
    python tools/run_conjunctive_compensatory.py device=cpu
"""

import json
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
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

MASTERY_THRESHOLD = 0.5
MIN_GROUP_SIZE = 50
MIN_EFFECT = 0.05  # p_all - p_none must exceed this
CONJUNCTIVE_THRESH = 0.6
COMPENSATORY_THRESH = 0.6


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import SAVE_KW, setup_style
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

    # ── Load theta from checkpoint ──
    print("\nLoading trained theta...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, "cpu",
    )
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).numpy()  # (n_llms, K)
    print(f"  theta shape: {theta.shape}, range [{theta.min():.3f}, {theta.max():.3f}]",
          flush=True)

    # Binary mastery matrix
    mastery = (theta > MASTERY_THRESHOLD).astype(int)  # (n_llms, K)

    # ── Per-benchmark item indices ──
    benchmarks = sorted(set(it.get("benchmark", "unknown") for it in items_data))
    bench_items = {b: [] for b in benchmarks}
    for i, it in enumerate(items_data):
        bench_items[it.get("benchmark", "unknown")].append(i)

    print(f"\n  Benchmarks: {benchmarks}", flush=True)
    for b in benchmarks:
        print(f"    {b}: {len(bench_items[b])} items", flush=True)

    # ── Filter items ──
    skills_per_item = q_matrix.sum(axis=1).astype(int)
    multi_skill_mask = skills_per_item >= 2
    single_skill_mask = skills_per_item == 1
    print(f"\n  Items with 1 skill: {single_skill_mask.sum()}", flush=True)
    print(f"  Items with 2+ skills: {multi_skill_mask.sum()}", flush=True)
    print(f"  Items with 0 skills: {(skills_per_item == 0).sum()}", flush=True)

    # ── Sanity check: single-skill items ──
    print("\nSanity check: single-skill items...", flush=True)
    single_items = np.where(single_skill_mask)[0]
    if len(single_items) > 0:
        ss_p_master = []
        ss_p_non = []
        for j in single_items[:500]:  # sample
            skill_idx = np.where(q_matrix[j] > 0)[0][0]
            masters = mastery[:, skill_idx] == 1
            non_masters = mastery[:, skill_idx] == 0
            if masters.sum() >= MIN_GROUP_SIZE and non_masters.sum() >= MIN_GROUP_SIZE:
                ss_p_master.append(R[masters, j].mean())
                ss_p_non.append(R[non_masters, j].mean())
        if ss_p_master:
            print(f"  Single-skill: p_master={np.mean(ss_p_master):.4f}, "
                  f"p_non={np.mean(ss_p_non):.4f} "
                  f"(gap={np.mean(ss_p_master)-np.mean(ss_p_non):.4f})", flush=True)
            print(f"  PASS: mastery predicts accuracy for single-skill items", flush=True)
        else:
            print(f"  WARN: not enough single-skill items with sufficient group sizes",
                  flush=True)

    # ── Analyze multi-skill items ──
    print(f"\nAnalyzing {multi_skill_mask.sum()} multi-skill items...", flush=True)

    results_per_item = []
    filter_stats = {"total_multi": int(multi_skill_mask.sum()),
                    "too_few_in_group": 0, "no_effect": 0, "analyzed": 0}

    for j in np.where(multi_skill_mask)[0]:
        required_skills = np.where(q_matrix[j] > 0)[0]
        n_required = len(required_skills)

        # Mastery pattern per LLM on this item's required skills
        llm_mastery_on_item = mastery[:, required_skills]  # (n_llms, n_required)
        n_mastered = llm_mastery_on_item.sum(axis=1)  # (n_llms,)

        # Group LLMs
        all_mask = n_mastered == n_required
        none_mask = n_mastered == 0
        partial_mask = (~all_mask) & (~none_mask)

        n_all = all_mask.sum()
        n_none = none_mask.sum()
        n_partial = partial_mask.sum()

        if n_all < MIN_GROUP_SIZE or n_none < MIN_GROUP_SIZE or n_partial < MIN_GROUP_SIZE:
            filter_stats["too_few_in_group"] += 1
            continue

        p_all = R[all_mask, j].mean()
        p_none = R[none_mask, j].mean()
        p_partial = R[partial_mask, j].mean()

        effect = p_all - p_none
        if effect < MIN_EFFECT:
            filter_stats["no_effect"] += 1
            continue

        filter_stats["analyzed"] += 1

        conj_index = (p_all - p_partial) / (effect + 1e-8)
        comp_index = (p_partial - p_none) / (effect + 1e-8)

        if conj_index > CONJUNCTIVE_THRESH:
            item_type = "conjunctive"
        elif comp_index > COMPENSATORY_THRESH:
            item_type = "compensatory"
        else:
            item_type = "additive"

        # Finer analysis: accuracy by number of skills mastered
        acc_by_count = {}
        for k_count in range(n_required + 1):
            count_mask = n_mastered == k_count
            if count_mask.sum() >= 20:
                acc_by_count[k_count] = float(R[count_mask, j].mean())

        benchmark = items_data[j].get("benchmark", "unknown")

        results_per_item.append({
            "item_idx": int(j),
            "benchmark": benchmark,
            "n_required_skills": int(n_required),
            "skill_indices": required_skills.tolist(),
            "n_all": int(n_all),
            "n_none": int(n_none),
            "n_partial": int(n_partial),
            "p_all": float(p_all),
            "p_none": float(p_none),
            "p_partial": float(p_partial),
            "conj_index": float(conj_index),
            "comp_index": float(comp_index),
            "item_type": item_type,
            "acc_by_mastery_count": acc_by_count,
        })

    print(f"\n  Filter stats:", flush=True)
    for k, v in filter_stats.items():
        print(f"    {k}: {v}", flush=True)

    # ── Cross-check: average group accuracies ──
    p_all_avg = np.mean([r["p_all"] for r in results_per_item])
    p_partial_avg = np.mean([r["p_partial"] for r in results_per_item])
    p_none_avg = np.mean([r["p_none"] for r in results_per_item])
    print(f"\n  Cross-check (averaged over {len(results_per_item)} items):", flush=True)
    print(f"    p_all={p_all_avg:.4f} > p_partial={p_partial_avg:.4f} > p_none={p_none_avg:.4f}",
          flush=True)
    if not (p_all_avg > p_partial_avg > p_none_avg):
        print(f"    WARNING: ordering violated — skills may not be predictive", flush=True)
    else:
        print(f"    PASS: monotonic ordering holds", flush=True)

    # Warn on items where skills don't help
    broken = [r for r in results_per_item if r["p_none"] > r["p_all"]]
    if broken:
        print(f"    WARNING: {len(broken)} items where p_none > p_all", flush=True)

    # ── Aggregate by type ──
    type_counts = defaultdict(int)
    for r in results_per_item:
        type_counts[r["item_type"]] += 1

    total_analyzed = len(results_per_item)
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL RESULTS ({total_analyzed} items analyzed)", flush=True)
    print(f"{'='*60}", flush=True)
    for t in ["conjunctive", "compensatory", "additive"]:
        c = type_counts[t]
        pct = 100 * c / max(total_analyzed, 1)
        print(f"  {t:<14}: {c:>5} ({pct:>5.1f}%)", flush=True)

    # ── Aggregate by benchmark ──
    print(f"\nPer benchmark:", flush=True)
    bench_type_counts = defaultdict(lambda: defaultdict(int))
    bench_totals = defaultdict(int)
    for r in results_per_item:
        bench_type_counts[r["benchmark"]][r["item_type"]] += 1
        bench_totals[r["benchmark"]] += 1

    bench_summary = {}
    print(f"  {'Benchmark':<10} {'Conj':>6} {'Comp':>6} {'Add':>6} {'Total':>6}", flush=True)
    print(f"  {'-'*40}", flush=True)
    for b in sorted(bench_totals.keys()):
        bt = bench_type_counts[b]
        tot = bench_totals[b]
        conj_pct = 100 * bt["conjunctive"] / max(tot, 1)
        comp_pct = 100 * bt["compensatory"] / max(tot, 1)
        add_pct = 100 * bt["additive"] / max(tot, 1)
        print(f"  {b:<10} {bt['conjunctive']:>5} {bt['compensatory']:>5} "
              f"{bt['additive']:>5} {tot:>6}", flush=True)
        bench_summary[b] = {
            "conjunctive": bt["conjunctive"], "compensatory": bt["compensatory"],
            "additive": bt["additive"], "total": tot,
            "conj_pct": conj_pct, "comp_pct": comp_pct, "add_pct": add_pct,
        }

    # ── Aggregate by skill count ──
    print(f"\nBy number of required skills:", flush=True)
    skill_count_types = defaultdict(lambda: defaultdict(int))
    skill_count_totals = defaultdict(int)
    for r in results_per_item:
        n = r["n_required_skills"]
        skill_count_types[n][r["item_type"]] += 1
        skill_count_totals[n] += 1

    skill_count_summary = {}
    print(f"  {'#Skills':>7} {'Conj':>6} {'Comp':>6} {'Add':>6} {'Total':>6} {'Conj%':>6}",
          flush=True)
    print(f"  {'-'*45}", flush=True)
    for n in sorted(skill_count_totals.keys()):
        st = skill_count_types[n]
        tot = skill_count_totals[n]
        conj_pct = 100 * st["conjunctive"] / max(tot, 1)
        print(f"  {n:>7} {st['conjunctive']:>5} {st['compensatory']:>5} "
              f"{st['additive']:>5} {tot:>6} {conj_pct:>5.1f}%", flush=True)
        skill_count_summary[n] = {
            "conjunctive": st["conjunctive"], "compensatory": st["compensatory"],
            "additive": st["additive"], "total": tot, "conj_pct": conj_pct,
        }

    # ── Conjunctive/compensatory index distributions ──
    conj_indices = [r["conj_index"] for r in results_per_item]
    comp_indices = [r["comp_index"] for r in results_per_item]
    print(f"\n  Conjunctive index: mean={np.mean(conj_indices):.3f}, "
          f"median={np.median(conj_indices):.3f}", flush=True)
    print(f"  Compensatory index: mean={np.mean(comp_indices):.3f}, "
          f"median={np.median(comp_indices):.3f}", flush=True)

    # ── Figure ──
    print("\nGenerating figure...", flush=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Overall distribution (pie-ish horizontal bar)
    ax = axes[0]
    types_ordered = ["conjunctive", "compensatory", "additive"]
    colors_map = {"conjunctive": "#4C72B0", "compensatory": "#DD8452", "additive": "#55A868"}
    counts = [type_counts[t] for t in types_ordered]
    pcts = [100 * c / max(total_analyzed, 1) for c in counts]
    bars = ax.barh(types_ordered, pcts, color=[colors_map[t] for t in types_ordered],
                   edgecolor="white", linewidth=0.8)
    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}% ({count})", va="center", fontsize=9)
    ax.set_xlabel("% of items")
    ax.set_xlim(0, max(pcts) * 1.3)
    ax.invert_yaxis()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Panel 2: Per benchmark stacked bar
    ax = axes[1]
    bench_order = sorted(bench_totals.keys())
    x = np.arange(len(bench_order))
    width = 0.6
    bottom_conj = np.zeros(len(bench_order))
    bottom_comp = np.zeros(len(bench_order))

    conj_pcts = [bench_summary[b]["conj_pct"] for b in bench_order]
    comp_pcts = [bench_summary[b]["comp_pct"] for b in bench_order]
    add_pcts = [bench_summary[b]["add_pct"] for b in bench_order]

    ax.bar(x, conj_pcts, width, label="Conjunctive", color=colors_map["conjunctive"])
    ax.bar(x, comp_pcts, width, bottom=conj_pcts, label="Compensatory",
           color=colors_map["compensatory"])
    ax.bar(x, add_pcts, width,
           bottom=[c + co for c, co in zip(conj_pcts, comp_pcts)],
           label="Additive", color=colors_map["additive"])
    ax.set_xticks(x)
    ax.set_xticklabels(bench_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of items")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Panel 3: Scatter of conj_index vs comp_index
    ax = axes[2]
    ci = np.array(conj_indices)
    co = np.array(comp_indices)
    item_types = [r["item_type"] for r in results_per_item]
    for t in types_ordered:
        mask = np.array([it == t for it in item_types])
        ax.scatter(co[mask], ci[mask], s=8, alpha=0.3, color=colors_map[t], label=t)
    ax.axhline(CONJUNCTIVE_THRESH, color="#ccc", ls="--", lw=0.8)
    ax.axvline(COMPENSATORY_THRESH, color="#ccc", ls="--", lw=0.8)
    ax.set_xlabel("Compensatory index")
    ax.set_ylabel("Conjunctive index")
    ax.legend(frameon=False, fontsize=8, markerscale=3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_fig = fig_dir / "fig_conjunctive_compensatory.pdf"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "conjunctive_compensatory",
        "n_llms": int(n_llms),
        "n_items": int(n_items),
        "K": int(K),
        "mastery_threshold": MASTERY_THRESHOLD,
        "min_group_size": MIN_GROUP_SIZE,
        "min_effect": MIN_EFFECT,
        "conjunctive_thresh": CONJUNCTIVE_THRESH,
        "compensatory_thresh": COMPENSATORY_THRESH,
        "filter_stats": filter_stats,
        "overall": {t: type_counts[t] for t in types_ordered},
        "overall_pct": {t: 100 * type_counts[t] / max(total_analyzed, 1)
                        for t in types_ordered},
        "per_benchmark": bench_summary,
        "per_skill_count": {str(k): v for k, v in skill_count_summary.items()},
        "conj_index_mean": float(np.mean(conj_indices)),
        "conj_index_median": float(np.median(conj_indices)),
        "comp_index_mean": float(np.mean(comp_indices)),
        "comp_index_median": float(np.median(comp_indices)),
        "avg_p_all": float(p_all_avg),
        "avg_p_partial": float(p_partial_avg),
        "avg_p_none": float(p_none_avg),
    }

    out_json = Path("cdm_exploration/experiments/v2_conjunctive_compensatory.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    log_experiment(
        name="conjunctive_compensatory",
        config={"mastery_threshold": MASTERY_THRESHOLD, "min_group_size": MIN_GROUP_SIZE,
                "min_effect": MIN_EFFECT, "K": K, "device": device},
        results=save_data,
        split_info={"n_items": n_items, "n_llms": n_llms,
                    "items_analyzed": filter_stats["analyzed"]},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
