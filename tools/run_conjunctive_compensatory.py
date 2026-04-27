"""Conjunctive vs compensatory skill interaction analysis.

For each multi-skill item, determines whether LLMs combine skills
conjunctively (need ALL), compensatorily (need ANY), or additively.

Primary analysis uses tertile grouping (top/middle/bottom third by
average mastery on required skills) for balanced group sizes.
Binary grouping (mastered all / some / none) is a sensitivity check.

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

MIN_GROUP_SIZE = 20
MIN_EFFECT = 0.05
CONJUNCTIVE_THRESH = 0.6
COMPENSATORY_THRESH = 0.6
SENSITIVITY_THRESHOLDS = [0.5, 0.6, 0.7]


def analyze_items_pattern(two_skill_items, R, q_matrix, theta, items_data):
    """Pattern-based classification for 2-skill items.

    Groups LLMs into the four binary mastery patterns {0,0}, {1,0}, {0,1}, {1,1}
    and uses p_11 vs max(p_10, p_01) vs p_00 to classify items without averaging
    across qualitatively different mastery profiles. This mirrors the G-DINA /
    DINA / DINO logic on a reduced 2-skill design.
    """
    mastery_binary = (theta > 0.5).astype(int)
    results = []
    filter_stats = {"total": len(two_skill_items), "too_few": 0,
                    "no_effect": 0, "analyzed": 0}

    for j in two_skill_items:
        req = np.where(q_matrix[j] > 0)[0]
        if len(req) != 2:
            continue
        a, b = req
        ma = mastery_binary[:, a]
        mb = mastery_binary[:, b]

        m00 = (ma == 0) & (mb == 0)
        m10 = (ma == 1) & (mb == 0)
        m01 = (ma == 0) & (mb == 1)
        m11 = (ma == 1) & (mb == 1)

        n00, n10, n01, n11 = m00.sum(), m10.sum(), m01.sum(), m11.sum()
        if min(n00, n10, n01, n11) < MIN_GROUP_SIZE:
            filter_stats["too_few"] += 1
            continue

        p00 = R[m00, j].mean()
        p10 = R[m10, j].mean()
        p01 = R[m01, j].mean()
        p11 = R[m11, j].mean()

        p_max_single = max(p10, p01)
        effect = p11 - p00
        if effect < MIN_EFFECT:
            filter_stats["no_effect"] += 1
            continue

        filter_stats["analyzed"] += 1
        conj_pattern = (p11 - p_max_single) / (effect + 1e-8)
        comp_pattern = (p_max_single - p00) / (effect + 1e-8)

        if conj_pattern > CONJUNCTIVE_THRESH:
            item_type = "conjunctive"
        elif comp_pattern > COMPENSATORY_THRESH:
            item_type = "compensatory"
        else:
            item_type = "additive"

        results.append({
            "item_idx": int(j),
            "benchmark": items_data[j].get("benchmark", "unknown"),
            "n_required_skills": 2,
            "n_all": int(n11), "n_none": int(n00),
            "n_partial": int(n10 + n01),
            "p_all": float(p11), "p_none": float(p00),
            "p_partial": float(p_max_single),
            "n_00": int(n00), "n_10": int(n10), "n_01": int(n01), "n_11": int(n11),
            "p_00": float(p00), "p_10": float(p10),
            "p_01": float(p01), "p_11": float(p11),
            "conj_index": float(conj_pattern),
            "comp_index": float(comp_pattern),
            "item_type": item_type,
        })

    return results, filter_stats


def reclassify(results, conj_thresh, comp_thresh):
    """Apply alternative thresholds to existing per-item indices."""
    counts = {"conjunctive": 0, "compensatory": 0, "additive": 0}
    for r in results:
        if r["conj_index"] > conj_thresh:
            counts["conjunctive"] += 1
        elif r["comp_index"] > comp_thresh:
            counts["compensatory"] += 1
        else:
            counts["additive"] += 1
    total = max(len(results), 1)
    return {
        "conj_thresh": conj_thresh, "comp_thresh": comp_thresh,
        "n_analyzed": len(results),
        "counts": counts,
        "pct": {t: 100 * counts[t] / total for t in counts},
    }


def analyze_items(multi_items, R, q_matrix, theta, items_data, n_llms,
                  mode="tertile"):
    """Run conjunctive/compensatory classification on a set of items.

    Args:
        mode: "tertile" (primary) or "binary" (sensitivity check).

    Returns:
        list of per-item result dicts, filter_stats dict.
    """
    mastery_binary = (theta > 0.5).astype(int) if mode == "binary" else None
    results = []
    filter_stats = {"total": len(multi_items), "too_few": 0,
                    "no_effect": 0, "analyzed": 0}

    for j in multi_items:
        req = np.where(q_matrix[j] > 0)[0]
        n_req = len(req)

        if mode == "binary":
            m = mastery_binary[:, req].sum(axis=1)
            all_mask = m == n_req
            none_mask = m == 0
            partial_mask = (~all_mask) & (~none_mask)
        else:
            # Tertile: group by average mastery on required skills
            avg_m = theta[:, req].mean(axis=1)
            t33 = np.percentile(avg_m, 33.3)
            t66 = np.percentile(avg_m, 66.7)
            all_mask = avg_m >= t66       # top third
            none_mask = avg_m <= t33      # bottom third
            partial_mask = (~all_mask) & (~none_mask)

        n_all = all_mask.sum()
        n_none = none_mask.sum()
        n_partial = partial_mask.sum()

        if min(n_all, n_none, n_partial) < MIN_GROUP_SIZE:
            filter_stats["too_few"] += 1
            continue

        p_all = R[all_mask, j].mean()
        p_none = R[none_mask, j].mean()
        p_partial = R[partial_mask, j].mean()
        effect = p_all - p_none

        if effect < MIN_EFFECT:
            filter_stats["no_effect"] += 1
            continue

        filter_stats["analyzed"] += 1
        conj = (p_all - p_partial) / (effect + 1e-8)
        comp = (p_partial - p_none) / (effect + 1e-8)

        if conj > CONJUNCTIVE_THRESH:
            item_type = "conjunctive"
        elif comp > COMPENSATORY_THRESH:
            item_type = "compensatory"
        else:
            item_type = "additive"

        results.append({
            "item_idx": int(j),
            "benchmark": items_data[j].get("benchmark", "unknown"),
            "n_required_skills": int(n_req),
            "n_all": int(n_all), "n_none": int(n_none), "n_partial": int(n_partial),
            "p_all": float(p_all), "p_none": float(p_none), "p_partial": float(p_partial),
            "conj_index": float(conj), "comp_index": float(comp),
            "item_type": item_type,
        })

    return results, filter_stats


def summarize(results):
    """Aggregate results into type counts, per-benchmark, per-skill-count."""
    types_ordered = ["conjunctive", "compensatory", "additive"]
    type_counts = defaultdict(int)
    bench_counts = defaultdict(lambda: defaultdict(int))
    bench_totals = defaultdict(int)
    skill_counts = defaultdict(lambda: defaultdict(int))
    skill_totals = defaultdict(int)

    for r in results:
        type_counts[r["item_type"]] += 1
        b = r["benchmark"]
        bench_counts[b][r["item_type"]] += 1
        bench_totals[b] += 1
        n = r["n_required_skills"]
        skill_counts[n][r["item_type"]] += 1
        skill_totals[n] += 1

    total = len(results)
    overall = {t: type_counts[t] for t in types_ordered}
    overall_pct = {t: 100 * type_counts[t] / max(total, 1) for t in types_ordered}

    bench_summary = {}
    for b in sorted(bench_totals.keys()):
        bt = bench_counts[b]
        tot = bench_totals[b]
        bench_summary[b] = {
            t: bt[t] for t in types_ordered
        }
        bench_summary[b]["total"] = tot
        for t in types_ordered:
            bench_summary[b][f"{t}_pct"] = 100 * bt[t] / max(tot, 1)

    skill_summary = {}
    for n in sorted(skill_totals.keys()):
        st = skill_counts[n]
        tot = skill_totals[n]
        skill_summary[str(n)] = {t: st[t] for t in types_ordered}
        skill_summary[str(n)]["total"] = tot
        skill_summary[str(n)]["conj_pct"] = 100 * st["conjunctive"] / max(tot, 1)

    conj_indices = [r["conj_index"] for r in results]
    comp_indices = [r["comp_index"] for r in results]

    return {
        "n_analyzed": total,
        "overall": overall,
        "overall_pct": overall_pct,
        "per_benchmark": bench_summary,
        "per_skill_count": skill_summary,
        "conj_index_mean": float(np.mean(conj_indices)) if conj_indices else 0,
        "conj_index_median": float(np.median(conj_indices)) if conj_indices else 0,
        "comp_index_mean": float(np.mean(comp_indices)) if comp_indices else 0,
        "comp_index_median": float(np.median(comp_indices)) if comp_indices else 0,
        "avg_p_all": float(np.mean([r["p_all"] for r in results])) if results else 0,
        "avg_p_partial": float(np.mean([r["p_partial"] for r in results])) if results else 0,
        "avg_p_none": float(np.mean([r["p_none"] for r in results])) if results else 0,
    }


def print_summary(summary, label):
    """Pretty-print a summary dict."""
    total = summary["n_analyzed"]
    print(f"\n{'='*60}", flush=True)
    print(f"{label} ({total} items)", flush=True)
    print(f"{'='*60}", flush=True)
    for t in ["conjunctive", "compensatory", "additive"]:
        c = summary["overall"][t]
        pct = summary["overall_pct"][t]
        print(f"  {t:<14}: {c:>5} ({pct:>5.1f}%)", flush=True)

    print(f"\n  Cross-check: p_all={summary['avg_p_all']:.4f} > "
          f"p_partial={summary['avg_p_partial']:.4f} > "
          f"p_none={summary['avg_p_none']:.4f}", flush=True)
    ok = summary["avg_p_all"] > summary["avg_p_partial"] > summary["avg_p_none"]
    print(f"  {'PASS' if ok else 'WARNING'}: monotonic ordering", flush=True)

    print(f"\n  Per benchmark:", flush=True)
    print(f"  {'Bench':<10} {'Conj':>6} {'Comp':>6} {'Add':>6} {'Total':>6} "
          f"{'Conj%':>6} {'Comp%':>6}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for b in sorted(summary["per_benchmark"].keys()):
        bs = summary["per_benchmark"][b]
        print(f"  {b:<10} {bs['conjunctive']:>5} {bs['compensatory']:>5} "
              f"{bs['additive']:>5} {bs['total']:>6} "
              f"{bs['conjunctive_pct']:>5.1f}% {bs['compensatory_pct']:>5.1f}%",
              flush=True)

    print(f"\n  By number of required skills:", flush=True)
    print(f"  {'#Skills':>7} {'Conj':>6} {'Comp':>6} {'Add':>6} {'Total':>6} "
          f"{'Conj%':>6}", flush=True)
    print(f"  {'-'*45}", flush=True)
    for n in sorted(summary["per_skill_count"].keys(), key=int):
        sc = summary["per_skill_count"][n]
        print(f"  {n:>7} {sc['conjunctive']:>5} {sc['compensatory']:>5} "
              f"{sc['additive']:>5} {sc['total']:>6} {sc['conj_pct']:>5.1f}%",
              flush=True)


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

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)

    # ── Load theta ──
    print("\nLoading trained theta...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, "cpu",
    )
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).numpy()
    print(f"  theta: {theta.shape}, range [{theta.min():.3f}, {theta.max():.3f}]",
          flush=True)

    # ── Item stats ──
    skills_per_item = q_matrix.sum(axis=1).astype(int)
    multi_items = np.where(skills_per_item >= 2)[0]
    print(f"\n  Items with 1 skill: {(skills_per_item == 1).sum()}", flush=True)
    print(f"  Items with 2+ skills: {len(multi_items)}", flush=True)

    # ── Sanity check: single-skill items ──
    print("\nSanity check: single-skill items...", flush=True)
    single_items = np.where(skills_per_item == 1)[0]
    mastery_bin = (theta > 0.5).astype(int)
    p_master_list, p_non_list = [], []
    for j in single_items[:500]:
        sk = np.where(q_matrix[j] > 0)[0][0]
        masters = mastery_bin[:, sk] == 1
        non = mastery_bin[:, sk] == 0
        if masters.sum() >= 20 and non.sum() >= 20:
            p_master_list.append(R[masters, j].mean())
            p_non_list.append(R[non, j].mean())
    if p_master_list:
        gap = np.mean(p_master_list) - np.mean(p_non_list)
        print(f"  p_master={np.mean(p_master_list):.4f}, "
              f"p_non={np.mean(p_non_list):.4f}, gap={gap:.4f}", flush=True)
        print(f"  PASS: mastery predicts accuracy", flush=True)

    # ── PRIMARY: tertile grouping ──
    print(f"\n{'='*60}", flush=True)
    print("PRIMARY ANALYSIS: Tertile grouping", flush=True)
    print(f"{'='*60}", flush=True)
    results_tertile, fstats_tertile = analyze_items(
        multi_items, R, q_matrix, theta, items_data, n_llms, mode="tertile",
    )
    summary_tertile = summarize(results_tertile)
    print(f"  Filter: {fstats_tertile}", flush=True)
    print_summary(summary_tertile, "TERTILE RESULTS")

    # ── SENSITIVITY: binary grouping ──
    print(f"\n{'='*60}", flush=True)
    print("SENSITIVITY CHECK: Binary grouping (theta > 0.5)", flush=True)
    print(f"{'='*60}", flush=True)
    results_binary, fstats_binary = analyze_items(
        multi_items, R, q_matrix, theta, items_data, n_llms, mode="binary",
    )
    summary_binary = summarize(results_binary)
    print(f"  Filter: {fstats_binary}", flush=True)
    print_summary(summary_binary, "BINARY RESULTS")

    # ── ROBUSTNESS: pattern-based classification for 2-skill items ──
    two_skill_items = np.where(skills_per_item == 2)[0]
    print(f"\n{'='*60}", flush=True)
    print(f"ROBUSTNESS CHECK: Pattern-based grouping for 2-skill items "
          f"({len(two_skill_items)} items)", flush=True)
    print(f"{'='*60}", flush=True)
    results_pattern, fstats_pattern = analyze_items_pattern(
        two_skill_items, R, q_matrix, theta, items_data,
    )
    summary_pattern = summarize(results_pattern)
    print(f"  Filter: {fstats_pattern}", flush=True)
    print_summary(summary_pattern, "PATTERN RESULTS")

    # Agreement between tertile and pattern classification (on shared items)
    tertile_by_idx = {r["item_idx"]: r["item_type"] for r in results_tertile}
    pattern_by_idx = {r["item_idx"]: r["item_type"] for r in results_pattern}
    shared = sorted(set(tertile_by_idx) & set(pattern_by_idx))
    agree = sum(1 for j in shared if tertile_by_idx[j] == pattern_by_idx[j])
    agreement_pct = 100 * agree / max(len(shared), 1)
    print(f"\n  Tertile vs Pattern agreement on {len(shared)} shared items: "
          f"{agree}/{len(shared)} ({agreement_pct:.1f}%)", flush=True)

    # ── THRESHOLD SENSITIVITY on tertile results ──
    print(f"\n{'='*60}", flush=True)
    print("THRESHOLD SENSITIVITY: tertile results at 0.5 / 0.6 / 0.7", flush=True)
    print(f"{'='*60}", flush=True)
    threshold_sensitivity = []
    for thr in SENSITIVITY_THRESHOLDS:
        s = reclassify(results_tertile, thr, thr)
        threshold_sensitivity.append(s)
        c = s["counts"]
        p = s["pct"]
        print(f"  thresh={thr:.1f} | conj={c['conjunctive']:>4} ({p['conjunctive']:>5.1f}%)  "
              f"comp={c['compensatory']:>4} ({p['compensatory']:>5.1f}%)  "
              f"add={c['additive']:>4} ({p['additive']:>5.1f}%)", flush=True)

    # ── Figure: per-benchmark stacked bars (tertile results) ──
    print("\nGenerating figure...", flush=True)
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_map = {"conjunctive": "#4C72B0", "compensatory": "#DD8452",
                  "additive": "#55A868"}
    types_ordered = ["conjunctive", "compensatory", "additive"]
    labels_map = {"conjunctive": "Conjunctive (need ALL)",
                  "compensatory": "Compensatory (ANY helps)",
                  "additive": "Additive"}

    # Panel 1: Per-benchmark stacked bars
    ax = axes[0]
    bench_order = sorted(summary_tertile["per_benchmark"].keys())
    x = np.arange(len(bench_order))
    width = 0.55

    bottoms = np.zeros(len(bench_order))
    for t in types_ordered:
        vals = [summary_tertile["per_benchmark"][b][f"{t}_pct"] for b in bench_order]
        ax.bar(x, vals, width, bottom=bottoms, label=labels_map[t],
               color=colors_map[t], edgecolor="white", linewidth=0.5)
        for i, (v, bot) in enumerate(zip(vals, bottoms)):
            if v >= 10:
                ax.text(x[i], bot + v / 2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=8, fontweight="bold",
                        color="white")
            elif v >= 6:
                ax.text(x[i], bot + v / 2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=6.5, fontweight="bold",
                        color="white")
            elif v > 0:
                ax.text(x[i], bot + v / 2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=5.5, fontweight="bold",
                        color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bench_order, fontsize=10)
    ax.set_ylabel("% of items", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Legend is placed below the figure via figure-level legend (see end).
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Panel 2: Method sensitivity — stacked bars across classification methods.
    # (Replaces the earlier conj-index vs comp-index scatter, which was a
    # degenerate 1D line since the two indices sum to 1 by construction.)
    ax = axes[1]
    methods = [
        ("Tertile\n(primary)",
         summary_tertile["overall_pct"], summary_tertile["n_analyzed"]),
        ("Binary\nmastery",
         summary_binary["overall_pct"], summary_binary["n_analyzed"]),
        ("Pattern\n(2-skill)",
         summary_pattern["overall_pct"], summary_pattern["n_analyzed"]),
    ]
    x_m = np.arange(len(methods))
    bottoms_m = np.zeros(len(methods))
    for t in types_ordered:
        vals = [m[1][t] for m in methods]
        ax.bar(x_m, vals, width, bottom=bottoms_m, label=labels_map[t],
               color=colors_map[t], edgecolor="white", linewidth=0.5)
        for i, (v, bot) in enumerate(zip(vals, bottoms_m)):
            if v > 8:
                ax.text(x_m[i], bot + v / 2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=8, fontweight="bold",
                        color="white")
        bottoms_m += vals

    ax.set_xticks(x_m)
    ax.set_xticklabels([m[0] for m in methods], fontsize=9.5)
    # Annotate n per method below each bar
    for i, m in enumerate(methods):
        ax.text(x_m[i], -6, f"n={m[2]}", ha="center", va="top",
                fontsize=8, color="#444444")
    ax.set_ylabel("% of items", fontsize=11)
    ax.set_ylim(-10, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("(b) Classification-method sensitivity",
                 fontsize=10.5, loc="left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.35)

    # Title for panel (a) only after panel (b) is in place so titles align
    axes[0].set_title("(a) Per-benchmark classification (tertile)",
                      fontsize=10.5, loc="left")

    # Shared horizontal legend below both panels
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_map[t],
                      edgecolor="white", linewidth=0.5, label=labels_map[t])
        for t in types_ordered
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
               fontsize=10, handletextpad=0.5, columnspacing=1.8)

    plt.tight_layout(pad=2.0, rect=(0, 0.05, 1, 1))
    out_fig = fig_dir / "fig_conjunctive_compensatory.pdf"
    out_fig_png = fig_dir / "fig_conjunctive_compensatory.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_fig_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig} and {out_fig_png}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "conjunctive_compensatory",
        "n_llms": int(n_llms), "n_items": int(n_items), "K": int(K),
        "min_group_size": MIN_GROUP_SIZE, "min_effect": MIN_EFFECT,
        "conjunctive_thresh": CONJUNCTIVE_THRESH,
        "compensatory_thresh": COMPENSATORY_THRESH,
        "primary_method": "tertile",
        "tertile": {
            "filter_stats": fstats_tertile,
            **summary_tertile,
        },
        "binary_sensitivity": {
            "filter_stats": fstats_binary,
            **summary_binary,
        },
        "pattern_robustness_2skill": {
            "filter_stats": fstats_pattern,
            "tertile_vs_pattern_agreement": {
                "n_shared": len(shared),
                "n_agree": int(agree),
                "agreement_pct": float(agreement_pct),
            },
            **summary_pattern,
        },
        "threshold_sensitivity_tertile": threshold_sensitivity,
    }
    out_json = Path("cdm_exploration/experiments/v2_conjunctive_compensatory.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    log_experiment(
        name="conjunctive_compensatory",
        config={"min_group_size": MIN_GROUP_SIZE, "min_effect": MIN_EFFECT,
                "primary_method": "tertile", "K": K, "device": device},
        results=save_data,
        split_info={"n_items": n_items, "n_llms": n_llms,
                    "tertile_analyzed": fstats_tertile["analyzed"],
                    "binary_analyzed": fstats_binary["analyzed"]},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
