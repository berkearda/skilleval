"""Benchmark redundancy mapping: skill overlap across benchmarks.

Analyzes which skills each benchmark tests, how much they overlap,
and which benchmarks are redundant vs unique. Produces UpSet plot,
heatmap, and redundancy matrix.

Usage:
    python tools/run_benchmark_redundancy.py device=cpu
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.device import seed_everything
    from cdmeval.utils.experiment import log_experiment
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading data...", flush=True)
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        skill_labels = json.load(f)

    n_items, K = q_matrix.shape
    n_llms = R.shape[0]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)
    skill_names = [skill_labels[str(i)] for i in range(K)]

    # ── Per-benchmark item indices ──
    bench_items = {b: [] for b in BENCHMARKS}
    unlabelled = 0
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in bench_items:
            bench_items[b].append(i)
        else:
            unlabelled += 1

    print(f"\n  Items per benchmark:", flush=True)
    for b in BENCHMARKS:
        print(f"    {b:<8}: {len(bench_items[b])}", flush=True)
    if unlabelled:
        print(f"    (unlabelled: {unlabelled})", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 1. SKILL COVERAGE PER BENCHMARK
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("1. SKILL COVERAGE PER BENCHMARK", flush=True)
    print(f"{'='*60}", flush=True)

    # Binary: skill k covered if any item has q_jk=1
    binary_coverage = {}  # {bench: set of skill indices}
    # Weighted: fraction of items in bench that require skill k
    weighted_coverage = {}  # {bench: (K,) array of fractions}
    # Item counts per (bench, skill)
    item_counts = {}  # {bench: (K,) array of counts}

    for b in BENCHMARKS:
        idx = np.array(bench_items[b], dtype=int)
        Q_b = q_matrix[idx]  # (n_b, K)
        binary_coverage[b] = set(np.where(Q_b.sum(axis=0) > 0)[0])
        item_counts[b] = Q_b.sum(axis=0).astype(int)  # (K,)
        weighted_coverage[b] = Q_b.mean(axis=0)  # (K,) fraction

    for b in BENCHMARKS:
        print(f"  {b:<8}: covers {len(binary_coverage[b]):>3}/100 skills, "
              f"mean items/skill={item_counts[b][item_counts[b]>0].mean():.1f}",
              flush=True)

    # Sanity: total unique skills
    all_covered = set()
    for b in BENCHMARKS:
        all_covered |= binary_coverage[b]
    print(f"\n  Total skills covered by any benchmark: {len(all_covered)}/100", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 2. PAIRWISE OVERLAP
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("2. PAIRWISE OVERLAP", flush=True)
    print(f"{'='*60}", flush=True)

    jaccard_matrix = np.zeros((5, 5))
    coverage_matrix = np.zeros((5, 5))  # (i,j) = % of i's skills covered by j
    pairwise_details = {}

    for i, bi in enumerate(BENCHMARKS):
        for j, bj in enumerate(BENCHMARKS):
            si, sj = binary_coverage[bi], binary_coverage[bj]
            inter = si & sj
            union = si | sj
            jaccard_matrix[i, j] = len(inter) / max(len(union), 1)
            coverage_matrix[i, j] = 100 * len(inter) / max(len(si), 1)

            if i < j:
                unique_i = si - sj
                unique_j = sj - si
                pairwise_details[f"{bi}-{bj}"] = {
                    "jaccard": float(jaccard_matrix[i, j]),
                    "intersection": len(inter),
                    "union": len(union),
                    "unique_to_first": len(unique_i),
                    "unique_to_second": len(unique_j),
                }

    # Print Jaccard matrix
    print(f"\n  Jaccard similarity:", flush=True)
    header = "          " + "".join(f"{b:>8}" for b in BENCHMARKS)
    print(header, flush=True)
    for i, b in enumerate(BENCHMARKS):
        row = f"  {b:<8}" + "".join(f"{jaccard_matrix[i,j]:>8.3f}" for j in range(5))
        print(row, flush=True)

    # Verify symmetry
    asym = np.abs(jaccard_matrix - jaccard_matrix.T).max()
    print(f"\n  Symmetry check: max |J(A,B)-J(B,A)| = {asym:.2e} "
          f"({'PASS' if asym < 1e-10 else 'FAIL'})", flush=True)

    # Print coverage matrix
    print(f"\n  Coverage matrix (row i, col j = % of i's skills covered by j):", flush=True)
    print(header, flush=True)
    for i, b in enumerate(BENCHMARKS):
        row = f"  {b:<8}" + "".join(f"{coverage_matrix[i,j]:>7.1f}%" for j in range(5))
        print(row, flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. UNIQUE SKILLS PER BENCHMARK
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("3. UNIQUE SKILLS PER BENCHMARK", flush=True)
    print(f"{'='*60}", flush=True)

    unique_skills = {}
    for b in BENCHMARKS:
        others = set()
        for b2 in BENCHMARKS:
            if b2 != b:
                others |= binary_coverage[b2]
        unique = binary_coverage[b] - others
        unique_skills[b] = sorted(unique)
        names = [skill_names[k] for k in unique_skills[b]]
        print(f"  {b:<8}: {len(unique)} unique skills", flush=True)
        for n in names[:5]:
            print(f"    - {n}", flush=True)
        if len(names) > 5:
            print(f"    ... and {len(names)-5} more", flush=True)

    total_unique = sum(len(v) for v in unique_skills.values())
    print(f"\n  Total unique-to-one-benchmark: {total_unique}/100", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. MULTI-WAY INTERSECTIONS (for UpSet plot)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("4. MULTI-WAY INTERSECTIONS", flush=True)
    print(f"{'='*60}", flush=True)

    # For each of the 2^5-1 non-empty subsets, find skills in exactly that subset
    intersections = []  # (frozenset of benchmarks, count, skill indices)
    for r in range(1, 6):
        for combo in combinations(range(5), r):
            benches_in = frozenset(combo)
            benches_out = frozenset(range(5)) - benches_in
            # Skills that are in ALL benches_in and NONE of benches_out
            skills_in = set(range(K))
            for idx in benches_in:
                skills_in &= binary_coverage[BENCHMARKS[idx]]
            for idx in benches_out:
                skills_in -= binary_coverage[BENCHMARKS[idx]]
            if skills_in:
                bench_names = tuple(BENCHMARKS[i] for i in sorted(benches_in))
                intersections.append((bench_names, len(skills_in), sorted(skills_in)))

    # Sort by count descending
    intersections.sort(key=lambda x: -x[1])
    print(f"  Non-empty intersections: {len(intersections)}/31", flush=True)
    for names, count, _ in intersections[:15]:
        label = " ∩ ".join(names)
        print(f"    {label:<40}: {count} skills", flush=True)
    if len(intersections) > 15:
        print(f"    ... ({len(intersections)-15} more)", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 5. OPTIMAL BENCHMARK SUBSET
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("5. OPTIMAL BENCHMARK SUBSET (greedy)", flush=True)
    print(f"{'='*60}", flush=True)

    optimal_subsets = {}
    remaining_benches = list(range(5))
    selected_benches = []
    covered = set()

    for size in range(1, 6):
        best_idx = -1
        best_gain = -1
        for idx in remaining_benches:
            gain = len(binary_coverage[BENCHMARKS[idx]] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        selected_benches.append(best_idx)
        covered |= binary_coverage[BENCHMARKS[best_idx]]
        remaining_benches.remove(best_idx)
        names = [BENCHMARKS[i] for i in selected_benches]
        optimal_subsets[size] = {
            "benchmarks": names,
            "coverage": len(covered),
            "coverage_pct": 100 * len(covered) / K,
        }
        print(f"  Best {size}: {' + '.join(names)} → {len(covered)}/100 skills "
              f"({100*len(covered)/K:.0f}%)", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 6. WEIGHTED REDUNDANCY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("6. WEIGHTED REDUNDANCY (by item count)", flush=True)
    print(f"{'='*60}", flush=True)

    weighted_jaccard = np.zeros((5, 5))
    for i, bi in enumerate(BENCHMARKS):
        for j, bj in enumerate(BENCHMARKS):
            ci = item_counts[bi].astype(float)
            cj = item_counts[bj].astype(float)
            min_counts = np.minimum(ci, cj)
            max_counts = np.maximum(ci, cj)
            denom = max_counts.sum()
            weighted_jaccard[i, j] = min_counts.sum() / max(denom, 1)

    print(f"  Weighted Jaccard (by item count):", flush=True)
    print(header, flush=True)
    for i, b in enumerate(BENCHMARKS):
        row = f"  {b:<8}" + "".join(f"{weighted_jaccard[i,j]:>8.3f}" for j in range(5))
        print(row, flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Figure 1: UpSet-style plot ──
    fig, (ax_bars, ax_dots) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 2]},
        sharex=True,
    )

    # Sort intersections by count
    top_n = min(20, len(intersections))
    top_ints = intersections[:top_n]

    x = np.arange(top_n)
    counts = [t[1] for t in top_ints]
    ax_bars.bar(x, counts, color="#4C72B0", edgecolor="white", linewidth=0.5)
    for xi, c in zip(x, counts):
        ax_bars.text(xi, c + 0.3, str(c), ha="center", fontsize=8, fontweight="bold")
    ax_bars.set_ylabel("# skills", fontsize=11)
    for sp in ["top", "right"]:
        ax_bars.spines[sp].set_visible(False)

    # Dot matrix
    bench_to_row = {b: i for i, b in enumerate(BENCHMARKS)}
    for xi, (bench_names, _, _) in enumerate(top_ints):
        for b in BENCHMARKS:
            if b in bench_names:
                ax_dots.scatter(xi, bench_to_row[b], s=60, color="#4C72B0", zorder=3)
            else:
                ax_dots.scatter(xi, bench_to_row[b], s=30, color="#DDDDDD", zorder=2)
        # Connect dots vertically
        active = [bench_to_row[b] for b in bench_names]
        if len(active) > 1:
            ax_dots.plot([xi, xi], [min(active), max(active)],
                         color="#4C72B0", lw=2, zorder=2)

    ax_dots.set_yticks(range(5))
    ax_dots.set_yticklabels(BENCHMARKS, fontsize=10)
    ax_dots.set_xlim(-0.5, top_n - 0.5)
    ax_dots.invert_yaxis()
    for sp in ["top", "right", "bottom"]:
        ax_dots.spines[sp].set_visible(False)
    ax_dots.tick_params(bottom=False)

    plt.tight_layout()
    out1 = fig_dir / "fig_benchmark_redundancy_upset.pdf"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}", flush=True)

    # ── Figure 2: Heatmap (benchmarks × skills) ──
    fig, ax = plt.subplots(figsize=(14, 3.5))
    # Sort skills by which benchmark uses them most
    skill_primary = np.zeros(K, dtype=int)
    for k in range(K):
        best_b = max(range(5), key=lambda i: item_counts[BENCHMARKS[i]][k])
        skill_primary[k] = best_b
    skill_order = np.argsort(skill_primary * 10000 -
                              np.array([item_counts[BENCHMARKS[skill_primary[k]]][k]
                                        for k in range(K)]))

    heat_data = np.zeros((5, K))
    for i, b in enumerate(BENCHMARKS):
        heat_data[i] = weighted_coverage[b]
    heat_data = heat_data[:, skill_order]

    im = ax.imshow(heat_data, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(5))
    ax.set_yticklabels(BENCHMARKS, fontsize=10)
    ax.set_xlabel("Skills (sorted by primary benchmark)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Fraction of items", shrink=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out2 = fig_dir / "fig_benchmark_redundancy_heatmap.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Figure 3: Redundancy matrix ──
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(coverage_matrix, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(5))
    ax.set_xticklabels(BENCHMARKS, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(BENCHMARKS, fontsize=10)
    for i in range(5):
        for j in range(5):
            color = "white" if coverage_matrix[i, j] > 60 else "black"
            ax.text(j, i, f"{coverage_matrix[i,j]:.0f}%", ha="center",
                    va="center", fontsize=10, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="% of row's skills covered by column", shrink=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out3 = fig_dir / "fig_benchmark_redundancy_matrix.pdf"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "benchmark_redundancy",
        "n_items": int(n_items), "n_llms": int(n_llms), "K": int(K),
        "benchmarks": BENCHMARKS,
        "items_per_benchmark": {b: len(bench_items[b]) for b in BENCHMARKS},
        "binary_coverage_count": {b: len(binary_coverage[b]) for b in BENCHMARKS},
        "total_skills_covered": len(all_covered),
        "jaccard_matrix": jaccard_matrix.tolist(),
        "coverage_matrix": coverage_matrix.tolist(),
        "weighted_jaccard_matrix": weighted_jaccard.tolist(),
        "pairwise_details": pairwise_details,
        "unique_skills": {b: {"count": len(unique_skills[b]),
                               "indices": unique_skills[b],
                               "names": [skill_names[k] for k in unique_skills[b]]}
                          for b in BENCHMARKS},
        "multi_way_intersections": [
            {"benchmarks": list(names), "n_skills": count, "skill_indices": sidx}
            for names, count, sidx in intersections
        ],
        "optimal_subsets": optimal_subsets,
    }

    out_json = Path("cdm_exploration/experiments/v2_benchmark_redundancy.json")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="benchmark_redundancy",
        config={"K": K, "benchmarks": BENCHMARKS},
        results=save_data,
        split_info={"n_items": n_items, "n_llms": n_llms},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
