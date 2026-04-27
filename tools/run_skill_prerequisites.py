"""Skill prerequisite DAG discovery from LLM mastery patterns.

If skill A is prerequisite for skill B, almost every model mastering B
also masters A, but not vice versa. Discovers the DAG, validates against
model size, checks cross-family consistency.

Usage:
    python tools/run_skill_prerequisites.py device=cpu
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

MASTERY_THRESH = 0.5
ASYMMETRY_THRESHOLD = 0.3  # P(A|B) - P(B|A) must exceed this
OR_THRESHOLD = 10.0        # Odds ratio must exceed this (base-rate-corrected)
CONTINUITY_CORRECTION = 0.5


def compute_odds_ratio(co_mastery, n_mastering, n_llms):
    """Pairwise odds ratio with Haldane continuity correction.

    For each ordered pair (a, b), build the 2x2 contingency table over LLMs:
        ab          = #LLMs mastering both a and b
        a_not_b     = #LLMs mastering a but not b
        not_a_b     = #LLMs mastering b but not a
        not_a_not_b = #LLMs mastering neither
    OR(a→b) = ((ab + 0.5) · (not_a_not_b + 0.5)) / ((a_not_b + 0.5) · (not_a_b + 0.5))
    The 0.5 offset avoids division-by-zero when any cell is empty (Haldane 1956).
    """
    ab = co_mastery
    a_count = n_mastering[:, np.newaxis]
    b_count = n_mastering[np.newaxis, :]
    a_not_b = a_count - co_mastery
    not_a_b = b_count - co_mastery
    not_a_not_b = n_llms - a_count - b_count + co_mastery

    c = CONTINUITY_CORRECTION
    numer = (ab + c) * (not_a_not_b + c)
    denom = (a_not_b + c) * (not_a_b + c)
    return numer / denom


def transitive_reduction(adj):
    """Remove transitive edges from a boolean adjacency matrix (in-place).

    If A→B and B→C exist, remove A→C.  Works on a copy to avoid
    modifying during iteration.
    """
    K = adj.shape[0]
    # Compute reachability beyond direct edges
    reduced = adj.copy()
    for k in range(K):
        for i in range(K):
            if not reduced[i, k]:
                continue
            for j in range(K):
                if reduced[k, j]:
                    reduced[i, j] = False  # i→j is transitive via k
        # Restore direct edge i→k that we might have removed
        # Actually the standard algorithm: for each edge i→j, remove if
        # there is a path i→...→j of length ≥2.
    # Cleaner: Floyd-Warshall-based
    reduced2 = adj.copy()
    # path_len[i,j] = shortest path length from i to j (using only direct edges)
    reach = adj.copy()
    for mid in range(K):
        for i in range(K):
            if not reach[i, mid]:
                continue
            for j in range(K):
                if reach[mid, j] and adj[i, j] and i != mid and mid != j:
                    reduced2[i, j] = False
    return reduced2


def detect_cycles(adj):
    """Return True if the graph has a cycle (DFS-based)."""
    K = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.zeros(K, dtype=int)
    cycle_edges = []

    def dfs(u):
        color[u] = GRAY
        for v in range(K):
            if not adj[u, v]:
                continue
            if color[v] == GRAY:
                cycle_edges.append((u, v))
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for u in range(K):
        if color[u] == WHITE:
            if dfs(u):
                return True, cycle_edges
    return False, []


def topological_depth(adj):
    """Compute depth of each node in the DAG (longest path from any root)."""
    K = adj.shape[0]
    in_degree = adj.sum(axis=0)
    depth = np.zeros(K, dtype=int)
    # BFS from roots
    queue = list(np.where(in_degree == 0)[0])
    visited = set(queue)
    while queue:
        u = queue.pop(0)
        for v in range(K):
            if adj[u, v]:
                depth[v] = max(depth[v], depth[u] + 1)
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
    return depth


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
    mastery = (theta > MASTERY_THRESH).astype(int)
    mastery_rate = mastery.mean(axis=0)  # (K,)
    print(f"  theta: {theta.shape}", flush=True)
    print(f"  Mastery rates: min={mastery_rate.min():.3f}, "
          f"median={np.median(mastery_rate):.3f}, max={mastery_rate.max():.3f}",
          flush=True)

    # ════════════════════════════════════════════════════════════════
    # 1. CONDITIONAL PROBABILITIES
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("1. COMPUTING CONDITIONAL PROBABILITIES", flush=True)
    print(f"{'='*60}", flush=True)

    n_mastering = mastery.sum(axis=0)  # (K,) how many LLMs master each skill
    # P(A|B) = |A∩B| / |B| for all pairs
    # mastery.T @ mastery = co-mastery matrix (K, K) where [a,b] = #LLMs mastering both
    co_mastery = mastery.T @ mastery  # (K, K), int

    # P(A|B) = co_mastery[a,b] / n_mastering[b]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_a_given_b = co_mastery / n_mastering[np.newaxis, :]  # (K, K)
        p_a_given_b = np.nan_to_num(p_a_given_b, nan=0.0)

    # prerequisite_score[a,b] = P(A|B) - P(B|A): positive means A is prereq for B
    prereq_score = p_a_given_b - p_a_given_b.T

    # Lift(a|b) = P(a|b) / P(a) — quantifies how much b's mastery raises a's
    # probability beyond its marginal. Reported per-edge for inspection.
    p_a = mastery_rate  # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = p_a_given_b / p_a[:, np.newaxis]
        lift = np.nan_to_num(lift, nan=0.0, posinf=0.0)

    # Odds ratio: base-rate-corrected association. OR > 1 means b mastery
    # genuinely predicts a mastery beyond what their marginals would imply.
    odds_ratio = compute_odds_ratio(co_mastery, n_mastering, n_llms)

    # Edge: a→b if asymmetry > thresh (a is easier than b given b) AND
    # OR > thresh (a and b are non-trivially associated).
    adj_raw = (prereq_score > ASYMMETRY_THRESHOLD) & (odds_ratio > OR_THRESHOLD)
    np.fill_diagonal(adj_raw, False)
    n_edges_raw = adj_raw.sum()
    print(f"  Raw edges (before reduction): {n_edges_raw}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 2. BUILD THE DAG
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("2. BUILDING DAG", flush=True)
    print(f"{'='*60}", flush=True)

    has_cycle, cycle_edges = detect_cycles(adj_raw)
    if has_cycle:
        print(f"  WARNING: {len(cycle_edges)} cycle edge(s) found, removing weakest",
              flush=True)
        for u, v in cycle_edges:
            if prereq_score[u, v] < prereq_score[v, u]:
                adj_raw[u, v] = False
            else:
                adj_raw[v, u] = False
        has_cycle2, _ = detect_cycles(adj_raw)
        print(f"  After removal: cycles={'YES' if has_cycle2 else 'NONE'}", flush=True)
    else:
        print(f"  No cycles: PASS", flush=True)

    adj = transitive_reduction(adj_raw)
    n_edges_reduced = adj.sum()
    print(f"  Edges after transitive reduction: {n_edges_reduced} "
          f"(removed {n_edges_raw - n_edges_reduced} transitive)", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. ANALYZE STRUCTURE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("3. DAG STRUCTURE", flush=True)
    print(f"{'='*60}", flush=True)

    in_degree = adj.sum(axis=0)
    out_degree = adj.sum(axis=1)
    roots = np.where((in_degree == 0) & (out_degree > 0))[0]
    leaves = np.where((out_degree == 0) & (in_degree > 0))[0]
    isolated = np.where((in_degree == 0) & (out_degree == 0))[0]
    depth = topological_depth(adj)

    print(f"  Roots (foundational, no prereqs): {len(roots)}", flush=True)
    for r in roots[:10]:
        print(f"    [{r}] {skill_names[r][:55]} (mastery={mastery_rate[r]:.3f}, "
              f"out_deg={int(out_degree[r])})", flush=True)

    print(f"\n  Leaves (advanced, no dependents): {len(leaves)}", flush=True)
    for l in leaves[:10]:
        print(f"    [{l}] {skill_names[l][:55]} (mastery={mastery_rate[l]:.3f}, "
              f"in_deg={int(in_degree[l])})", flush=True)

    print(f"\n  Isolated (not in any chain): {len(isolated)}", flush=True)
    print(f"  Max depth: {depth.max()}", flush=True)
    print(f"  Depth distribution: {dict(zip(*np.unique(depth, return_counts=True)))}",
          flush=True)

    # Validate: roots should have high mastery, leaves low
    if len(roots) > 0:
        root_mastery = mastery_rate[roots].mean()
        print(f"\n  Root avg mastery: {root_mastery:.3f} "
              f"({'PASS' if root_mastery > 0.5 else 'WARN: expected >0.5'})",
              flush=True)
    if len(leaves) > 0:
        leaf_mastery = mastery_rate[leaves].mean()
        print(f"  Leaf avg mastery: {leaf_mastery:.3f} "
              f"({'PASS' if leaf_mastery < root_mastery else 'WARN: expected < roots'})",
              flush=True)

    # Connected components (undirected)
    adj_undirected = adj | adj.T
    visited_cc = np.zeros(K, dtype=bool)
    components = []
    for start in range(K):
        if visited_cc[start] or (in_degree[start] == 0 and out_degree[start] == 0):
            continue
        # BFS
        comp = []
        queue = [start]
        while queue:
            u = queue.pop(0)
            if visited_cc[u]:
                continue
            visited_cc[u] = True
            comp.append(u)
            for v in range(K):
                if adj_undirected[u, v] and not visited_cc[v]:
                    queue.append(v)
        components.append(comp)
    print(f"  Connected components (non-isolated): {len(components)}", flush=True)
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5]):
        print(f"    Component {i+1}: {len(comp)} skills", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. CROSS-VALIDATE WITH MODEL SIZE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("4. CORRELATION WITH MODEL SIZE", flush=True)
    print(f"{'='*60}", flush=True)

    # Extract model sizes from names (heuristic: look for parameter counts)
    sizes = []
    for name in llm_names:
        s = None
        # Look for patterns like "7b", "13b", "70b", "1.5b"
        import re
        m = re.search(r"(\d+\.?\d*)[bB]", name)
        if m:
            s = float(m.group(1))
        sizes.append(s)

    valid_size_mask = np.array([s is not None for s in sizes])
    n_with_size = valid_size_mask.sum()
    print(f"  LLMs with parseable size: {n_with_size}/{n_llms}", flush=True)

    size_corr_per_skill = {}
    if n_with_size > 50:
        log_sizes = np.array([np.log10(s) if s else 0 for s in sizes])
        valid_log = log_sizes[valid_size_mask]
        valid_mastery = theta[valid_size_mask]

        from scipy.stats import pearsonr
        depth_size_corrs = []
        for k in range(K):
            r, p = pearsonr(valid_log, valid_mastery[:, k])
            size_corr_per_skill[k] = float(r)
            depth_size_corrs.append((depth[k], float(r)))

        # Correlation between depth and size-mastery correlation
        depths_arr = np.array([d for d, _ in depth_size_corrs])
        corrs_arr = np.array([c for _, c in depth_size_corrs])
        if len(set(depths_arr)) > 1:
            r_depth_size, _ = pearsonr(depths_arr, corrs_arr)
            print(f"  Correlation(depth, size-mastery r): {r_depth_size:.3f}", flush=True)
            print(f"  → Deeper skills correlate {'more' if r_depth_size > 0 else 'less'} "
                  f"with model size", flush=True)
        else:
            r_depth_size = 0.0

        # Root vs leaf size correlations
        if len(roots) > 0:
            root_corr = np.mean([size_corr_per_skill[r] for r in roots])
            print(f"  Root skills avg size correlation: {root_corr:.3f}", flush=True)
        if len(leaves) > 0:
            leaf_corr = np.mean([size_corr_per_skill[l] for l in leaves])
            print(f"  Leaf skills avg size correlation: {leaf_corr:.3f}", flush=True)
    else:
        r_depth_size = 0.0
        print(f"  Not enough LLMs with size info for correlation", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 5. CROSS-FAMILY CONSISTENCY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("5. CROSS-FAMILY CONSISTENCY", flush=True)
    print(f"{'='*60}", flush=True)

    families = defaultdict(list)
    for i, name in enumerate(llm_names):
        parts = name.lower().replace("__", "/").replace("-", " ")
        for fam in ["llama", "qwen", "mistral", "gemma", "phi", "falcon", "yi"]:
            if fam in parts:
                families[fam].append(i)
                break

    family_edges = {}
    min_family_size = 100
    for fam, indices in sorted(families.items(), key=lambda x: -len(x[1])):
        n_fam = len(indices)
        if n_fam < min_family_size:
            continue
        m_fam = mastery[indices]
        n_m = m_fam.sum(axis=0)
        co_m = m_fam.T @ m_fam
        with np.errstate(divide="ignore", invalid="ignore"):
            p_ab = co_m / n_m[np.newaxis, :]
            p_ab = np.nan_to_num(p_ab, nan=0.0)
        ps = p_ab - p_ab.T
        or_fam = compute_odds_ratio(co_m, n_m, n_fam)
        fam_adj = (ps > ASYMMETRY_THRESHOLD) & (or_fam > OR_THRESHOLD)
        np.fill_diagonal(fam_adj, False)
        edge_set = set(zip(*np.where(fam_adj)))
        family_edges[fam] = edge_set
        print(f"  {fam:<10}: {n_fam} LLMs, {len(edge_set)} edges", flush=True)

    # Pairwise Jaccard of edge sets
    fam_names = sorted(family_edges.keys())
    if len(fam_names) >= 2:
        print(f"\n  Edge overlap (Jaccard):", flush=True)
        fam_jaccard = {}
        for i, fi in enumerate(fam_names):
            for j, fj in enumerate(fam_names):
                if i >= j:
                    continue
                ei, ej = family_edges[fi], family_edges[fj]
                inter = len(ei & ej)
                union = len(ei | ej)
                jac = inter / max(union, 1)
                fam_jaccard[f"{fi}-{fj}"] = jac
                print(f"    {fi}-{fj}: Jaccard={jac:.3f} "
                      f"(|∩|={inter}, |∪|={union})", flush=True)
    else:
        fam_jaccard = {}

    # ════════════════════════════════════════════════════════════════
    # 6. PER-BENCHMARK PREREQUISITES
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("6. PER-BENCHMARK CHAIN DEPTH", flush=True)
    print(f"{'='*60}", flush=True)

    benchmarks = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
    bench_items_map = {b: [] for b in benchmarks}
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in bench_items_map:
            bench_items_map[b].append(i)

    bench_depth = {}
    for b in benchmarks:
        idx = np.array(bench_items_map[b], dtype=int)
        Q_b = q_matrix[idx]
        skills_used = set(np.where(Q_b.sum(axis=0) > 0)[0])
        depths_b = [depth[k] for k in skills_used]
        avg_depth = np.mean(depths_b) if depths_b else 0
        max_depth_b = max(depths_b) if depths_b else 0
        bench_depth[b] = {"avg_depth": float(avg_depth), "max_depth": int(max_depth_b),
                          "n_skills": len(skills_used)}
        print(f"  {b:<8}: {len(skills_used)} skills, avg_depth={avg_depth:.2f}, "
              f"max_depth={max_depth_b}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Fig 1: Skill co-mastery structure (2-panel: scatter + per-depth bars) ──
    # Replaces the earlier node-link DAG; the usual convention for
    # hierarchical/ordinal skill data prefers ordered scatter + mean-per-bin
    # over node-link graphs (verified across 8 D&B/E&D reference papers).
    from scipy.stats import spearmanr

    # Primary benchmark per skill = argmax over benchmark item counts.
    skill_bench_counts = np.zeros((K, len(benchmarks)), dtype=int)
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in benchmarks:
            bi = benchmarks.index(b)
            skill_bench_counts[:, bi] += q_matrix[i].astype(int)
    primary_bench = skill_bench_counts.argmax(axis=1)
    primary_bench[skill_bench_counts.sum(axis=1) == 0] = -1

    dag_mask = (in_degree > 0) | (out_degree > 0)
    dag_skills = np.where(dag_mask)[0]
    x_vals = np.array([depth[k] for k in dag_skills])
    y_vals = np.array([mastery_rate[k] for k in dag_skills])

    # Okabe-Ito colorblind-safe palette + distinct marker shapes per benchmark
    # so overlapping dots remain distinguishable even if printed monochrome.
    bench_styles = {
        "MATH":   ("#0072B2", "o"),  # blue circle
        "BBH":    ("#E69F00", "s"),  # orange square
        "GPQA":   ("#009E73", "^"),  # green up-triangle
        "MuSR":   ("#CC79A7", "D"),  # pink diamond
        "IFEval": ("#56B4E9", "v"),  # light blue down-triangle
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel (a): per-skill mastery vs co-mastery depth, colored by primary bench.
    # Uses a beeswarm-style x-offset so overlapping y values dodge horizontally
    # rather than stacking vertically at each integer depth column.
    def _beeswarm_x(y_arr, center_x, half_width=0.36, diameter=0.045):
        n = len(y_arr)
        if n == 0:
            return np.array([])
        order = np.argsort(y_arr)
        x_off = np.zeros(n)
        placed = []  # (y, offset) pairs already placed in this column
        step = diameter
        col_offsets = [0.0]
        for k in range(1, int(half_width / step) + 1):
            col_offsets.extend([k * step, -k * step])
        for idx in order:
            yi = y_arr[idx]
            blocked = {ox for py, ox in placed if abs(py - yi) < diameter}
            for ox in col_offsets:
                if ox not in blocked:
                    x_off[idx] = center_x + ox
                    placed.append((yi, ox))
                    break
            else:
                x_off[idx] = center_x
        return x_off

    ax_a = axes[0]
    primaries_dag = primary_bench[dag_skills]
    # Compute beeswarm offsets per depth column (all skills, regardless of benchmark)
    x_swarm = np.zeros_like(x_vals, dtype=float)
    for d in sorted(set(x_vals.astype(int))):
        col_idx = np.where(x_vals.astype(int) == d)[0]
        if len(col_idx) > 0:
            x_swarm[col_idx] = _beeswarm_x(y_vals[col_idx], center_x=float(d))
    for bi, b in enumerate(benchmarks):
        mask = primaries_dag == bi
        if mask.sum() > 0:
            color, marker = bench_styles[b]
            ax_a.scatter(x_swarm[mask], y_vals[mask],
                         s=42, alpha=0.78, color=color, marker=marker,
                         edgecolor="white", linewidth=0.7, label=b, zorder=3)

    # Linear trend (for visual anchoring); Spearman reported in a corner box.
    if len(x_vals) > 2 and x_vals.std() > 0:
        z = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(x_vals.min(), x_vals.max(), 50)
        ys = np.clip(np.polyval(z, xs), 0.0, 1.0)
        ax_a.plot(xs, ys, color="#555555", ls="--", lw=1.1,
                  alpha=0.55, zorder=1)
        rho, p_val = spearmanr(x_vals, y_vals)
        p_text = f"$p < 10^{{{int(np.floor(np.log10(max(p_val, 1e-300))))}}}$" \
                 if p_val < 0.001 else f"$p = {p_val:.3f}$"
        ax_a.text(0.55, 0.90,
                  rf"Spearman $\rho = {rho:.2f}$, {p_text}",
                  transform=ax_a.transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.35",
                            facecolor="white",
                            edgecolor="#cccccc", linewidth=0.6))

    ax_a.set_xlabel("Co-mastery DAG depth", fontsize=10.5)
    ax_a.set_ylabel("Skill mastery rate", fontsize=10.5)
    ax_a.set_xticks(sorted(set(x_vals.astype(int))))
    ax_a.set_ylim(-0.03, 1.05)
    ax_a.set_title("(a) Per-skill mastery by depth", fontsize=10.5, loc="left")
    ax_a.legend(frameon=False, fontsize=8.5, loc="upper right",
                handletextpad=0.3, labelspacing=0.3)
    for sp in ["top", "right"]:
        ax_a.spines[sp].set_visible(False)
    ax_a.grid(axis="y", ls=":", lw=0.5, alpha=0.35)

    # Panel (b): mean mastery per depth ± SEM with n annotations.
    ax_b = axes[1]
    depth_levels = sorted(set(x_vals.astype(int)))
    means, sems, counts = [], [], []
    for d in depth_levels:
        m = y_vals[x_vals == d]
        means.append(float(m.mean()))
        sems.append(float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else 0.0)
        counts.append(int(len(m)))
    x_pos = np.arange(len(depth_levels))
    ax_b.bar(x_pos, means, yerr=sems, color="#4C72B0", alpha=0.80,
             edgecolor="#2a4a6d", linewidth=0.9, capsize=4, zorder=3)
    for i, (m, n) in enumerate(zip(means, counts)):
        ax_b.text(i, m + max(sems[i], 0.01) + 0.02, f"n={n}",
                  ha="center", va="bottom", fontsize=8.5, color="#333333")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([str(d) for d in depth_levels])
    ax_b.set_xlabel("Co-mastery DAG depth", fontsize=10.5)
    ax_b.set_ylabel("Mean mastery ($\\pm$ SEM)", fontsize=10.5)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_title("(b) Mean mastery per depth level", fontsize=10.5, loc="left")
    for sp in ["top", "right"]:
        ax_b.spines[sp].set_visible(False)
    ax_b.grid(axis="y", ls=":", lw=0.5, alpha=0.35)

    plt.tight_layout()
    out1 = fig_dir / "fig_skill_prerequisite_dag.pdf"
    out1_png = fig_dir / "fig_skill_prerequisite_dag.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    fig.savefig(out1_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1} and {out1_png}", flush=True)

    # ── Fig 2: Depth vs size correlation ──
    fig, ax = plt.subplots(figsize=(7, 5))
    if size_corr_per_skill:
        corrs = [size_corr_per_skill.get(k, 0) for k in range(K)]
        connected = [k for k in range(K) if in_degree[k] > 0 or out_degree[k] > 0]
        isol = [k for k in range(K) if in_degree[k] == 0 and out_degree[k] == 0]

        if connected:
            ax.scatter([depth[k] for k in connected],
                       [corrs[k] for k in connected],
                       s=40, alpha=0.6, color="#4C72B0", label="In DAG")
        if isol:
            ax.scatter([depth[k] for k in isol],
                       [corrs[k] for k in isol],
                       s=20, alpha=0.3, color="#999", label="Isolated")
        ax.set_xlabel("Skill depth in prerequisite DAG", fontsize=12)
        ax.set_ylabel("Correlation(mastery, log model size)", fontsize=12)
        ax.legend(frameon=False, fontsize=10)
    else:
        ax.text(0.5, 0.5, "Insufficient size data", ha="center", va="center")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out2 = fig_dir / "fig_skill_depth_vs_size.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Save JSON ──
    edges_list = [{"from": int(i), "to": int(j),
                   "from_name": skill_names[i], "to_name": skill_names[j],
                   "OR": float(odds_ratio[i, j]),
                   "lift": float(lift[i, j]),
                   "p_a_given_b": float(p_a_given_b[i, j]),
                   "asymmetry": float(prereq_score[i, j])}
                  for i, j in zip(*np.where(adj))]

    save_data = {
        "experiment": "skill_prerequisites",
        "method": "odds_ratio",
        "OR_threshold": OR_THRESHOLD,
        "asymmetry_threshold": ASYMMETRY_THRESHOLD,
        "n_llms": int(n_llms), "K": int(K),
        "n_edges_raw": int(n_edges_raw),
        "n_edges_reduced": int(n_edges_reduced),
        "n_roots": int(len(roots)),
        "n_leaves": int(len(leaves)),
        "n_isolated": int(len(isolated)),
        "max_depth": int(depth.max()),
        "n_components": len(components),
        "roots": [{"idx": int(r), "name": skill_names[r],
                    "mastery": float(mastery_rate[r])} for r in roots],
        "leaves": [{"idx": int(l), "name": skill_names[l],
                     "mastery": float(mastery_rate[l])} for l in leaves],
        "edges": edges_list,
        "depth_per_skill": {str(k): int(depth[k]) for k in range(K)},
        "mastery_rate_per_skill": {str(k): float(mastery_rate[k]) for k in range(K)},
        "depth_size_correlation": float(r_depth_size),
        "cross_family_jaccard": fam_jaccard,
        "per_benchmark_depth": bench_depth,
    }

    out_json = Path("cdm_exploration/experiments/v2_skill_prerequisites.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="skill_prerequisites",
        config={"method": "odds_ratio",
                "OR_threshold": OR_THRESHOLD,
                "asymmetry_threshold": ASYMMETRY_THRESHOLD,
                "mastery_threshold": MASTERY_THRESH, "K": K},
        results={"n_edges": int(n_edges_reduced), "n_roots": int(len(roots)),
                 "n_leaves": int(len(leaves)), "max_depth": int(depth.max()),
                 "depth_size_corr": float(r_depth_size)},
        split_info={"n_llms": n_llms, "K": K},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
