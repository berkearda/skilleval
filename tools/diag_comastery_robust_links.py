#!/usr/bin/env python3
"""Co-mastery under the predicted-accuracy mastery rule: graph summary, cut-off robustness, the longest chains, and the
chains for the main-text figure (the review's point 2; supervisor feedback: a main-text figure instead of an appendix pointer;
24 Sep 2026).

No model is run. Reads the graphs written by tools/run_skill_prerequisites.py (Euler job 14856635):
  v2_skill_prerequisites_predicted{040,050,060}.json   mastery = mean predicted P(correct) on the skill's items > cut
  v2_skill_prerequisites_observed050.json               model-free comparison (observed accuracy > 0.5)
  v2_skill_prerequisites_theta050check.json             the old rule (theta > 0.5), identical to the submitted graph
and the mastery scores in v2_predicted_skill_accuracy.npz (tools/predict_skill_accuracy.py).

A link is "robust" if it is present in the reduced graph at all three cut-offs (0.4, 0.5, 0.6).

Figure chains, fixed by rule and not by inspection: the longest chain of robust links whose skills all take most of their
items from MATH, and the same for IFEval. Both benchmarks have answers that cannot be guessed from a few options; BBH
skills such as Boolean expressions (True/False) reach a predicted accuracy of 0.5 by guessing alone. The script asserts
that each longest chain is unique.

Depth: run_skill_prerequisites.py::topological_depth visits every skill once in breadth-first order, so a skill whose
depth grows after it was visited never passes the larger depth on to its children, and depth is underestimated (T-136).
Here the longest chain is computed with Kahn's algorithm, which fixes a skill's depth only after all its parents; the
reported (underestimated) values are stored next to it for the record.

Output: cdm_exploration/experiments/v2_comastery_robust.json

    python3 tools/diag_comastery_robust_links.py
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

EXP = REPO / "cdm_exploration/experiments"
DATA = REPO / "cdm_exploration/data/cdm_ready"
BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def closure(edge_set, K=100):
    A = np.zeros((K, K), bool)
    for i, j in edge_set:
        A[i, j] = True
    for k in range(K):
        A |= A[:, [k]] & A[[k], :]
    return {(int(i), int(j)) for i, j in zip(*np.where(A))}


def longest_chain_depth(edge_set, K=100):
    """Depth = number of links on the longest chain ending at each skill (Kahn's algorithm)."""
    children, indeg = {k: [] for k in range(K)}, np.zeros(K, int)
    for a, b in edge_set:
        children[a].append(b)
        indeg[b] += 1
    depth, queue, done = np.zeros(K, int), deque(k for k in range(K) if indeg[k] == 0), 0
    while queue:
        u = queue.popleft()
        done += 1
        for v in children[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    assert done == K, "graph has a cycle"
    return depth


def raw_link_count(M, asym_cut=0.3, or_cut=10.0, c=0.5):
    """Links before cycle removal and reduction, with the rule and odds ratio of run_skill_prerequisites.py."""
    n = M.sum(axis=0).astype(float)
    co = M.T.astype(float) @ M.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.nan_to_num(co / n[np.newaxis, :], nan=0.0)          # p[a, b] = P(A | B)
    a_not_b, not_a_b = n[:, np.newaxis] - co, n[np.newaxis, :] - co
    neither = M.shape[0] - n[:, np.newaxis] - n[np.newaxis, :] + co
    odds_ratio = ((co + c) * (neither + c)) / ((a_not_b + c) * (not_a_b + c))
    adj = ((p - p.T) > asym_cut) & (odds_ratio > or_cut)
    np.fill_diagonal(adj, False)
    return int(adj.sum())


def longest_paths(edge_set):
    children = {}
    for a, b in edge_set:
        children.setdefault(a, []).append(b)
    paths = []

    def extend(path):
        if path[-1] not in children:
            paths.append(path)
        for n in children.get(path[-1], []):
            extend(path + [n])
    for s in {a for a, _ in edge_set} - {b for _, b in edge_set}:
        extend([s])
    top = max(len(p) for p in paths)
    return [p for p in paths if len(p) == top]


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="comastery_robust_links")
    names = json.load(open(DATA / "cluster_labels_v2_K100.json"))
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    counts = np.zeros((q.shape[1], len(BENCHMARKS)), int)   # items per skill and benchmark, as in run_skill_prerequisites.py
    for i, it in enumerate(items):
        counts[:, BENCHMARKS.index(it["benchmark"])] += q[i].astype(int)
    primary = {s: BENCHMARKS[int(counts[s].argmax())] for s in range(q.shape[1])}

    tags = ["predicted040", "predicted050", "predicted060", "observed050", "theta050check"]
    G = {t: json.load(open(EXP / f"v2_skill_prerequisites_{t}.json")) for t in tags}
    edges = {t: {(e["from"], e["to"]) for e in G[t]["edges"]} for t in tags}
    cl = {t: closure(edges[t]) for t in tags}
    depth = {t: longest_chain_depth(edges[t]) for t in tags}
    for t in tags:
        reported = G[t]["depth_per_skill"]
        reported = np.array([reported[str(k)] if isinstance(reported, dict) else reported[k] for k in range(100)])
        assert reported.max() == G[t]["max_depth"] and (depth[t] >= reported).all()
    robust = edges["predicted040"] & edges["predicted050"] & edges["predicted060"]

    # mastery matrix at 0.5 from the saved scores; cross-check against the graph file
    mastered = np.load(EXP / "v2_predicted_skill_accuracy.npz")["pred_skill_acc"] > 0.5
    prev = mastered.mean(axis=0)
    rate = G["predicted050"]["mastery_rate_per_skill"]
    assert all(abs(prev[s] - rate[str(s)]) < 1e-12 for s in range(100))
    edge050 = {(e["from"], e["to"]): e for e in G["predicted050"]["edges"]}

    # null control (F9): shuffle each skill's mastery across LLMs independently, which keeps how many LLMs master each
    # skill but removes any relation between skills; count the links the same rule then finds (200 shuffles, seed 42)
    n_raw = raw_link_count(mastered)
    assert n_raw == G["predicted050"]["n_edges_raw"], (n_raw, G["predicted050"]["n_edges_raw"])
    rng = np.random.default_rng(42)
    null_counts = []
    for _ in range(200):
        shuffled = np.column_stack([rng.permutation(mastered[:, k]) for k in range(mastered.shape[1])])
        null_counts.append(raw_link_count(shuffled))
    null_control = {"n_raw_links_real": n_raw, "n_shuffles": 200, "null_mean_raw_links": float(np.mean(null_counts)),
                    "null_max_raw_links": int(np.max(null_counts)), "seed": 42}

    def link(a, b):
        n_a, n_b, n_ab = int(mastered[:, a].sum()), int(mastered[:, b].sum()), int((mastered[:, a] & mastered[:, b]).sum())
        e = edge050[(a, b)]
        assert abs(n_ab / n_b - e["p_a_given_b"]) < 1e-12 and abs(n_ab / n_b - n_ab / n_a - e["asymmetry"]) < 1e-12
        return {"from": a, "to": b, "n_llms_master_from": n_a, "n_llms_master_to": n_b, "n_llms_master_both": n_ab,
                "p_from_given_to": n_ab / n_b, "p_to_given_from": n_ab / n_a, "odds_ratio": e["OR"]}

    chains = {}
    for bench in ("MATH", "IFEval"):
        sub = {(a, b) for a, b in robust if primary[a] == bench and primary[b] == bench}
        best = longest_paths(sub)
        assert len(best) == 1, (bench, best)
        ch = best[0]
        links = [link(a, b) for a, b in zip(ch, ch[1:])]
        chains[bench] = {"rule": f"longest chain of robust links among skills with most items from {bench}",
                         "skills": [{"skill": s, "name": names[str(s)], "prevalence": float(prev[s]),
                                     "n_llms_master": int(mastered[:, s].sum()), "primary_benchmark": primary[s],
                                     "n_items": int(counts[s].sum()),
                                     "items_per_benchmark": dict(zip(BENCHMARKS, counts[s].tolist()))} for s in ch],
                         "links": links,
                         "min_p_from_given_to": min(l["p_from_given_to"] for l in links)}

    g05 = G["predicted050"]
    # how asymmetric the links are in general, not only in the example: shares in both directions over all links at 0.5
    p_fwd = np.array([e["p_a_given_b"] for e in g05["edges"]])
    p_rev = p_fwd - np.array([e["asymmetry"] for e in g05["edges"]])
    link_shares = {"n_links": len(p_fwd), "median_p_from_given_to": float(np.median(p_fwd)),
                   "median_p_to_given_from": float(np.median(p_rev)),
                   "share_links_p_from_given_to_above_half": float((p_fwd > 0.5).mean()),
                   "min_p_from_given_to": float(p_fwd.min())}
    assert link_shares["n_links"] == g05["n_edges_reduced"]
    res = {"experiment": "comastery_robust_links", "rule": "mastered = mean predicted P(correct) on the skill's items > cut",
           "n_llms": int(mastered.shape[0]),
           "graph_at_0.5": {**{k: g05[k] for k in ("n_connected_skills", "n_edges_reduced", "n_roots", "n_leaves",
                                                   "n_isolated", "n_components")},
                            "longest_chain_links": int(depth["predicted050"].max()),
                            "max_depth_reported_underestimated": g05["max_depth"]},
           "sweep": {t: {"n_connected_skills": G[t]["n_connected_skills"], "n_edges_reduced": G[t]["n_edges_reduced"],
                         "longest_chain_links": int(depth[t].max()), "max_depth_reported_underestimated": G[t]["max_depth"]}
                     for t in ("predicted040", "predicted060", "observed050", "theta050check")},
           "n_links_robust_all_cutoffs": len(robust),
           "n_links_robust_and_in_observed_graph": len(robust & edges["observed050"]),
           "orderings": {"predicted050_vs_observed050": {"shared": len(cl["predicted050"] & cl["observed050"]),
                                                         "reversed": len({(j, i) for i, j in cl["predicted050"]} & cl["observed050"]),
                                                         "n_observed": len(cl["observed050"]),
                                                         "n_predicted": len(cl["predicted050"])},
                         "theta_vs_observed050": {"shared": len(cl["theta050check"] & cl["observed050"]),
                                                  "reversed": len({(j, i) for i, j in cl["theta050check"]} & cl["observed050"]),
                                                  "n_theta": len(cl["theta050check"])}},
           "chains": chains,
           "link_shares_at_0.5": link_shares,
           "null_control_at_0.5": null_control,
           "verified": bool(ok)}
    (EXP / "v2_comastery_robust.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("graph_at_0.5", "sweep", "n_links_robust_all_cutoffs",
                                          "n_links_robust_and_in_observed_graph", "orderings", "link_shares_at_0.5",
                                          "null_control_at_0.5")}, indent=1))
    for bench, c in chains.items():
        print(bench, " -> ".join(f"{s['name']} ({100 * s['prevalence']:.1f}%, {s['n_items']} items)" for s in c["skills"]))
        for l in c["links"]:
            print(f"   {l['from']}->{l['to']}: {l['n_llms_master_both']} of the {l['n_llms_master_to']} LLMs mastering the later "
                  f"skill master the earlier one ({100 * l['p_from_given_to']:.1f}%); reverse {100 * l['p_to_given_from']:.1f}%")
    log_experiment(name="diag_comastery_robust_links", config={"cutoffs": [0.4, 0.5, 0.6], "chain_rule": "longest robust chain per benchmark (MATH, IFEval)"},
                   results={k: res[k] for k in ("graph_at_0.5", "sweep", "n_links_robust_all_cutoffs", "orderings", "chains",
                                                  "link_shares_at_0.5", "null_control_at_0.5")},
                   split_info={"n_train_items": len(tr), "n_test_items": len(te), "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
