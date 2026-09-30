#!/usr/bin/env python3
"""An external review, point 2: results under "mastered = theta > 0.5" against predicted and observed skill accuracy.

Reads the outputs of tools/run_mastery_predicted.sbatch (no model is run):
  v2_predicted_skill_accuracy.{npz,json}
  v2_skill_prerequisites{tag}.json, v2_conjunctive_compensatory{tag}.json for
  tag in _theta050check, _predicted040, _predicted050, _predicted060, _observed050.
The theta check must equal the submitted files v2_skill_prerequisites.json and v2_conjunctive_compensatory.json.

Also recomputes tab:divergent (items 6398 BBH and 8261 IFEval: accuracy of LLMs mastering both skills, only one, or
neither) under each definition, directly from the response matrix.

Output: cdm_exploration/experiments/v2_mastery_definition_compare.json

    python3 tools/diag_mastery_definition_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.evaluation.skill_mastery import skill_accuracy
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

EXP = REPO / "cdm_exploration/experiments"
DATA = REPO / "cdm_exploration/data/cdm_ready"
RUNS = [("theta", 0.5, "_theta050check"), ("predicted", 0.4, "_predicted040"), ("predicted", 0.5, "_predicted050"),
        ("predicted", 0.6, "_predicted060"), ("observed", 0.5, "_observed050")]
TABLE_ITEMS = {"BBH": 6398, "IFEval": 8261}      # tab:divergent, found in the theta pattern results
NAMED_LLMS = {"allenai__Llama-3.1-Tulu-3-70B": 31, "Bllossom__llama-3.2-Korean-Bllossom-AICA-5B": 11}
EXAMPLE_SKILL = "Evaluating Truthfulness Of Nested Statements"
EXAMPLE_LLMS = ("meta-llama__Llama-3.2-1B", "meta-llama__Llama-3.2-1B-Instruct")   # Instruct: the 22 Sep example


def graph_summary(d):
    fam = list(d["cross_family_jaccard"].values())
    return {"n_connected_skills": d["n_connected_skills"], "n_links": d["n_edges_reduced"], "max_depth": d["max_depth"],
            "n_roots": d["n_roots"], "n_leaves": d["n_leaves"], "n_isolated": d["n_isolated"],
            "spearman_depth_prevalence": d["spearman_depth_prevalence"],
            "depth_size_correlation": d["depth_size_correlation"],
            "avg_depth_per_benchmark": {b: v["avg_depth"] for b, v in d["per_benchmark_depth"].items()},
            "cross_family_jaccard_min": min(fam), "cross_family_jaccard_max": max(fam),
            "leaves": [(x["name"], x["mastery"]) for x in d["leaves"]]}


def edges(d):
    return {(e["from"], e["to"]) for e in d["edges"]}


def conj_summary(d):
    out = {}
    for key, name in (("tertile", "tertile"), ("binary_sensitivity", "binary"), ("pattern_robustness_2skill", "pattern")):
        out[name] = {"n": d[key]["n_analyzed"], **{t: d[key]["overall_pct"][t] for t in ("conjunctive", "compensatory", "additive")}}
    out["tertile"]["per_benchmark_conj_pct"] = {b: v["conjunctive_pct"] for b, v in d["tertile"]["per_benchmark"].items()}
    out["tertile"]["per_benchmark_comp_pct"] = {b: v["compensatory_pct"] for b, v in d["tertile"]["per_benchmark"].items()}
    out["agreement_tertile_pattern"] = d["pattern_robustness_2skill"]["tertile_vs_pattern_agreement"]
    thr = {round(s["conj_thresh"], 1): s["pct"] for s in d["threshold_sensitivity_tertile"]}
    out["tertile_reclassified_0.6_to_0.7_pct"] = thr[0.6]["compensatory"] - thr[0.7]["compensatory"]
    return out


def table_row(R, score, cut, j, a, b):
    m = score > cut
    groups = {"both": m[:, a] & m[:, b], "skill1_only": m[:, a] & ~m[:, b], "skill2_only": ~m[:, a] & m[:, b],
              "neither": ~m[:, a] & ~m[:, b]}
    return {g: {"n": int(mask.sum()), "acc": float(R[mask, j].mean()) if mask.any() else None} for g, mask in groups.items()}


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="mastery_definition_compare")
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    skills = json.load(open(DATA / "cluster_labels_v2_K100.json"))
    npz = np.load(EXP / "v2_predicted_skill_accuracy.npz")
    pj = json.load(open(EXP / "v2_predicted_skill_accuracy.json"))
    scores = {"theta": npz["theta"], "predicted": npz["pred_skill_acc"], "observed": skill_accuracy(R, Q)}
    assert np.allclose(scores["observed"], npz["obs_skill_acc"], atol=1e-5)

    # The theta check must reproduce the submitted files exactly.
    sub_g = json.load(open(EXP / "v2_skill_prerequisites.json"))
    sub_c = json.load(open(EXP / "v2_conjunctive_compensatory.json"))
    chk_g = json.load(open(EXP / "v2_skill_prerequisites_theta050check.json"))
    chk_c = json.load(open(EXP / "v2_conjunctive_compensatory_theta050check.json"))
    g_diff = [k for k in sub_g if k != "depth_size_correlation" and sub_g[k] != chk_g[k]]
    c_diff = [k for k in sub_c if sub_c[k] != chk_c[k]]
    theta_check = not g_diff and not c_diff and abs(sub_g["depth_size_correlation"] - chk_g["depth_size_correlation"]) < 1e-9
    print(f"theta check reproduces the submitted files: {theta_check} (graph diffs {g_diff}, conj diffs {c_diff})")

    names = json.load(open(DATA / "response_matrix_v2_full_llms.json"))
    k_ex = next(int(k) for k, v in skills.items() if v == EXAMPLE_SKILL)
    example = {n: {"theta": float(scores["theta"][names.index(n), k_ex]),
                   "predicted": float(scores["predicted"][names.index(n), k_ex]),
                   "observed": float(scores["observed"][names.index(n), k_ex])} for n in EXAMPLE_LLMS}
    print("example skill:", {n.split("__")[-1]: {k: round(v, 2) for k, v in d.items()} for n, d in example.items()})
    base_edges, base_depth = edges(chk_g), np.array([chk_g["depth_per_skill"][str(k)] for k in range(100)])
    align = json.load(open(EXP / "v2_alignment_tax.json"))
    align_delta = np.array([align["per_skill"][str(k)]["mean_delta"] for k in range(100)])
    res = {"experiment": "mastery_definition_compare", "theta_check_reproduces_submitted": theta_check,
           "predict_checks": {k: pj[k] for k in ("test_auc", "test_auc_pass", "example", "cells_disagree_at_0.5",
                                                   "r_predicted_observed_cells")},
           "example_skill": {"skill": EXAMPLE_SKILL, "llms": example},
           "runs": {}}
    prev_t = (scores["theta"] > 0.5).mean(0)
    for src, cut, tag in RUNS:
        g = json.load(open(EXP / f"v2_skill_prerequisites{tag}.json"))
        c = json.load(open(EXP / f"v2_conjunctive_compensatory{tag}.json"))
        e, dep = edges(g), np.array([g["depth_per_skill"][str(k)] for k in range(100)])
        prev = (scores[src] > cut).mean(0)
        run = {"mastery": src, "cut": cut, "graph": graph_summary(g),
               "links_shared_with_theta_graph": len(e & base_edges),
               "link_jaccard_with_theta_graph": len(e & base_edges) / max(len(e | base_edges), 1),
               "depth_spearman_with_theta_graph": float(spearmanr(dep, base_depth)[0]),
               "prevalence_change_vs_theta_pts": {"median_abs": float(100 * np.median(np.abs(prev - prev_t))),
                                                  "p90_abs": float(100 * np.percentile(np.abs(prev - prev_t), 90)),
                                                  "max_abs": float(100 * np.max(np.abs(prev - prev_t)))},
               "conjunctive": conj_summary(c), "tab_divergent": {}}
        for bench, j in TABLE_ITEMS.items():
            a, b = np.where(Q[j] > 0)[0]
            run["tab_divergent"][bench] = {"item": j, "skill1": skills[str(a)], "skill2": skills[str(b)],
                                           **table_row(R, scores[src], cut, j, a, b)}
        # Appendix alignment-by-depth: mean base-to-instruct difference per depth level (deltas unchanged, depths new).
        run["align_delta_by_depth"] = {int(d): {"n": int((dep == d).sum()), "mean_delta": float(align_delta[dep == d].mean())}
                                       for d in sorted(set(dep.tolist()))}
        # The two LLMs named in the paragraph next to tab:divergent (item 6398).
        a, b = np.where(Q[TABLE_ITEMS["BBH"]] > 0)[0]
        run["named_llms_item_6398"] = {name: {"masters_skill1": bool(scores[src][i, a] > cut),
                                              "masters_skill2": bool(scores[src][i, b] > cut),
                                              "score_skill1": float(scores[src][i, a]), "score_skill2": float(scores[src][i, b]),
                                              "correct": int(R[i, TABLE_ITEMS["BBH"]])}
                                       for name, i in NAMED_LLMS.items()}
        res["runs"][tag] = run

    for tag, r in res["runs"].items():
        gs = r["graph"]
        print(f"\n{tag}: {gs['n_connected_skills']} skills, {gs['n_links']} links, depth {gs['max_depth']}, "
              f"roots {gs['n_roots']}, leaves {gs['n_leaves']}, rho {gs['spearman_depth_prevalence']:.2f}, "
              f"shared links {r['links_shared_with_theta_graph']} (Jaccard {r['link_jaccard_with_theta_graph']:.2f}), "
              f"depth rho vs theta graph {r['depth_spearman_with_theta_graph']:.2f}")
        print("   prevalence change vs theta (pts):", {k: round(v, 1) for k, v in r["prevalence_change_vs_theta_pts"].items()})
        print("   conj:", {m: (v["n"], round(v["conjunctive"], 1), round(v["compensatory"], 1), round(v["additive"], 1))
                           for m, v in r["conjunctive"].items() if m in ("tertile", "binary", "pattern")})
        for bench, t in r["tab_divergent"].items():
            print(f"   {bench} item {t['item']}:", {g: (t[g]["n"], None if t[g]["acc"] is None else round(t[g]["acc"], 2))
                                                  for g in ("both", "skill1_only", "skill2_only", "neither")})
        print("   alignment delta by depth:", {d: (v["n"], round(v["mean_delta"], 3)) for d, v in r["align_delta_by_depth"].items()})
        print("   named LLMs on item 6398:", {n.split("__")[-1]: (v["masters_skill1"], v["masters_skill2"], v["correct"])
                                              for n, v in r["named_llms_item_6398"].items()})
    res["verified"] = bool(ok and theta_check and pj["test_auc_pass"])
    (EXP / "v2_mastery_definition_compare.json").write_text(json.dumps(res, indent=2))
    log_experiment(name="diag_mastery_definition_compare", config={"seed": 42, "runs": [r[2] for r in RUNS]},
                   results={t: {k: v for k, v in r.items() if k != "graph"} for t, r in res["runs"].items()},
                   split_info={"n_train_items": len(tr), "n_test_items": len(te), "random_state": 42},
                   verified=res["verified"])


if __name__ == "__main__":
    main()
