#!/usr/bin/env python3
"""T-123: submitted cold-start and adaptive-selection results against the rerun with the 763 LLMs held out.

Reads result files only (no model is run):
  submitted : v2_cold_start_fixed.json, v2_adaptive_testing_v2_fixed_{largeN,smallN}.json   (network trained on 3,811 LLMs)
  rerun     : the same names with the suffix _llmsplit                                    (network trained on 3,048 LLMs)
Both use the same 763 LLMs (20%, random_state 42), the same items, calibration recipe and seeds.

Output: cdm_exploration/experiments/v2_llm_holdout_compare.json

    python3 tools/diag_llm_holdout_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

EXP = REPO / "cdm_exploration/experiments"


def by_n(rows, key="N"):
    return {int(r[key]): r for r in rows}


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="llm_holdout_compare")
    old, new = json.load(open(EXP / "v2_cold_start_fixed.json")), json.load(open(EXP / "v2_cold_start_fixed_llmsplit.json"))
    o, n = by_n(old["summary"]), by_n(new["summary"])
    cold = []
    for N in sorted(o):
        cold.append({"N": N, "auc_submitted": o[N]["auc_mean"], "auc_llmsplit": n[N]["auc_mean"],
                     "auc_change": n[N]["auc_mean"] - o[N]["auc_mean"],
                     "pct_of_full_submitted": o[N]["pct_of_full"], "pct_of_full_llmsplit": n[N]["pct_of_full"],
                     "pct_of_same_llm_full_calibration_llmsplit": n[N].get("pct_of_full_calibration")})
    res = {"experiment": "llm_holdout_compare",
           "reference_auc": {"submitted_full_training_50_training_llms": old["full_training_auc"],
                             "llmsplit_full_training_50_training_llms": new["full_training_auc"],
                             "llmsplit_full_calibration_same_763_llms": new.get("full_calibration_auc_same_llms")},
           "cold_start": cold, "adaptive": {}}
    for tag in ("largeN", "smallN"):
        a = json.load(open(EXP / f"v2_adaptive_testing_v2_fixed_{tag}.json"))
        b = json.load(open(EXP / f"v2_adaptive_testing_v2_fixed_llmsplit_{tag}.json"))
        ao, bo = by_n(a["summary"]), by_n(b["summary"])
        rows = []
        for N in sorted(ao):
            row = {"N": N}
            for c in ("random", "heuristic", "trace"):
                row[f"{c}_submitted"] = ao[N][f"{c}_mean"]
                row[f"{c}_llmsplit"] = bo[N][f"{c}_mean"]
            row["heuristic_minus_random_submitted"] = ao[N]["heuristic_mean"] - ao[N]["random_mean"]
            row["heuristic_minus_random_llmsplit"] = bo[N]["heuristic_mean"] - bo[N]["random_mean"]
            rows.append(row)
        res["adaptive"][tag] = rows
    # The paper's significance measure for the small-budget gain (Appendix adaptive selection): gain divided by
    # sqrt of the mean squared per-method SE, with SE = SD over 50 LLMs x 5 seeds / sqrt(5 - 1). Same formula as
    # tools/diag_paper_numbers_recheck_1826.py::check_sigma, applied to both runs.
    import math
    sig = {}
    for name, f in (("submitted", "v2_adaptive_testing_v2_fixed_smallN.json"), ("llmsplit", "v2_adaptive_testing_v2_fixed_llmsplit_smallN.json")):
        rows = by_n(json.load(open(EXP / f))["summary"])
        for N in (5, 10):
            r = rows[N]; gain = r["heuristic_mean"] - r["random_mean"]
            se = math.sqrt(((r["random_std"] / 2) ** 2 + (r["heuristic_std"] / 2) ** 2) / 2)
            sig[f"{name}_N{N}"] = {"gain_pts": 100 * gain, "sigma": gain / se}
    res["small_budget_significance"] = sig
    # Optimiser-sensitivity table (Appendix cold start), N = 500, from tools/diag_cold_start_convergence.py
    sens = {}
    for name, f in (("submitted", "v2_cold_start_convergence_diag.json"), ("llmsplit", "v2_cold_start_convergence_diag_llmsplit.json")):
        d = json.load(open(EXP / f)); ref = d["reference_trained_theta_norm"]
        rows = {f"lr{r['lr']}_steps{r['steps']}": {"auc": r["auc"], "theta_raw_mean_abs": r["theta_raw_mean_abs"],
                                                  "pct_of_reference": 100 * r["theta_raw_mean_abs"] / ref}
                for r in d["results"] if r["N"] == 500}
        rows["longest_minus_default_auc_pts"] = 100 * (rows["lr0.1_steps2000"]["auc"] - rows["lr0.05_steps500"]["auc"])
        sens[name] = {"reference_trained_theta_norm": ref, "N500": rows}
    res["optimiser_sensitivity"] = sens
    res["verified"] = bool(ok)
    (EXP / "v2_llm_holdout_compare.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["reference_auc"], indent=1))
    print(f"{'N':>5} {'AUC sub':>9} {'AUC new':>9} {'change':>8} {'%full sub':>10} {'%full new':>10} {'%same-LLM':>10}")
    for r in cold:
        s = r["pct_of_same_llm_full_calibration_llmsplit"]
        print(f"{r['N']:>5} {r['auc_submitted']:9.4f} {r['auc_llmsplit']:9.4f} {r['auc_change']:+8.4f} "
              f"{r['pct_of_full_submitted']:9.1f}% {r['pct_of_full_llmsplit']:9.1f}% {('%9.1f%%' % s) if s else '       n/a'}")
    print("\nsmall-budget significance (paper measure):", {k: (round(v["gain_pts"], 2), round(v["sigma"], 2)) for k, v in sig.items()})
    for tag, rows in res["adaptive"].items():
        print(f"\nadaptive {tag}: N | heuristic - random, submitted -> rerun")
        for r in rows:
            print(f"  {r['N']:>4} | {r['heuristic_minus_random_submitted']:+.4f} -> {r['heuristic_minus_random_llmsplit']:+.4f}")
    log_experiment(name="diag_llm_holdout_compare", config={"seed": 42},
                   results={k: res[k] for k in ("reference_auc", "cold_start", "adaptive", "small_budget_significance", "optimiser_sensitivity")},
                   split_info={"n_train": len(tr), "n_test": len(te), "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
