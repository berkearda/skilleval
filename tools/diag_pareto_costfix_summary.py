#!/usr/bin/env python3
"""Headline numbers of the corrected-price cost analysis, as named fields (reads v2_pareto_costfix.json; no model is run)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits
EXP = REPO / "cdm_exploration/experiments"


def at(points, tau, key="threshold"):
    return next(p for p in points if abs(p[key] - tau) < 1e-9)


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="pareto_costfix_summary")
    d = json.load(open(EXP / "v2_pareto_costfix.json")); s = d["strongest"]["accuracy"]
    M = {k: v["corrected_prices"] for k, v in d["methods"].items()}
    B = {k: v["submitted_prices"] for k, v in d["methods"].items()}
    main_pts, irt, knn = M["skilleval_main"]["points"], M["irtnet_d232"]["points"], M["knn_k10"]["points"]
    def best_under(pts, c):
        a = [p["accuracy"] for p in pts if p["cost_pct"] <= c]; return max(a) if a else None
    dominated = sum(1 for p in irt if any((b := best_under(q, p["cost_pct"])) is not None and b >= p["accuracy"] for q in (main_pts, knn)))
    kb = max(knn, key=lambda p: p["accuracy"])
    picks_note = "four LLMs re-priced; see price_corrections in v2_pareto_costfix.json"
    res = {"experiment": "pareto_costfix_summary", "n_llms_repriced": len(d["price_corrections"]), "note": picks_note,
           "main_model": {"parity": M["skilleval_main"]["cheapest_point_at_or_above_strongest"],
                          "tau095": at(main_pts, 0.95), "tau080": at(main_pts, 0.80),
                          "tau095_points_over_strongest": 100 * (at(main_pts, 0.95)["accuracy"] - s),
                          "tau095_cost_reduction_pct": 100 - at(main_pts, 0.95)["cost_pct"],
                          "submitted_prices_tau080": at(B["skilleval_main"]["points"], 0.80)},
           "five_seed": {"parity": d["five_seed"]["corrected_prices"]["cheapest_point_at_or_above_strongest"]},
           "irtnet_d232": {"parity": M["irtnet_d232"]["cheapest_point_at_or_above_strongest"],
                           "n_points_dominated_by_skilleval_or_knn": dominated, "n_points": len(irt)},
           "knn_k10": {"best_point": kb, "best_acc_pct_of_strongest": 100 * kb["accuracy"] / s,
                       "reaches_strongest": M["knn_k10"]["cheapest_point_at_or_above_strongest"] is not None},
           "verified": bool(ok)}
    (EXP / "v2_pareto_costfix_summary.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    log_experiment(name="diag_pareto_costfix_summary", config={"seed": 42}, results={k: res[k] for k in ("main_model", "five_seed", "irtnet_d232", "knn_k10")},
                   split_info={"n_train": len(tr), "n_test": len(te), "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
