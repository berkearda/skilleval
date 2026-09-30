#!/usr/bin/env python3
"""Numbers printed in Section 5 (base vs instruction-tuned) that were never registered (T-132 "register before use").

Reads v2_alignment_tax.json only (no model run) and stores the counts the text states, so the registry can read them:
skills with mean difference above +0.01 / below -0.01, Bonferroni-significant skills and their signs, the largest
significant decrease, and the per-benchmark means and p-values of Figure 4(a) (skills grouped by primary benchmark).

Output: cdm_exploration/experiments/v2_aligntax_text_numbers.json

    python3 tools/diag_aligntax_text_numbers.py
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


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="aligntax_text_numbers")
    d = json.load(open(EXP / "v2_alignment_tax.json"))
    st = d["statistical_tests"]
    per = st["per_skill"]
    md = np.array([s["mean_delta"] for s in per])
    sig = [s for s in per if s["sig_bonferroni"]]
    res = {"n_pairs": d["n_pairs"],
           "n_skills_up_001": int((md > 0.01).sum()), "n_skills_down_001": int((md < -0.01).sum()),
           "n_sig_bonferroni": len(sig), "n_sig_positive": sum(s["mean_delta"] > 0 for s in sig),
           "n_sig_negative": sum(s["mean_delta"] < 0 for s in sig),
           "largest_sig_decrease_abs": float(-min(s["mean_delta"] for s in sig if s["mean_delta"] < 0)),
           "per_benchmark": st["per_benchmark_sem"], "verified": bool(ok)}
    assert res["n_skills_up_001"] == d["skills_improved"] and res["n_skills_down_001"] == d["skills_degraded"]
    assert res["n_sig_bonferroni"] == st["n_significant_bonferroni"]
    (EXP / "v2_aligntax_text_numbers.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_benchmark"}, indent=1))
    log_experiment(name="diag_aligntax_text_numbers", config={"source": "v2_alignment_tax.json"},
                   results={k: v for k, v in res.items() if k != "verified"},
                   split_info={"n_train_items": len(tr), "n_test_items": len(te), "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
