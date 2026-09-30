#!/usr/bin/env python3
"""Table 2 baseline (b), the text-conditioned 2PL: one summary of the runs of Euler job 14890464 (an external review).

Reads v2_irt2pl_textitems_protocolA_s42.json (ID, seed 42) and v2_irt2pl_textitems_protocolB_s{42..46}.json (OOD, five
seeds), the 15-epoch schedule shared by every Table 2 model (decisions 2026-09-23: the 60-epoch convergence check stays a
logged sensitivity, not the reported value). Writes cdm_exploration/experiments/v2_irt2pl_textitems_summary.json with
the ID values and the OOD mean and SD (population SD over seeds, as for SkillEval's five-seed values).

    python3 tools/diag_irt2pl_textitems_summary.py
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
SEEDS = [42, 43, 44, 45, 46]


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="irt2pl_textitems_summary")
    a = json.load(open(EXP / "v2_irt2pl_textitems_protocolA_s42.json"))
    bs = [json.load(open(EXP / f"v2_irt2pl_textitems_protocolB_s{s}.json")) for s in SEEDS]
    assert a["epochs_max"] == 15 and all(b["epochs_max"] == 15 for b in bs), "15-epoch runs expected"
    assert all(b["n_test_items"] == len(te) for b in bs)
    res = {"experiment": "irt2pl_textitems_summary", "model": "TextItemIRT2PL", "epochs_max": 15, "lr": a["lr"],
           "batch_size": a["batch_size"],
           "id": {k: a[k] for k in ("test_auc", "test_acc", "test_rmse", "seed")},
           "ood": {"seeds": SEEDS}}
    for k in ("test_auc", "test_acc", "test_rmse"):
        v = np.array([b[k] for b in bs])
        res["ood"][f"{k}_mean"], res["ood"][f"{k}_std"], res["ood"][f"{k}_per_seed"] = float(v.mean()), float(v.std()), v.tolist()
    res["verified"] = bool(ok and a["verified"] and all(b["verified"] for b in bs))
    (EXP / "v2_irt2pl_textitems_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"id": res["id"], "ood": {k: v for k, v in res["ood"].items() if not k.endswith("per_seed")}}, indent=1))
    log_experiment(name="irt2pl_textitems_summary", config={"epochs_max": 15, "seeds_ood": SEEDS},
                   results={"id": res["id"], "ood": res["ood"]},
                   split_info={"n_train_items": len(tr), "n_test_items": len(te), "random_state": 42},
                   verified=res["verified"])


if __name__ == "__main__":
    main()
