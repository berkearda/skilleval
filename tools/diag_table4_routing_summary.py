#!/usr/bin/env python3
"""Table 4 (routing) from existing results only: no model is run (23 Sep 2026).

Collects Acc@k on the 1,905 held-out items for every router already evaluated:
  random, strongest single LLM   v2_baselines.json and the main model's log entry
  IRT 2PL                        log entry irt_baseline_v2 (ranks LLMs by theta)
  KNN (10 nearest items)         v2_knn_baseline.json
  SkillEval main run             log entry train_expanded (3,811 LLMs; batch 1024; the model behind every other analysis)
  SkillEval, three reruns        log entries train_expanded_seed{42,43,44}_K100full (same recipe; mean and sample SD)
  IRTNet d=232/512/1024          v2_irtnet_d{d}_s{42,43,44}.json, the Table 2 OOD runs (mean and sample SD; added
                                 25 Sep 2026 so the routing table has the multi-seed learned baseline, d=1024 printed)
All routers share the item split (80/20, random_state 42) and the Acc@k definition of
cdmeval/evaluation/baselines.py::evaluate_rankings.

Output: cdm_exploration/experiments/v2_table4_routing_summary.json

    python3 tools/diag_table4_routing_summary.py
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
KS = [1, 3, 5, 10]


def log_entry(pred):
    log = json.load(open(EXP / "experiment_log.json"))
    entries = log if isinstance(log, list) else log.get("experiments", log.get("entries", []))
    hits = [e for e in entries if pred(e)]
    assert hits, "log entry not found"
    return hits[0]["results"]


def main():
    seed_everything(42)
    tr, te = train_test_split(np.arange(9523), test_size=0.2, random_state=42)
    ok = verify_splits(tr, te, expected_seed=42, label="table4_routing_summary")
    base = json.load(open(EXP / "v2_baselines.json"))
    knn = json.load(open(EXP / "v2_knn_baseline.json"))
    main_run = log_entry(lambda e: e["experiment"] == "train_expanded" and e["config"].get("n_llms") == 3811)
    irt = log_entry(lambda e: e["experiment"] == "irt_baseline_v2")
    reruns = [log_entry(lambda e, s=s: e["experiment"] == f"train_expanded_seed{s}_K100full") for s in (42, 43, 44)]
    assert base["n_test_items"] == knn["n_test_items"] == len(te) == 1905

    res = {"experiment": "table4_routing_summary", "n_test_items": 1905, "n_llms": 3811, "rows": {}}
    res["rows"]["random"] = {"acc@1": base["random_acc1"]}
    res["rows"]["strongest"] = {"acc@1": main_run["strongest_acc1"], "llm": base["strongest_model"]}
    res["rows"]["irt2pl"] = {f"acc@{k}": irt[f"routing_acc@{k}"] for k in KS}
    res["rows"]["knn"] = {f"acc@{k}": knn["routing"][f"acc@{k}"] for k in KS}
    res["rows"]["skilleval_main"] = {f"acc@{k}": main_run[f"acc@{k}"] for k in KS}
    res["rows"]["skilleval_reruns"] = {}
    for k in KS:
        v = np.array([r[f"acc@{k}"] for r in reruns])
        res["rows"]["skilleval_reruns"][f"acc@{k}_mean"] = float(v.mean())
        res["rows"]["skilleval_reruns"][f"acc@{k}_sd_sample"] = float(v.std(ddof=1))
        res["rows"]["skilleval_reruns"][f"acc@{k}_values"] = v.tolist()
    for d in (232, 512, 1024):
        runs = [json.load(open(EXP / f"v2_irtnet_d{d}_s{s}.json")) for s in (42, 43, 44)]
        assert all(r["n_test_items"] == r["n_test_items_evaluated"] == 1905 for r in runs)
        row = res["rows"][f"irtnet{d}_seeds"] = {}
        for k in KS:
            v = np.array([r[f"routing_acc@{k}"] for r in runs])
            row[f"acc@{k}_mean"] = float(v.mean())
            row[f"acc@{k}_sd_sample"] = float(v.std(ddof=1))
            row[f"acc@{k}_values"] = v.tolist()
    assert abs(res["rows"]["strongest"]["acc@1"] - base["strongest_acc1"]) < 1e-12
    res["verified"] = bool(ok)
    (EXP / "v2_table4_routing_summary.json").write_text(json.dumps(res, indent=2))
    for name, row in res["rows"].items():
        print(f"{name:18s}", {k: round(100 * v, 2) for k, v in row.items() if isinstance(v, float)})
    log_experiment(name="diag_table4_routing_summary", config={"source": "existing result files and log entries"},
                   results=res["rows"], split_info={"n_train_items": len(tr), "n_test_items": len(te), "random_state": 42},
                   verified=res["verified"])


if __name__ == "__main__":
    main()
