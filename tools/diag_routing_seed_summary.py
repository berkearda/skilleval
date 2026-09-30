#!/usr/bin/env python3
"""Routing Acc@1 across seeds, kept apart by training setting (no training; reads logged results only).

  main configuration : batch size 1024, seeds 42-44, reruns of 12 Sep 2026 (experiment_log entries
                       train_expanded_seed{42,43,44}_K100full); the paper's 0.657 is the 29 March run of this
                       configuration with seed 42
  five-seed study    : batch size 64, seeds 42-46, April 2026 (v2_multi_seed.json)

Both the sample standard deviation (ddof = 1) and the population one are written, because the submitted paper
printed population values (0.017, 0.0007).

Output: cdm_exploration/experiments/v2_routing_seed_summary.json
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

EXP = REPO / "cdm_exploration/experiments"
CK = REPO / "cdm_exploration/checkpoints"
RERUNS = {42: "expanded/text_conditioned_protocolB_K100full.pt", 43: "multi_seed/seed_43_K100full.pt",
          44: "multi_seed/seed_44_K100full.pt"}


def summary(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "values": [float(x) for x in v], "mean": float(v.mean()),
            "sd_sample": float(v.std(ddof=1)), "sd_population": float(v.std())}


def main():
    seed_everything(42)
    main_ck = torch.load(CK / "expanded/text_conditioned_protocolB.pt", map_location="cpu", weights_only=False)
    ok = verify_splits(np.array(main_ck["train_items"]), np.array(main_ck["test_items"]), expected_seed=42,
                       label="routing_seed_summary")
    log = json.load(open(EXP / "experiment_log.json"))
    acc, auc, same = [], [], {}
    for s, rel in RERUNS.items():
        e = [x for x in log if x["experiment"] == f"train_expanded_seed{s}_K100full"][-1]
        assert e["config"].get("qmatrix") == "qmatrix_v2_K100.npy" and e["config"].get("n_llms") == 3811
        acc.append(e["results"]["acc@1"]); auc.append(e["results"]["test_auc"])
        c = torch.load(CK / rel, map_location="cpu", weights_only=False)
        same[s] = bool(all(c["config"].get(k) == v for k, v in main_ck["config"].items())
                       and sorted(c["train_items"]) == sorted(main_ck["train_items"])
                       and sorted(c["test_items"]) == sorted(main_ck["test_items"])
                       and all(c["model_state_dict"][k].shape == t.shape for k, t in main_ck["model_state_dict"].items()))
    five = json.load(open(EXP / "v2_multi_seed.json"))
    main_entry = next(x for x in log if x["experiment"] == "train_expanded" and x["config"].get("n_llms") == 3811)
    res = {"experiment": "routing_seed_summary",
           "main_model_29mar": {"acc1": main_entry["results"]["acc@1"], "test_auc": main_entry["results"]["test_auc"]},
           "batch1024_reruns": {"seeds": list(RERUNS), "same_config_split_and_shapes_as_main_model": same,
                                "acc1": summary(acc), "test_auc": summary(auc)},
           "batch64_five_seed": {"seeds": five["seeds"], "acc1": summary(five["routing_acc1_per_seed"]),
                                 "test_auc": summary(five["test_auc_per_seed"])},
           "verified": bool(ok and all(same.values()))}
    (EXP / "v2_routing_seed_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    log_experiment(name="diag_routing_seed_summary", config={"seed": 42, "reruns": RERUNS},
                   results={k: res[k] for k in ("main_model_29mar", "batch1024_reruns", "batch64_five_seed")},
                   split_info={"n_train": len(main_ck["train_items"]), "n_test": len(main_ck["test_items"]), "random_state": 42},
                   verified=res["verified"])


if __name__ == "__main__":
    main()
