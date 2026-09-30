#!/usr/bin/env python3
"""Two descriptive facts about the inputs of the pipeline, from data files only (no model, no training).

1. Transfer experiment (MuSR and IFEval held out): how many skills are left without any training item, and
   how much of each held-out benchmark involves such skills. The skills themselves exist in the Q-matrix,
   which was built on the full item bank; what they lack is response-based training.
2. Input excerpts: skill extraction and item embeddings of the submitted pipeline used the stored
   200-character `question_preview` of each prompt. How often is that excerpt identical for two items?

Output: cdm_exploration/experiments/v2_transfer_skill_coverage.json

    python3 tools/diag_transfer_skill_coverage.py
"""
import collections, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

D = REPO / "cdm_exploration/data/cdm_ready"
OUT = REPO / "cdm_exploration/experiments/v2_transfer_skill_coverage.json"
HELD_OUT = ["MuSR", "IFEval"]


def main():
    seed_everything(42)
    ck = torch.load(REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt", map_location="cpu", weights_only=False)
    ok = verify_splits(np.array(ck["train_items"]), np.array(ck["test_items"]), expected_seed=42, label="transfer_skill_coverage")
    Q = np.load(D / "qmatrix_v2_K100.npy")
    items = json.load(open(D / "response_matrix_v2_full_items.json"))
    bench = np.array([it["benchmark"] for it in items])
    train = ~np.isin(bench, HELD_OUT)
    untrained = np.where(Q[train].sum(0) == 0)[0]
    trained = np.setdiff1d(np.arange(Q.shape[1]), untrained)
    per = {}
    for b in HELD_OUT:
        m = bench == b
        used = np.where(Q[m].sum(0) > 0)[0]
        per[b] = {"n_items": int(m.sum()), "n_skills_used": int(len(used)),
                  "n_skills_used_without_training_items": int(np.isin(used, untrained).sum()),
                  "pct_item_skill_links_on_untrained_skills": float(100 * Q[m][:, untrained].sum() / Q[m].sum()),
                  "pct_items_touching_an_untrained_skill": float(100 * (Q[m][:, untrained].sum(1) > 0).mean()),
                  "pct_items_with_only_untrained_skills": float(100 * ((Q[m][:, untrained].sum(1) > 0) & (Q[m][:, trained].sum(1) == 0)).mean())}
    prev = [" ".join(it["question_preview"].split()) for it in items]
    cnt = collections.Counter(prev)
    shared = np.array([cnt[p] > 1 for p in prev])
    res = {"experiment": "transfer_skill_coverage", "held_out": HELD_OUT, "n_training_items": int(train.sum()),
           "n_skills_without_training_items": int(len(untrained)), "per_held_out_benchmark": per,
           "input_excerpt": {"max_chars": int(max(len(it["question_preview"]) for it in items)),
                             "n_items": len(prev), "n_distinct_excerpts": len(cnt),
                             "pct_items_sharing_their_excerpt": float(100 * shared.mean()),
                             "pct_sharing_by_benchmark": {b: float(100 * shared[bench == b].mean()) for b in sorted(set(bench))}},
           "verified": bool(ok)}
    OUT.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    log_experiment(name="diag_transfer_skill_coverage", config={"seed": 42, "held_out": HELD_OUT},
                   results={k: res[k] for k in ("n_training_items", "n_skills_without_training_items", "per_held_out_benchmark", "input_excerpt")},
                   split_info={"n_train": len(ck["train_items"]), "n_test": len(ck["test_items"]), "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
