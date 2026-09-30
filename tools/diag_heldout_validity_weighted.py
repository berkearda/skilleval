#!/usr/bin/env python3
"""Held-out construct validity of Theta without a skill cut-off (five seeds).

For each LLM, correlate its mastery profile theta_i with its accuracy profile on HELD-OUT items,
across skills. Skills differ a lot in how many held-out items they have (1 to several hundred), so
three ways of counting the skills are reported side by side:

  cutoff10_unweighted  skills with >= 10 held-out items, equal weight  (the reported value, 0.6816)
  all_unweighted       every skill with >= 1 held-out item, equal weight
  all_weighted         every skill with >= 1 held-out item, weight = number of held-out items

Null control: the same weighted statistic after shuffling which theta column goes with which skill.

Output: cdm_exploration/experiments/v2_heldout_validity_weighted.json

    python3 tools/diag_heldout_validity_weighted.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

DATA = REPO / "cdm_exploration/data/cdm_ready"
MS = REPO / "cdm_exploration/checkpoints/multi_seed"
EXP = REPO / "cdm_exploration/experiments"
SEEDS = [42, 43, 44, 45, 46]
N_PERM = 50
REPORTED_VALUE = 0.6815985964705945        # v2_final_circularity_checks.json : C_per_llm_profile.raw_mean_r


def weighted_corr_rows(A, B, w):
    """Weighted Pearson r between matching rows of A and B (one value per LLM)."""
    w = w / w.sum()
    ca = A - (A * w).sum(1, keepdims=True)
    cb = B - (B * w).sum(1, keepdims=True)
    return (w * ca * cb).sum(1) / np.sqrt((w * ca * ca).sum(1) * (w * cb * cb).sum(1))


def main():
    seed_everything(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    ck0 = torch.load(MS / "text_conditioned_seed_42.pt", map_location="cpu", weights_only=False)
    train_items = np.array(sorted(ck0["train_items"]), dtype=int)
    test_items = np.array(sorted(ck0["test_items"]), dtype=int)
    ok = verify_splits(train_items, test_items, expected_seed=42, label="heldout_validity_weighted")

    n_ho = np.array([int((Q[test_items, k] > 0).sum()) for k in range(Q.shape[1])])
    thetas = {s: torch.sigmoid(torch.load(MS / f"text_conditioned_seed_{s}.pt", map_location="cpu",
                                          weights_only=False)["model_state_dict"]["student_emb.weight"]).numpy()
              for s in SEEDS}

    def stat(skills, w, center=False, perm=None):
        acc = np.column_stack([R[:, test_items[Q[test_items, k] > 0]].mean(axis=1) for k in skills])
        vals = []
        for s in SEEDS:
            th = thetas[s][:, skills if perm is None else perm]
            a, b = (th - th.mean(0), acc - acc.mean(0)) if center else (th, acc)
            vals.append(float(np.nanmean(weighted_corr_rows(a, b, w.astype(float)))))
        return {"mean": float(np.mean(vals)), "std_across_seeds": float(np.std(vals)), "per_seed": vals}

    k10, kall = np.where(n_ho >= 10)[0], np.where(n_ho >= 1)[0]
    w_all = n_ho[kall]
    variants = {
        "cutoff10_unweighted": stat(k10, np.ones(len(k10))),
        "all_unweighted": stat(kall, np.ones(len(kall))),
        "all_weighted": stat(kall, w_all),
        "all_weighted_skill_centered": stat(kall, w_all, center=True),
    }
    assert abs(variants["cutoff10_unweighted"]["mean"] - REPORTED_VALUE) < 1e-9, "does not reproduce the reported value"

    rng = np.random.default_rng(42)
    placebo = [stat(kall, w_all, perm=kall[rng.permutation(len(kall))])["mean"] for _ in range(N_PERM)]
    real = variants["all_weighted"]["mean"]
    out = {
        "experiment": "heldout_validity_weighted", "seeds": SEEDS,
        "n_skills_cutoff10": int(len(k10)), "n_skills_any_heldout": int(len(kall)),
        "n_skills_1_to_3_heldout": int(((n_ho >= 1) & (n_ho <= 3)).sum()),
        "effective_n_skills_under_weights": float(w_all.sum() ** 2 / (w_all.astype(float) ** 2).sum()),
        "largest_skill_weight_share": float(w_all.max() / w_all.sum()),
        "variants": variants,
        "placebo_shuffled_skills_all_weighted": {"n_perm": N_PERM, "mean": float(np.mean(placebo)),
                                                 "std": float(np.std(placebo)), "max": float(np.max(placebo)),
                                                 "z": float((real - np.mean(placebo)) / np.std(placebo))},
        "reproduces_rebuttal_value": True, "verified": bool(ok),
    }
    (EXP / "v2_heldout_validity_weighted.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=2))
    for k, v in variants.items():
        print(f"  {k:30s} r = {v['mean']:.4f} +/- {v['std_across_seeds']:.4f}")
    log_experiment(name="diag_heldout_validity_weighted",
                   config={"seed": 42, "seeds": SEEDS, "n_perm": N_PERM, "checkpoints": "multi_seed/text_conditioned_seed_*.pt"},
                   results={"variants": {k: {"mean": v["mean"], "std": v["std_across_seeds"]} for k, v in variants.items()},
                            "placebo": out["placebo_shuffled_skills_all_weighted"],
                            "effective_n_skills": out["effective_n_skills_under_weights"]},
                   split_info={"n_train": len(train_items), "n_test": len(test_items), "random_state": 42},
                   verified=bool(ok))


if __name__ == "__main__":
    main()
