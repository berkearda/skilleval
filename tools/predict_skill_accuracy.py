#!/usr/bin/env python3
"""Predicted accuracy of every LLM on every skill, from the main model (an external review, point 2, 2026-09-22).

theta has no fixed location (predictions depend on theta - d only), so "mastered = theta > 0.5" is not identified.
The replacement is the model's predicted accuracy on the skill's items: for LLM i and skill k, the mean predicted
probability of a correct answer over all items assigned to k. Predictions do not change under the shift, so neither
does this score. Observed skill accuracy (no model) is saved next to it as a model-free comparison.

Checks printed and saved:
  - test AUC over the 1,905 test items reproduces the main model's 0.7170 (right checkpoint, eval mode);
  - an example LLM (base Llama-3.2-1B on "Evaluating Truthfulness Of Nested Statements"). The example discussed with
    Berke on 22 Sep was Llama-3.2-1B-Instruct (theta 0.79, predicted 0.37, observed 0.38); it is checked from the saved
    matrix in tools/diag_mastery_definition_compare.py.

Output: cdm_exploration/experiments/v2_predicted_skill_accuracy.npz (pred_skill_acc, obs_skill_acc, theta: 3811 x 100)
        and v2_predicted_skill_accuracy.json (checks, per-skill prevalence under theta and predicted accuracy).

    python tools/predict_skill_accuracy.py --device cuda
    python tools/predict_skill_accuracy.py --device cpu --n_llms 40      # smoke test, writes *_smoke files
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.evaluation.skill_mastery import skill_accuracy
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
MAIN_CKPT = "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"
MAIN_TEST_AUC = 0.7170448            # main model's test AUC (provenance_map.md)
EXAMPLE = ("Llama-3.2-1B", "Evaluating Truthfulness Of Nested Statements")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MAIN_CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=8, help="LLMs per forward pass (each LLM x all items)")
    ap.add_argument("--n_llms", type=int, default=0, help="smoke test: first N LLMs only")
    args = ap.parse_args()

    seed_everything(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    text = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    llm_names = json.load(open(DATA / "response_matrix_v2_full_llms.json"))
    skill_names = json.load(open(DATA / "cluster_labels_v2_K100.json"))
    n_all, n_items = R.shape
    K = q_matrix.shape[1]

    net = TextConditionedNet(K, n_all, text.shape[1])
    load_checkpoint(REPO / args.ckpt, net, "cpu")
    net.to(args.device).eval()
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).cpu().numpy()

    n_llms = args.n_llms or n_all
    smoke = n_llms < n_all
    text_t = torch.tensor(text, dtype=torch.float32, device=args.device)
    q_t = torch.tensor(q_matrix, dtype=torch.float32, device=args.device)
    P = np.zeros((n_llms, n_items), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n_llms, args.chunk):
            ids = torch.arange(s, min(s + args.chunk, n_llms), device=args.device)
            c = len(ids)
            out = net(ids.repeat_interleave(n_items), text_t.repeat(c, 1), q_t.repeat(c, 1))
            P[s:s + c] = out.view(c, n_items).cpu().numpy()
    print(f"predicted {n_llms} x {n_items} in {time.time() - t0:.0f}s; P in [{P.min():.4f}, {P.max():.4f}]", flush=True)
    assert np.isfinite(P).all() and 0 <= P.min() and P.max() <= 1

    R = R[:n_llms]
    theta = theta[:n_llms]
    pred = skill_accuracy(P, q_matrix)
    obs = skill_accuracy(R, q_matrix)

    # Check 1: the main model's test AUC over the item split used in training (Protocol B, random_state 42).
    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    ok = verify_splits(train_items, test_items, expected_seed=42, label="predict_skill_accuracy")
    test_auc = float(roc_auc_score(R[:, test_items].ravel(), P[:, test_items].ravel()))
    auc_ok = smoke or abs(test_auc - MAIN_TEST_AUC) < 5e-4
    print(f"test AUC {test_auc:.4f} (main model {MAIN_TEST_AUC:.4f}) -> {'PASS' if auc_ok else 'FAIL'}", flush=True)

    # Check 2: the example from the review note.
    k_ex = next(int(k) for k, v in skill_names.items() if v == EXAMPLE[1])
    i_ex = [i for i, n in enumerate(llm_names[:n_llms]) if n.split("__")[-1] == EXAMPLE[0]]
    example = [{"llm": llm_names[i], "skill": EXAMPLE[1], "theta": float(theta[i, k_ex]),
                "predicted": float(pred[i, k_ex]), "observed": float(obs[i, k_ex])} for i in i_ex]
    print("example:", example, flush=True)

    # Descriptive comparison of the two definitions at 0.5.
    prev_theta = (theta > 0.5).mean(axis=0)
    prev_pred = (pred > 0.5).mean(axis=0)
    disagree = float(((theta > 0.5) != (pred > 0.5)).mean())
    r_pred_obs = float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])
    print(f"cells where theta>0.5 and predicted>0.5 disagree: {100 * disagree:.1f}%; "
          f"r(predicted, observed) over cells = {r_pred_obs:.3f}", flush=True)

    tag = "_smoke" if smoke else ""
    np.savez_compressed(EXP / f"v2_predicted_skill_accuracy{tag}.npz", pred_skill_acc=pred.astype(np.float32),
                        obs_skill_acc=obs.astype(np.float32), theta=theta.astype(np.float32),
                        n_items_per_skill=(q_matrix > 0).sum(axis=0))
    res = {"experiment": "predict_skill_accuracy", "checkpoint": args.ckpt, "n_llms": n_llms, "n_items": n_items,
           "K": K, "definition": "mean predicted P(correct) over all items assigned to the skill (train and test items)",
           "test_auc": test_auc, "test_auc_expected": MAIN_TEST_AUC, "test_auc_pass": bool(auc_ok),
           "example": example, "cells_disagree_at_0.5": disagree, "r_predicted_observed_cells": r_pred_obs,
           "prevalence_theta_0.5": prev_theta.tolist(), "prevalence_predicted_0.5": prev_pred.tolist(),
           "verified": bool(ok and auc_ok)}
    (EXP / f"v2_predicted_skill_accuracy{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"saved v2_predicted_skill_accuracy{tag}.npz/.json", flush=True)
    if not smoke:
        log_experiment(name="predict_skill_accuracy", config={"checkpoint": args.ckpt, "seed": 42},
                       results={k: res[k] for k in ("test_auc", "example", "cells_disagree_at_0.5",
                                                    "r_predicted_observed_cells")},
                       split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                                   "random_state": 42}, verified=bool(ok and auc_ok))
    if not auc_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
