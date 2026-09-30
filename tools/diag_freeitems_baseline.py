#!/usr/bin/env python3
"""Interrogate Table 2 baseline (a), the NCDM with free item parameters (an external review, point 1).

Design-system rules N3 (no un-interrogated baseline claim) and N1 (naive-alternative envelope on the same axes).

Checks, all on the Protocol A (interaction-wise) split of the trained run, Euler job 14849752:
  1. Reproduce the reported test metrics from the saved checkpoint.
  2. Degenerate behaviour: discrimination and difficulty spread, items whose predictions do not vary across LLMs,
     items whose discrimination collapsed to zero, and what those items look like in the training data.
  3. Naive alternatives on the same split and test triplets: global mean, per-LLM mean, per-item mean, and the
     additive logistic combination of the two means. These are what "free item parameters" must beat to mean anything.

Output: cdm_exploration/experiments/v2_freeitems_baseline_diag.json

    python3 tools/diag_freeitems_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.modeling.free_item import FreeItemNCDM
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
CKPT = REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolA_K100_freeitems.pt"
REPORTED = EXP / "v2_ncdm_protocolA_K100_freeitems_results.json"


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "rmse": float(np.sqrt(((y - p) ** 2).mean()))}


def main():
    seed_everything(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    n_llms, n_items = R.shape
    K = Q.shape[1]

    # Same split as tools/run_ncdm_protocolA.py: 80/10/10 over all triplets, random_state 42.
    llm_id = np.repeat(np.arange(n_llms, dtype=np.int32), n_items)
    item_id = np.tile(np.arange(n_items, dtype=np.int32), n_llms)
    y_all = R.ravel().astype(np.float32)
    all_idx = np.arange(len(y_all))
    trvl_idx, te_idx = train_test_split(all_idx, test_size=0.10, random_state=42)
    tr_idx, va_idx = train_test_split(trvl_idx, test_size=10.0 / 90.0, random_state=42)
    ok = verify_splits(trvl_idx, te_idx, expected_seed=42, label="freeitems_protocolA")
    print(f"  train {len(tr_idx):,} val {len(va_idx):,} test {len(te_idx):,}", flush=True)

    # ── 1. Reproduce the reported metrics from the checkpoint ──
    net = FreeItemNCDM(K, n_llms, n_items)
    load_checkpoint(CKPT, net, "cpu")
    net.eval()
    q_t = torch.tensor(Q, dtype=torch.float32)
    preds = np.zeros(len(te_idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(te_idx), 65536):
            idx = te_idx[s:s + 65536]
            i_b = torch.tensor(item_id[idx].astype(np.int64))
            preds[s:s + len(idx)] = net(torch.tensor(llm_id[idx].astype(np.int64)), i_b, q_t[i_b]).numpy()
    y_te = y_all[te_idx]
    got = metrics(y_te, preds)
    rep = json.load(open(REPORTED))
    reproduced = all(abs(got[k] - rep[f"test_{k}"]) < 1e-3 for k in ("auc", "acc", "rmse"))
    print(f"  reproduced from checkpoint: {got} against reported "
          f"{ {k: rep['test_' + k] for k in ('auc', 'acc', 'rmse')} } -> {reproduced}", flush=True)

    # ── 2. Degenerate behaviour ──
    with torch.no_grad():
        alpha = torch.sigmoid(net.e_difficulty.weight).numpy().ravel()          # discrimination, one per item
        d = torch.sigmoid(net.k_difficulty.weight).numpy()                       # difficulty, K per item
        theta = torch.sigmoid(net.student_emb.weight).numpy()
    train_item_acc = np.zeros(n_items)
    cnt = np.bincount(item_id[tr_idx], minlength=n_items)
    train_item_acc = np.bincount(item_id[tr_idx], weights=y_all[tr_idx], minlength=n_items) / np.maximum(cnt, 1)
    d_required = np.array([d[j, Q[j] > 0].mean() for j in range(n_items)])       # difficulty on the item's own skills
    # Predictions for every LLM on a sample of items, to see whether any item is flat across LLMs.
    rng = np.random.default_rng(42)
    sample = rng.choice(n_items, size=500, replace=False)
    flat = []
    with torch.no_grad():
        for j in sample:
            ids = torch.arange(n_llms)
            p = net(ids, torch.full((n_llms,), int(j)), q_t[int(j)].repeat(n_llms, 1)).numpy()
            flat.append((float(p.std()), float(p.mean())))
    flat_std = np.array([f[0] for f in flat])
    degenerate = {
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()), "alpha_median": float(np.median(alpha)),
        "share_alpha_below_001": float((alpha < 0.01).mean()), "share_alpha_above_099": float((alpha > 0.99).mean()),
        "train_acc_of_items_with_alpha_below_001": float(train_item_acc[alpha < 0.01].mean()) if (alpha < 0.01).any() else None,
        "train_acc_spread_of_those_items": float(train_item_acc[alpha < 0.01].std()) if (alpha < 0.01).any() else None,
        "difficulty_mean": float(d_required.mean()), "difficulty_std_across_items": float(d_required.std()),
        "corr_difficulty_vs_train_item_accuracy": float(np.corrcoef(d_required, train_item_acc)[0, 1]),
        "corr_alpha_vs_train_item_accuracy": float(np.corrcoef(alpha, train_item_acc)[0, 1]),
        "sampled_items": len(sample),
        "share_items_flat_across_llms_sd_below_001": float((flat_std < 0.01).mean()),
        "median_prediction_sd_across_llms": float(np.median(flat_std)),
        "theta_mean": float(theta.mean()), "theta_std": float(theta.std()),
    }
    print("  degeneracy:", json.dumps(degenerate, indent=1), flush=True)

    # ── 3. Naive alternatives on the same test triplets (N1 envelope) ──
    g = float(y_all[tr_idx].mean())
    llm_cnt = np.bincount(llm_id[tr_idx], minlength=n_llms)
    llm_mean = np.bincount(llm_id[tr_idx], weights=y_all[tr_idx], minlength=n_llms) / np.maximum(llm_cnt, 1)
    item_mean = train_item_acc
    eps = 1e-6
    logit = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
    naive = {
        "global_mean": metrics(y_te, np.full(len(te_idx), g, dtype=np.float32)),
        "llm_mean": metrics(y_te, llm_mean[llm_id[te_idx]]),
        "item_mean": metrics(y_te, item_mean[item_id[te_idx]]),
        "llm_plus_item_logit": metrics(y_te, 1 / (1 + np.exp(-(logit(llm_mean[llm_id[te_idx]]) + logit(item_mean[item_id[te_idx]]) - logit(np.array(g)))))),
    }
    for k, v in naive.items():
        print(f"  naive {k:>20}: AUC {v['auc']:.4f} acc {v['acc']:.4f} rmse {v['rmse']:.4f}", flush=True)

    others = {}
    for name, f, field in (("skilleval_id", "v2_ncdm_protocolA_K100_results.json", "test_auc"),
                           ("irt_2pl_id", "v2_irt_baseline_protocolA.json", "test_auc")):
        p = EXP / f
        if p.exists():
            others[name] = json.load(open(p))[field]
    print("  for reference:", others, flush=True)

    res = {"experiment": "freeitems_baseline_diag", "checkpoint": str(CKPT.relative_to(REPO)),
           "job": 14849752, "reported": {k: rep[f"test_{k}"] for k in ("auc", "acc", "rmse")},
           "recomputed_from_checkpoint": got, "reproduced": bool(reproduced),
           "degenerate_behaviour": degenerate, "naive_alternatives_same_split": naive,
           "reference_auc": others, "verified": bool(ok and reproduced)}
    (EXP / "v2_freeitems_baseline_diag.json").write_text(json.dumps(res, indent=2))
    log_experiment(name="diag_freeitems_baseline", config={"seed": 42, "checkpoint": str(CKPT.name)},
                   results={k: res[k] for k in ("recomputed_from_checkpoint", "degenerate_behaviour",
                                                 "naive_alternatives_same_split", "reference_auc")},
                   split_info={"n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
                               "random_state": 42}, verified=bool(ok and reproduced))
    print("  wrote v2_freeitems_baseline_diag.json", flush=True)


if __name__ == "__main__":
    main()
