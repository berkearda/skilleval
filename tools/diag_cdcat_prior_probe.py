#!/usr/bin/env python3
"""Probe before redesigning CD-CAT (24 Sep 2026): how well does a profile predict a new LLM's answers with ZERO answers,
when it starts from the average profile of the training LLMs instead of 0.5 on every skill?

The cold-start and CD-CAT runs (T-123, network text_conditioned_protocolB_llmsplit3048.pt) start every skill at
theta = 0.5 (raw 0); with no answers this gives mean test AUC 0.547 over the 763 held-out LLMs (cold3048.N0_auc_mean).
Theta has no fixed location per skill (the review's point 2), so 0.5 is an arbitrary start. The natural start in Bayesian
multidimensional adaptive testing is the population mean of the training LLMs. No model is trained here.

Output: cdm_exploration/experiments/v2_cdcat_prior_probe.json

    python3 tools/diag_cdcat_prior_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
CKPT = REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB_llmsplit3048.pt"


def predict_with_theta(net, mastery, items, q_matrix, text_embs, device):
    """Copy of tools/run_adaptive_testing_v2.py::predict_with_theta (that module needs matplotlib)."""
    idx = items.astype(int)
    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)
    stat = torch.tensor(mastery, dtype=torch.float32, device=device).unsqueeze(0).expand(len(idx), -1)
    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))
        e_d = torch.sigmoid(net.e_difficulty_proj(te))
        x = e_d * (stat - k_d) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
    return pred.cpu().numpy()


def main():
    seed_everything(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q.shape[1]
    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    train_llms, test_llms = train_test_split(np.arange(n_llms), test_size=0.2, random_state=42)
    ok = verify_splits(train_items, test_items, expected_seed=42, label="cdcat_prior_probe")
    assert len(train_llms) == 3048 and len(test_llms) == 763
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(CKPT, net, "cpu")
    net.eval()
    theta_train = torch.sigmoid(net.student_emb.weight[torch.tensor(train_llms)]).detach().numpy()
    starts = {"all_0.5": np.full(K, 0.5), "population_mean": theta_train.mean(axis=0)}
    res = {"experiment": "cdcat_prior_probe", "checkpoint": CKPT.name, "n_train_llms": len(train_llms), "starts": {}}
    for name, theta in starts.items():
        p = predict_with_theta(net, theta, test_items, q, emb, "cpu")      # same prediction for every new LLM
        aucs = np.array([roc_auc_score(R[l, test_items], p) for l in test_llms])
        res["starts"][name] = {"auc_mean_all_763": float(aucs.mean()), "auc_std_all_763": float(aucs.std()),
                               "auc_mean_first_50": float(aucs[:50].mean())}
        print(f"{name:16s} zero answers: mean test AUC {aucs.mean():.4f} over 763 held-out LLMs "
              f"({aucs[:50].mean():.4f} over the 50 used in CD-CAT)")
    ref = json.load(open(EXP / "v2_cold_start_fixed_llmsplit.json"))
    n0 = [r for r in ref["summary"] if r["N"] == 0][0]["auc_mean"]
    assert abs(res["starts"]["all_0.5"]["auc_mean_all_763"] - n0) < 0.005, (res["starts"]["all_0.5"], n0)
    res["reproduces_registered_N0"] = n0
    res["verified"] = bool(ok)
    (EXP / "v2_cdcat_prior_probe.json").write_text(json.dumps(res, indent=2))
    log_experiment(name="diag_cdcat_prior_probe", config={"checkpoint": CKPT.name},
                   results=res["starts"], split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                                                      "n_train_llms": 3048, "n_test_llms": 763, "random_state": 42},
                   verified=bool(ok))


if __name__ == "__main__":
    main()
