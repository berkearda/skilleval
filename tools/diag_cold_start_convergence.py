"""Convergence diagnostic for cold-start.

Question: in the original `tools/cold_start_v2.py`, theta is initialized
to zero and refined by 50 Adam steps at lr=0.01. The trained-LLM theta
distribution lives in raw-space ~[-2, +2] (pop_mean stats from earlier
audit). With Adam at lr=0.01 x 50 steps, theta can move by at most ~0.5
in any direction, which doesn't reach the trained region. Hypothesis:
the AUC trajectory (0.54 -> 0.58 from N=0 to N=500) reflects optimizer
NON-convergence, not a real "few items don't help much" finding.

Test: re-run cold-start with the same zero init but MORE steps and
HIGHER lr. If AUC jumps significantly, the original was undertrained.

We also track the L2 norm of theta_raw at the end of calibration, to
see how far theta drifted from zero.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def calibrate_with_settings(net, R, llm_indices, cal_items_per_llm, q_matrix,
                              text_embs, device, K, lr, steps,
                              return_norm=False):
    """Same as batch_calibrate but with adjustable lr / steps and norm logging."""
    net.eval()
    n_cal = len(llm_indices)
    theta_raw = nn.Parameter(torch.zeros(n_cal, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    all_llm_local, all_items_flat, all_scores = [], [], []
    for local_idx, (llm_idx, cal_items) in enumerate(zip(llm_indices, cal_items_per_llm)):
        cal = cal_items.astype(int)
        all_llm_local.extend([local_idx] * len(cal))
        all_items_flat.extend(cal.tolist())
        all_scores.extend(R[llm_idx, cal].tolist())
    llm_local_t = torch.tensor(all_llm_local, dtype=torch.int64, device=device)
    scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device)

    unique_items = np.unique(all_items_flat)
    item_to_local = {int(it): i for i, it in enumerate(unique_items)}
    te_unique = torch.tensor(text_embs[unique_items], dtype=torch.float32, device=device)
    qr_unique = torch.tensor(q_matrix[unique_items], dtype=torch.float32, device=device)
    with torch.no_grad():
        k_diff_unique = torch.sigmoid(net.k_difficulty_proj(te_unique))
        e_diff_unique = torch.sigmoid(net.e_difficulty_proj(te_unique))
    items_local = torch.tensor([item_to_local[int(it)] for it in all_items_flat],
                                dtype=torch.int64, device=device)

    losses = []
    for step in range(steps):
        stat = torch.sigmoid(theta_raw[llm_local_t])
        k_d = k_diff_unique[items_local]
        e_d = e_diff_unique[items_local]
        qr_d = qr_unique[items_local]
        x = e_d * (stat - k_d) * qr_d
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
        loss = loss_fn(pred, scores_t)
        losses.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if return_norm:
        norm = float(theta_raw.detach().abs().mean())  # mean abs of raw theta
    else:
        norm = None
    return torch.sigmoid(theta_raw).detach().cpu().numpy(), losses, norm


def main() -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint
    from tools.cold_start_v2 import batch_predict

    seed_everything(42)
    data_dir = REPO / "cdm_exploration" / "data" / "cdm_ready"
    device = resolve_device("cpu")

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2,
                                                  random_state=42)
    train_llms, test_llms = train_test_split(np.arange(n_llms), test_size=0.2,
                                                random_state=42)

    # Sub-sample held-out LLMs for speed (this is a diagnostic)
    rng = np.random.RandomState(42)
    test_llms_sub = rng.choice(test_llms, 100, replace=False)

    ckpt_path = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device).eval()

    # Trained-theta reference: what's the typical |theta_raw| of train LLMs?
    with torch.no_grad():
        train_theta_raw = net.student_emb(
            torch.tensor(train_llms[:200], device=device))
    train_theta_norm = float(train_theta_raw.abs().mean())
    print(f"Reference: trained-theta mean |raw value| = {train_theta_norm:.4f}")
    print(f"  (cold-start theta needs to reach this scale to match training)")
    print()

    settings = [
        ("original (lr=0.01, steps=50)",  0.01, 50),
        ("more steps (lr=0.01, steps=500)", 0.01, 500),
        ("higher lr (lr=0.05, steps=50)",   0.05, 50),
        ("both (lr=0.05, steps=500)",       0.05, 500),
        ("very long (lr=0.1, steps=2000)",  0.1, 2000),
    ]

    rows = []
    for label, lr, steps in settings:
        print(f"\n=== {label} ===")
        for N in [10, 100, 500]:
            cal_per_llm = [rng.choice(train_items, size=N, replace=False)
                            for _ in test_llms_sub]
            theta_mat, losses, norm = calibrate_with_settings(
                net, R, test_llms_sub, cal_per_llm, q_matrix, text_embs,
                device, K, lr=lr, steps=steps, return_norm=True)
            preds = batch_predict(net, theta_mat, test_items, q_matrix,
                                    text_embs, device)
            aucs = []
            for i, llm_idx in enumerate(test_llms_sub):
                y = R[llm_idx, test_items.astype(int)]
                if len(np.unique(y)) >= 2:
                    aucs.append(roc_auc_score(y, preds[i]))
            auc_m = float(np.mean(aucs))
            print(f"  N={N:>4}: AUC={auc_m:.4f}  final_loss={losses[-1]:.4f}  "
                    f"|theta_raw| mean={norm:.4f}  (vs trained {train_theta_norm:.4f})")
            rows.append({"setting": label, "lr": lr, "steps": steps, "N": N,
                          "auc": auc_m, "theta_raw_mean_abs": norm,
                          "final_loss": float(losses[-1])})

    out = REPO / "cdm_exploration" / "experiments" / "v2_cold_start_convergence_diag.json"
    json.dump({"reference_trained_theta_norm": train_theta_norm,
                 "n_test_llms": int(len(test_llms_sub)),
                 "results": rows}, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
