"""Cold-start evaluation on v2 full dataset (3811 LLMs x 9523 items).

Batched implementation: fits all held-out LLMs simultaneously per calibration size.

Usage:
    python tools/cold_start_v2.py device=cuda
    python tools/cold_start_v2.py device=cpu
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def batch_predict(net, theta_matrix, item_indices, q_matrix, text_embs, device,
                  batch_size=4096):
    """Predict P(correct) for multiple LLMs on multiple items.

    Args:
        net: Trained TextConditionedNet (frozen, on device).
        theta_matrix: (n_llms, K) mastery values in [0,1] as numpy.
        item_indices: (n_items,) item indices to predict on.
        q_matrix: (total_items, K) full Q-matrix.
        text_embs: (total_items, d) full text embeddings.
        device: torch device.
        batch_size: max predictions per forward pass.

    Returns:
        (n_llms, n_items) numpy array of P(correct).
    """
    net.eval()
    n_llms = theta_matrix.shape[0]
    items = item_indices.astype(int)
    n_items = len(items)

    # Precompute item difficulty (shared across all LLMs)
    te = torch.tensor(text_embs[items], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[items], dtype=torch.float32, device=device)

    with torch.no_grad():
        k_diff = torch.sigmoid(net.k_difficulty_proj(te))  # (n_items, K)
        e_diff = torch.sigmoid(net.e_difficulty_proj(te))   # (n_items, 1)

    theta_t = torch.tensor(theta_matrix, dtype=torch.float32, device=device)  # (n_llms, K)
    all_preds = torch.zeros(n_llms, n_items, device=device)

    with torch.no_grad():
        for li in range(0, n_llms, max(1, batch_size // n_items)):
            li_end = min(li + max(1, batch_size // n_items), n_llms)
            batch_theta = theta_t[li:li_end]  # (batch_llms, K)

            for si in range(li, li_end):
                stat = theta_t[si].unsqueeze(0).expand(n_items, -1)  # (n_items, K)
                x = e_diff * (stat - k_diff) * qr
                h1 = torch.sigmoid(net.prednet_full1(x))
                h2 = torch.sigmoid(net.prednet_full2(h1))
                pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
                all_preds[si] = pred

    return all_preds.cpu().numpy()


def batch_calibrate(net, R, llm_indices, cal_items_per_llm, q_matrix, text_embs,
                    device, K, lr=0.01, steps=50):
    """Fit theta for multiple LLMs simultaneously.

    Args:
        net: Frozen TextConditionedNet.
        R: (total_llms, total_items) response matrix.
        llm_indices: (n_cal_llms,) indices of LLMs to calibrate.
        cal_items_per_llm: list of arrays, cal_items_per_llm[i] = item indices for LLM i.
        q_matrix: (total_items, K).
        text_embs: (total_items, d).
        device: torch device.
        K: number of skills.
        lr: learning rate.
        steps: SGD steps.

    Returns:
        (n_cal_llms, K) mastery matrix in [0,1].
    """
    net.eval()
    n_cal = len(llm_indices)

    # Fresh theta for all calibration LLMs
    theta_raw = nn.Parameter(torch.zeros(n_cal, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    # Precompute per-LLM calibration data
    # Pack into flat tensors: (total_cal_pairs,)
    all_llm_local = []   # local index 0..n_cal-1
    all_items_flat = []
    all_scores = []

    for local_idx, (llm_idx, cal_items) in enumerate(zip(llm_indices, cal_items_per_llm)):
        cal = cal_items.astype(int)
        n = len(cal)
        all_llm_local.extend([local_idx] * n)
        all_items_flat.extend(cal.tolist())
        all_scores.extend(R[llm_idx, cal].tolist())

    llm_local_t = torch.tensor(all_llm_local, dtype=torch.int64, device=device)
    items_t = torch.tensor(all_items_flat, dtype=torch.int64, device=device)
    scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device)

    # Precompute item features for all calibration items
    unique_items = np.unique(all_items_flat)
    item_to_local = {int(it): i for i, it in enumerate(unique_items)}
    te_unique = torch.tensor(text_embs[unique_items], dtype=torch.float32, device=device)
    qr_unique = torch.tensor(q_matrix[unique_items], dtype=torch.float32, device=device)

    with torch.no_grad():
        k_diff_unique = torch.sigmoid(net.k_difficulty_proj(te_unique))
        e_diff_unique = torch.sigmoid(net.e_difficulty_proj(te_unique))

    # Map flat item indices to local unique indices
    items_local = torch.tensor([item_to_local[int(it)] for it in all_items_flat],
                               dtype=torch.int64, device=device)

    for step in range(steps):
        stat = torch.sigmoid(theta_raw[llm_local_t])           # (total_pairs, K)
        k_d = k_diff_unique[items_local]                        # (total_pairs, K)
        e_d = e_diff_unique[items_local]                        # (total_pairs, 1)
        qr_d = qr_unique[items_local]                           # (total_pairs, K)

        x = e_d * (stat - k_d) * qr_d
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)

        loss = loss_fn(pred, scores_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(theta_raw).detach().cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={K}", flush=True)

    # ── Splits ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    print(f"  Items: {len(train_items)} train, {len(test_items)} test", flush=True)
    print(f"  LLMs: {len(train_llms)} train, {len(test_llms)} held-out", flush=True)

    # ── Load model ──
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device)
    net.eval()

    # ── Full-training baseline ──
    print("\nComputing full-training baseline (50 train LLMs)...", flush=True)
    sample_train = train_llms[:50]
    with torch.no_grad():
        raw = net.student_emb(torch.tensor(sample_train, device=device)).cpu().numpy()
    full_mastery = 1.0 / (1.0 + np.exp(-raw))  # (50, K)
    full_preds = batch_predict(net, full_mastery, test_items, q_matrix, text_embs, device)

    full_aucs = []
    for i, llm_idx in enumerate(sample_train):
        y_true = R[llm_idx, test_items.astype(int)]
        if len(np.unique(y_true)) >= 2:
            full_aucs.append(roc_auc_score(y_true, full_preds[i]))
    full_auc = float(np.mean(full_aucs))
    print(f"  Full-training AUC: {full_auc:.4f}", flush=True)

    # ── Cold-start evaluation ──
    cal_sizes = [0, 1, 5, 10, 50, 100, 500]
    n_repeats = 3
    rng = np.random.RandomState(42)

    print(f"\nCold-start: {len(test_llms)} LLMs, sizes={cal_sizes}, repeats={n_repeats}", flush=True)

    summary = []
    for N in cal_sizes:
        print(f"\n  N={N}...", end=" ", flush=True)
        reps = 1 if N == 0 else n_repeats
        all_aucs = []
        all_accs = []

        for rep in range(reps):
            if N == 0:
                theta_mat = np.full((len(test_llms), K), 0.5)
            else:
                cal_per_llm = [rng.choice(train_items, size=min(N, len(train_items)),
                                          replace=False) for _ in test_llms]
                theta_mat = batch_calibrate(
                    net, R, test_llms, cal_per_llm, q_matrix, text_embs,
                    device, K, lr=0.01, steps=50,
                )

            # Batch predict all held-out LLMs on test items
            preds = batch_predict(net, theta_mat, test_items, q_matrix, text_embs, device)

            # Compute per-LLM AUC
            for i, llm_idx in enumerate(test_llms):
                y_true = R[llm_idx, test_items.astype(int)]
                if len(np.unique(y_true)) < 2:
                    continue
                auc = roc_auc_score(y_true, preds[i])
                acc = accuracy_score(y_true, (preds[i] >= 0.5).astype(int))
                all_aucs.append(auc)
                all_accs.append(acc)

        auc_m, auc_s = np.mean(all_aucs), np.std(all_aucs)
        acc_m = np.mean(all_accs)
        pct = auc_m / full_auc * 100
        print(f"AUC={auc_m:.4f}±{auc_s:.4f}, Acc={acc_m:.4f}, {pct:.1f}% of full", flush=True)
        summary.append({"N": N, "auc_mean": float(auc_m), "auc_std": float(auc_s),
                         "acc_mean": float(acc_m), "pct_of_full": float(pct)})

    # ── Summary table ──
    print(f"\n{'='*70}", flush=True)
    print(f"COLD-START RESULTS (v2, {len(test_llms)} held-out LLMs)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>6} {'AUC mean':>10} {'AUC std':>10} {'Acc mean':>10} {'% of full':>10}", flush=True)
    print("-" * 70, flush=True)
    for s in summary:
        print(f"{s['N']:>6} {s['auc_mean']:>10.4f} {s['auc_std']:>10.4f} "
              f"{s['acc_mean']:>10.4f} {s['pct_of_full']:>9.1f}%", flush=True)
    print(f"\n  Full-training AUC: {full_auc:.4f}", flush=True)

    # ── Figure ──
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = [s["N"] for s in summary]
    aucs = [s["auc_mean"] for s in summary]
    stds = [s["auc_std"] for s in summary]

    # Handle N=0 on log scale by plotting at x=0.8
    plot_ns = [max(n, 0.8) for n in ns]
    ax.fill_between(plot_ns, [a - s for a, s in zip(aucs, stds)],
                    [a + s for a, s in zip(aucs, stds)], alpha=0.2, color="#4C72B0")
    ax.plot(plot_ns, aucs, "o-", color="#4C72B0", lw=2, markersize=7, label="Cold-start AUC")
    ax.axhline(full_auc, ls="--", color="#55A868", lw=1.5, label=f"Full training ({full_auc:.3f})")
    ax.axhline(0.5, ls=":", color="#999999", lw=1, label="Random guess")
    threshold_90 = full_auc * 0.9
    ax.axhline(threshold_90, ls="--", color="#DD8452", lw=1, alpha=0.7,
               label=f"90% of full ({threshold_90:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Number of calibration items", fontsize=12)
    ax.set_ylabel("AUC on held-out test items", fontsize=12)
    ax.set_xticks([0.8, 1, 5, 10, 50, 100, 500])
    ax.set_xticklabels(["0", "1", "5", "10", "50", "100", "500"])
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out_fig = fig_dir / "fig_cold_start_v2.pdf"
    fig.savefig(out_fig, **SAVE_KW)
    plt.close()
    print(f"\nSaved: {out_fig}", flush=True)

    # ── Save ──
    save_data = {
        "dataset": "v2_full", "n_llms": n_llms, "n_test_llms": len(test_llms),
        "n_items": n_items, "n_test_items": len(test_items),
        "K": K, "full_training_auc": full_auc,
        "calibration_sizes": cal_sizes, "n_repeats": n_repeats,
        "summary": summary,
    }
    out_json = Path("cdm_exploration/experiments/v2_cold_start.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label="cold_start_v2")
    log_experiment(
        name="cold_start_v2",
        config={"n_test_llms": len(test_llms), "K": K,
                "cal_sizes": cal_sizes, "n_repeats": n_repeats, "device": device},
        results={"full_auc": full_auc, "summary": summary},
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_train_llms": len(train_llms), "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
