"""Cold-start evaluation on v2 full dataset (3811 LLMs x 9523 items).

Usage:
    python tools/cold_start_v2.py device=cpu
    python tools/cold_start_v2.py device=mps
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


def fit_theta(net, responses, cal_items, q_matrix, text_embs, device, lr=0.01, epochs=100):
    """Fit mastery vector for a new LLM from calibration responses."""
    net.eval()
    net = net.to(device)
    K = net.knowledge_dim

    cal = cal_items.astype(int)
    y = torch.tensor(responses[cal], dtype=torch.float32, device=device)
    te = torch.tensor(text_embs[cal], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[cal], dtype=torch.float32, device=device)

    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        k_diff = torch.sigmoid(net.k_difficulty_proj(te))
        e_diff = torch.sigmoid(net.e_difficulty_proj(te))

    for _ in range(epochs):
        stat = torch.sigmoid(theta_raw).expand(len(cal), -1)
        x = e_diff * (stat - k_diff) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()


def predict_with_theta(net, mastery, items, q_matrix, text_embs, device):
    """Predict P(correct) for a single LLM on specified items."""
    net.eval()
    net = net.to(device)
    idx = items.astype(int)
    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)
    stat = torch.tensor(mastery, dtype=torch.float32, device=device).unsqueeze(0).expand(len(idx), -1)

    with torch.no_grad():
        k_diff = torch.sigmoid(net.k_difficulty_proj(te))
        e_diff = torch.sigmoid(net.e_difficulty_proj(te))
        x = e_diff * (stat - k_diff) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
    return pred.cpu().numpy()


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
    # Item split (Protocol B)
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)

    # LLM split (hold out 20%)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    print(f"  Item split: {len(train_items)} train, {len(test_items)} test", flush=True)
    print(f"  LLM split: {len(train_llms)} train, {len(test_llms)} test ({len(test_llms)} held-out)", flush=True)

    # ── Load model ──
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)

    # ── Full-training baseline AUC ──
    # Evaluate the trained model on train LLMs + test items
    print("\nComputing full-training baseline...", flush=True)
    full_aucs = []
    for llm_idx in train_llms[:50]:  # sample 50 for speed
        responses = R[llm_idx]
        # Use the trained embedding for this LLM
        with torch.no_grad():
            raw = net.student_emb(torch.tensor([llm_idx])).numpy()
            mastery = 1.0 / (1.0 + np.exp(-raw.squeeze()))
        preds = predict_with_theta(net, mastery, test_items, q_matrix, text_embs, device)
        y_true = responses[test_items.astype(int)]
        if len(np.unique(y_true)) >= 2:
            full_aucs.append(roc_auc_score(y_true, preds))
    full_auc = float(np.mean(full_aucs))
    print(f"  Full-training AUC (50 train LLMs): {full_auc:.4f}", flush=True)

    # ── Cold-start evaluation ──
    cal_sizes = [0, 1, 5, 10, 50, 100, 500]
    n_repeats = 5
    rng = np.random.RandomState(42)

    print(f"\nCold-start evaluation: {len(test_llms)} held-out LLMs", flush=True)
    print(f"  Calibration sizes: {cal_sizes}", flush=True)
    print(f"  Repeats: {n_repeats}", flush=True)

    results = {N: {"aucs": [], "accs": []} for N in cal_sizes}

    for li, llm_idx in enumerate(test_llms):
        if (li + 1) % 50 == 0 or li == 0:
            print(f"  LLM {li+1}/{len(test_llms)}", flush=True)
        responses = R[llm_idx]

        for N in cal_sizes:
            reps = 1 if N == 0 else n_repeats
            for rep in range(reps):
                if N == 0:
                    mastery = np.full(K, 0.5)
                else:
                    cal = rng.choice(train_items, size=min(N, len(train_items)), replace=False)
                    mastery = fit_theta(net, responses, cal, q_matrix, text_embs, device)

                preds = predict_with_theta(net, mastery, test_items, q_matrix, text_embs, device)
                y_true = responses[test_items.astype(int)]

                if len(np.unique(y_true)) < 2:
                    continue

                auc = roc_auc_score(y_true, preds)
                acc = accuracy_score(y_true, (preds >= 0.5).astype(int))
                results[N]["aucs"].append(auc)
                results[N]["accs"].append(acc)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"COLD-START RESULTS (v2 dataset, {len(test_llms)} held-out LLMs)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>6} {'AUC mean':>10} {'AUC std':>10} {'Acc mean':>10} {'% of full':>10}", flush=True)
    print("-" * 70, flush=True)

    summary = []
    for N in cal_sizes:
        aucs = results[N]["aucs"]
        accs = results[N]["accs"]
        if len(aucs) == 0:
            continue
        auc_m, auc_s = np.mean(aucs), np.std(aucs)
        acc_m = np.mean(accs)
        pct = auc_m / full_auc * 100
        print(f"{N:>6} {auc_m:>10.4f} {auc_s:>10.4f} {acc_m:>10.4f} {pct:>9.1f}%", flush=True)
        summary.append({"N": N, "auc_mean": auc_m, "auc_std": auc_s,
                         "acc_mean": acc_m, "pct_of_full": pct})

    print(f"\n  Full-training AUC: {full_auc:.4f}", flush=True)

    # Old dataset comparison
    print(f"\n  Old dataset comparison (235 LLMs, K=50):", flush=True)
    print(f"    N=0:   AUC=0.714 (81.8%)", flush=True)
    print(f"    N=50:  AUC=0.740 (84.8%)", flush=True)
    print(f"    N=500: AUC=0.747 (85.6%)", flush=True)

    # ── Figure ──
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ns = [s["N"] for s in summary]
    aucs = [s["auc_mean"] for s in summary]
    stds = [s["auc_std"] for s in summary]

    ax.fill_between(ns, [a - s for a, s in zip(aucs, stds)],
                    [a + s for a, s in zip(aucs, stds)],
                    alpha=0.2, color="#4C72B0")
    ax.plot(ns, aucs, "o-", color="#4C72B0", lw=2, markersize=7,
            label="Cold-start AUC")
    ax.axhline(full_auc, ls="--", color="#55A868", lw=1.5,
               label=f"Full training ({full_auc:.3f})")
    ax.axhline(0.5, ls=":", color="#999999", lw=1, label="Random guess")

    # 90% threshold
    threshold_90 = full_auc * 0.9
    ax.axhline(threshold_90, ls="--", color="#DD8452", lw=1, alpha=0.7,
               label=f"90% of full ({threshold_90:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Number of calibration items", fontsize=12)
    ax.set_ylabel("AUC on held-out test items", fontsize=12)
    ax.set_xticks([0.8, 1, 5, 10, 50, 100, 500])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_fig = fig_dir / "fig_cold_start_v2.pdf"
    fig.savefig(out_fig, **SAVE_KW)
    plt.close()
    print(f"\nSaved: {out_fig}", flush=True)

    # ── Save ──
    save_data = {
        "dataset": "v2_full",
        "n_llms": n_llms, "n_test_llms": len(test_llms),
        "n_items": n_items, "n_test_items": len(test_items),
        "K": K, "full_training_auc": full_auc,
        "calibration_sizes": cal_sizes,
        "n_repeats": n_repeats,
        "summary": summary,
    }
    out_json = Path("cdm_exploration/experiments/v2_cold_start.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved: {out_json}", flush=True)

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label="cold_start_v2")
    log_experiment(
        name="cold_start_v2",
        config={"n_llms": n_llms, "n_test_llms": len(test_llms), "K": K,
                "cal_sizes": cal_sizes, "n_repeats": n_repeats, "device": device},
        results={"full_auc": full_auc, "summary": summary},
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_train_llms": len(train_llms), "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
