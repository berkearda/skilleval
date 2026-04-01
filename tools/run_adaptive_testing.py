"""CD-CAT: Cognitive Diagnostic Computerized Adaptive Testing.

Compares random vs adaptive item selection for profiling new LLMs.
Adaptive strategy uses Fisher Information to select maximally informative items.

Usage:
    python tools/run_adaptive_testing.py device=cuda
    python tools/run_adaptive_testing.py device=cpu model.epochs=1
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def predict_with_theta(net, mastery, items, q_matrix, text_embs, device):
    """Predict P(correct) for one LLM on specified items."""
    net.eval()
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


def compute_fisher_information(net, theta_raw, item_indices, q_matrix, text_embs, device):
    """Compute Fisher information for each candidate item given current theta.

    Fisher info for binary response: I(θ) = p(1-p) * (∂logit/∂θ)²
    We approximate by: items where predicted P is closest to 0.5 are most informative.
    This is the standard CAT heuristic (maximum information criterion).
    """
    K = theta_raw.shape[1]
    idx = item_indices.astype(int)
    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)
    stat = torch.sigmoid(theta_raw).expand(len(idx), -1)

    with torch.no_grad():
        k_diff = torch.sigmoid(net.k_difficulty_proj(te))
        e_diff = torch.sigmoid(net.e_difficulty_proj(te))
        x = e_diff * (stat - k_diff) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)

    p = pred.cpu().numpy()
    # Fisher info ∝ p(1-p) — maximized at p=0.5
    info = p * (1 - p)
    # Weight by number of active skills (items testing more skills give more info)
    n_skills = q_matrix[idx].sum(axis=1)
    info = info * np.maximum(n_skills, 1)
    return info


def fit_theta_single_step(net, theta_raw, optimizer, item_idx, response, q_matrix, text_embs, device):
    """One gradient step to update theta given a single observed response."""
    te = torch.tensor(text_embs[int(item_idx)], dtype=torch.float32, device=device).unsqueeze(0)
    qr = torch.tensor(q_matrix[int(item_idx)], dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor([float(response)], dtype=torch.float32, device=device)

    stat = torch.sigmoid(theta_raw)

    k_diff = torch.sigmoid(net.k_difficulty_proj(te))
    e_diff = torch.sigmoid(net.e_difficulty_proj(te))
    x = e_diff * (stat - k_diff) * qr
    h1 = torch.sigmoid(net.prednet_full1(x))
    h2 = torch.sigmoid(net.prednet_full2(h1))
    pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)

    loss = nn.BCELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def fit_theta_batch(net, responses, cal_items, q_matrix, text_embs, device, K, lr=0.01, steps=50):
    """Fit theta from a batch of calibration items (for random strategy)."""
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

    for _ in range(steps):
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    from cdmeval.validation import validate_data

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
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)

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

    # Freeze all parameters
    for p in net.parameters():
        p.requires_grad = False

    # ── Experiment ──
    cal_sizes = [5, 10, 20, 50, 100, 200, 500]
    n_repeats = 3
    # Use a subset of held-out LLMs for speed
    n_eval_llms = min(100, len(test_llms))
    eval_llms = test_llms[:n_eval_llms]

    print(f"\nCD-CAT experiment: {n_eval_llms} LLMs, sizes={cal_sizes}, repeats={n_repeats}", flush=True)

    results_random = {N: [] for N in cal_sizes}
    results_adaptive = {N: [] for N in cal_sizes}

    for li, llm_idx in enumerate(eval_llms):
        if (li + 1) % 20 == 0 or li == 0:
            print(f"  LLM {li+1}/{n_eval_llms}", flush=True)
        responses = R[llm_idx]
        y_test = responses[test_items.astype(int)]
        if len(np.unique(y_test)) < 2:
            continue

        for N in cal_sizes:
            for rep in range(n_repeats):
                rng = np.random.RandomState(42 + rep)

                # ── Strategy A: Random ──
                cal_random = rng.choice(train_items, size=min(N, len(train_items)), replace=False)
                mastery_random = fit_theta_batch(net, responses, cal_random, q_matrix, text_embs, device, K)
                preds_random = predict_with_theta(net, mastery_random, test_items, q_matrix, text_embs, device)
                auc_random = roc_auc_score(y_test, preds_random)
                results_random[N].append(auc_random)

                # ── Strategy B: Adaptive (Fisher Information) ──
                theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
                optimizer = torch.optim.Adam([theta_raw], lr=0.05)
                remaining = set(train_items.tolist())
                remaining_arr = train_items.copy()

                for step in range(min(N, len(train_items))):
                    # Compute Fisher info for remaining items
                    info = compute_fisher_information(
                        net, theta_raw, remaining_arr, q_matrix, text_embs, device
                    )
                    # Pick highest info item
                    best_local = np.argmax(info)
                    best_item = remaining_arr[best_local]

                    # Observe response and update theta
                    fit_theta_single_step(
                        net, theta_raw, optimizer,
                        best_item, responses[int(best_item)],
                        q_matrix, text_embs, device,
                    )

                    # Remove from pool
                    remaining.discard(int(best_item))
                    remaining_arr = np.array(sorted(remaining))

                mastery_adaptive = torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()
                preds_adaptive = predict_with_theta(net, mastery_adaptive, test_items, q_matrix, text_embs, device)
                auc_adaptive = roc_auc_score(y_test, preds_adaptive)
                results_adaptive[N].append(auc_adaptive)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"CD-CAT RESULTS ({n_eval_llms} held-out LLMs)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>6} {'Random AUC':>12} {'Adaptive AUC':>14} {'Gain':>8}", flush=True)
    print("-" * 70, flush=True)

    summary = []
    for N in cal_sizes:
        r_mean = np.mean(results_random[N])
        r_std = np.std(results_random[N])
        a_mean = np.mean(results_adaptive[N])
        a_std = np.std(results_adaptive[N])
        gain = a_mean - r_mean
        print(f"{N:>6} {r_mean:>8.4f}±{r_std:.4f} {a_mean:>10.4f}±{a_std:.4f} {gain:>+8.4f}", flush=True)
        summary.append({
            "N": N,
            "random_mean": float(r_mean), "random_std": float(r_std),
            "adaptive_mean": float(a_mean), "adaptive_std": float(a_std),
            "gain": float(gain),
        })

    # ── Figure ──
    print("\nGenerating figure...", flush=True)
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ns = [s["N"] for s in summary]
    r_aucs = [s["random_mean"] for s in summary]
    r_stds = [s["random_std"] for s in summary]
    a_aucs = [s["adaptive_mean"] for s in summary]
    a_stds = [s["adaptive_std"] for s in summary]

    ax.fill_between(ns, [a - s for a, s in zip(a_aucs, a_stds)],
                    [a + s for a, s in zip(a_aucs, a_stds)], alpha=0.15, color="#4C72B0")
    ax.fill_between(ns, [a - s for a, s in zip(r_aucs, r_stds)],
                    [a + s for a, s in zip(r_aucs, r_stds)], alpha=0.15, color="#DD8452")
    ax.plot(ns, a_aucs, "o-", color="#4C72B0", lw=2, markersize=7, label="Adaptive (Fisher info)")
    ax.plot(ns, r_aucs, "s--", color="#DD8452", lw=1.5, markersize=6, label="Random selection")

    ax.set_xscale("log")
    ax.set_xlabel("Number of calibration items", fontsize=13)
    ax.set_ylabel("AUC on held-out test items", fontsize=13)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_fig = fig_dir / "fig_adaptive_testing.pdf"
    fig.savefig(out_fig, **SAVE_KW)
    plt.close()
    print(f"Saved: {out_fig}", flush=True)

    # ── Save ──
    save_data = {
        "experiment": "adaptive_testing",
        "n_eval_llms": n_eval_llms,
        "n_repeats": n_repeats,
        "K": K, "n_items": n_items,
        "summary": summary,
    }
    out_json = Path("cdm_exploration/experiments/v2_adaptive_testing.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    log_experiment(
        name="adaptive_testing",
        config={"n_eval_llms": n_eval_llms, "K": K, "n_repeats": n_repeats,
                "cal_sizes": cal_sizes, "device": device},
        results={"summary": summary},
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_train_llms": len(train_llms), "n_test_llms": len(test_llms)},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
