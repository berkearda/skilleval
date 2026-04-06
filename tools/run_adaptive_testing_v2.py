"""CD-CAT v2: Theoretically grounded adaptive testing with three item selection criteria.

Criteria:
  a) Heuristic: p(1-p) * sum(q_j)
  b) Trace: p(1-p) * ||grad_theta z_j||^2
  c) D-optimal: c_j * g_j^T @ I_cum_inv @ g_j with Sherman-Morrison updates

Theta update: MAP on full history (15 Adam steps on cumulative loss + L2 prior).

Usage:
    python tools/run_adaptive_testing_v2.py device=cuda
    python tools/run_adaptive_testing_v2.py device=cpu
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
    """Predict P(correct) for one LLM on items."""
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


def map_update(net, items_seen, responses_seen, q_matrix, text_embs, device, K,
               lr=0.01, steps=15, lam=0.01):
    """MAP estimate of theta from full observation history."""
    if len(items_seen) == 0:
        return np.full(K, 0.5)

    idx = np.array(items_seen, dtype=int)
    y = torch.tensor(np.array(responses_seen, dtype=np.float32), device=device)
    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)

    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))
        e_d = torch.sigmoid(net.e_difficulty_proj(te))

    for _ in range(steps):
        stat = torch.sigmoid(theta_raw).expand(len(idx), -1)
        x = e_d * (stat - k_d) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
        loss = loss_fn(pred, y) + lam * (theta_raw ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()


def compute_gradients_and_preds(net, theta_raw, candidate_items, q_matrix, text_embs, device):
    """Compute g_j = grad(z_j, theta_raw) and p_j for each candidate item.

    Returns: grads (n_candidates, K), preds (n_candidates,)
    """
    idx = candidate_items.astype(int)
    n = len(idx)
    K = theta_raw.shape[1]

    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))
        e_d = torch.sigmoid(net.e_difficulty_proj(te))

    grads = torch.zeros(n, K, device=device)
    preds = torch.zeros(n, device=device)

    for i in range(n):
        theta_raw.requires_grad_(True)
        if theta_raw.grad is not None:
            theta_raw.grad.zero_()

        stat = torch.sigmoid(theta_raw)
        x = e_d[i:i+1] * (stat - k_d[i:i+1]) * qr[i:i+1]
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        z = net.prednet_full3(h2).squeeze()  # pre-sigmoid logit

        g = torch.autograd.grad(z, theta_raw, retain_graph=False)[0].squeeze()
        grads[i] = g.detach()
        preds[i] = torch.sigmoid(z).detach()

        theta_raw.requires_grad_(False)

    return grads.cpu().numpy(), preds.cpu().numpy()


def select_heuristic(net, theta_raw, candidates, q_matrix, text_embs, device):
    """Heuristic: p(1-p) * sum(q_j)."""
    idx = candidates.astype(int)
    mastery = torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()
    p = predict_with_theta(net, mastery, candidates, q_matrix, text_embs, device)
    info = p * (1 - p)
    n_skills = q_matrix[idx].sum(axis=1)
    scores = info * np.maximum(n_skills, 1)
    return int(candidates[np.argmax(scores)])


def select_trace(net, theta_raw, candidates, q_matrix, text_embs, device):
    """Trace criterion: p(1-p) * ||g_j||^2."""
    grads, preds = compute_gradients_and_preds(net, theta_raw, candidates, q_matrix, text_embs, device)
    c = preds * (1 - preds)
    grad_norm_sq = (grads ** 2).sum(axis=1)
    scores = c * grad_norm_sq
    return int(candidates[np.argmax(scores)])


def select_doptimal(net, theta_raw, candidates, q_matrix, text_embs, device, I_cum_inv):
    """D-optimal: c_j * g_j^T @ I_cum_inv @ g_j. Returns (selected_item, updated_I_cum_inv)."""
    grads, preds = compute_gradients_and_preds(net, theta_raw, candidates, q_matrix, text_embs, device)
    c = preds * (1 - preds)

    # Score each candidate
    scores = np.zeros(len(candidates))
    for i in range(len(candidates)):
        g = grads[i]
        scores[i] = c[i] * (g @ I_cum_inv @ g)

    best = np.argmax(scores)
    best_item = int(candidates[best])

    # Sherman-Morrison update
    g_best = grads[best]
    c_best = c[best]
    Ig = I_cum_inv @ g_best
    denom = 1.0 + c_best * (g_best @ Ig)
    I_cum_inv = I_cum_inv - (c_best * np.outer(Ig, Ig)) / denom

    return best_item, I_cum_inv


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

    # ── Load ──
    print("\nLoading...", flush=True)
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

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    print(f"  {n_llms} LLMs, K={K}, {len(test_items)} test items", flush=True)

    # ── Load model ──
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device); net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # ── Experiment ──
    cal_sizes = [10, 50, 100, 200, 500]
    n_repeats = 3
    n_eval = min(100, len(test_llms))
    eval_llms = test_llms[:n_eval]
    criteria = ["random", "heuristic", "trace", "doptimal"]

    print(f"\nCD-CAT v2: {n_eval} LLMs, N={cal_sizes}, repeats={n_repeats}", flush=True)
    print(f"  Criteria: {criteria}", flush=True)

    results = {crit: {N: [] for N in cal_sizes} for crit in criteria}

    for li, llm_idx in enumerate(eval_llms):
        if (li + 1) % 10 == 0 or li == 0:
            print(f"  LLM {li+1}/{n_eval}", flush=True)
        responses = R[llm_idx]
        y_test = responses[test_items.astype(int)]
        if len(np.unique(y_test)) < 2:
            continue

        for N in cal_sizes:
            for rep in range(n_repeats):
                rng = np.random.RandomState(42 + rep)

                # ── Random ──
                cal = rng.choice(train_items, size=min(N, len(train_items)), replace=False)
                mastery = map_update(net, cal.tolist(), responses[cal].tolist(),
                                     q_matrix, text_embs, device, K)
                preds = predict_with_theta(net, mastery, test_items, q_matrix, text_embs, device)
                results["random"][N].append(roc_auc_score(y_test, preds))

                # ── Adaptive criteria ──
                for crit in ["heuristic", "trace", "doptimal"]:
                    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
                    remaining = set(train_items.tolist())
                    remaining_arr = train_items.copy()
                    items_seen, responses_seen = [], []

                    if crit == "doptimal":
                        I_cum_inv = (1.0 / 0.01) * np.eye(K)

                    for step in range(min(N, len(train_items))):
                        # Select item
                        if crit == "heuristic":
                            chosen = select_heuristic(net, theta_raw, remaining_arr,
                                                       q_matrix, text_embs, device)
                        elif crit == "trace":
                            chosen = select_trace(net, theta_raw, remaining_arr,
                                                   q_matrix, text_embs, device)
                        elif crit == "doptimal":
                            chosen, I_cum_inv = select_doptimal(
                                net, theta_raw, remaining_arr,
                                q_matrix, text_embs, device, I_cum_inv)

                        # Observe and accumulate
                        items_seen.append(chosen)
                        responses_seen.append(float(responses[chosen]))
                        remaining.discard(chosen)
                        remaining_arr = np.array(sorted(remaining))

                        # MAP update on full history
                        mastery = map_update(net, items_seen, responses_seen,
                                             q_matrix, text_embs, device, K)
                        theta_raw = nn.Parameter(
                            torch.log(torch.tensor(mastery / (1 - np.clip(mastery, 1e-6, 1-1e-6)),
                                                    dtype=torch.float32, device=device)).unsqueeze(0))

                    preds = predict_with_theta(net, mastery, test_items, q_matrix, text_embs, device)
                    results[crit][N].append(roc_auc_score(y_test, preds))

    # ── Summary ──
    print(f"\n{'='*80}", flush=True)
    print(f"CD-CAT v2 RESULTS ({n_eval} LLMs)", flush=True)
    print(f"{'='*80}", flush=True)
    hdr = f"{'N':>6}"
    for crit in criteria:
        hdr += f"  {crit:>14}"
    print(hdr, flush=True)
    print("-" * 80, flush=True)

    summary = []
    for N in cal_sizes:
        row = {"N": N}
        line = f"{N:>6}"
        for crit in criteria:
            vals = results[crit][N]
            m, s = np.mean(vals), np.std(vals)
            line += f"  {m:>7.4f}+/-{s:.4f}"
            row[f"{crit}_mean"] = float(m)
            row[f"{crit}_std"] = float(s)
        print(line, flush=True)
        summary.append(row)

    # ── Figure ──
    print("\nGenerating figure...", flush=True)
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"random": "#888888", "heuristic": "#DD8452", "trace": "#4C72B0", "doptimal": "#22C55E"}
    styles = {"random": ("s", "--"), "heuristic": ("^", "-."), "trace": ("o", "-"), "doptimal": ("D", "-")}
    labels = {"random": "Random", "heuristic": "Heuristic", "trace": "Trace", "doptimal": "D-optimal"}

    for crit in criteria:
        means = [np.mean(results[crit][N]) for N in cal_sizes]
        stds = [np.std(results[crit][N]) for N in cal_sizes]
        marker, ls = styles[crit]
        ax.fill_between(cal_sizes, [m-s for m, s in zip(means, stds)],
                        [m+s for m, s in zip(means, stds)],
                        alpha=0.08 if crit == "random" else 0.12, color=colors[crit])
        ax.plot(cal_sizes, means, f"{marker}{ls}", color=colors[crit], lw=2,
                markersize=7, label=labels[crit], alpha=0.85 if crit != "random" else 0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of calibration items", fontsize=13)
    ax.set_ylabel("AUC on held-out test items", fontsize=13)
    ax.set_xticks(cal_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.1, linewidth=0.5)
    ax.grid(False, axis="y")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_fig = fig_dir / "fig_adaptive_testing_v2.pdf"
    fig.savefig(out_fig, **SAVE_KW)
    plt.close()
    print(f"Saved: {out_fig}", flush=True)

    # ── Save ──
    save_data = {
        "experiment": "adaptive_testing_v2",
        "n_eval_llms": n_eval, "n_repeats": n_repeats, "K": K,
        "cal_sizes": cal_sizes, "criteria": criteria,
        "summary": summary,
    }
    out_json = Path("cdm_exploration/experiments/v2_adaptive_testing_v2.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    log_experiment(
        name="adaptive_testing_v2",
        config={"n_eval": n_eval, "K": K, "n_repeats": n_repeats,
                "cal_sizes": cal_sizes, "criteria": criteria, "device": device},
        results={"summary": summary},
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_train_llms": len(train_llms), "n_test_llms": len(test_llms)},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
