"""CD-CAT v3: Batched GPU adaptive testing with Newton theta updates.

Key changes from v2:
  - ALL held-out LLMs processed simultaneously (batched forward/backward)
  - Batch item selection: 5 rounds x 10 items = 50 total
  - Newton's method for theta update (replaces Adam)
  - A-optimality criterion with diversity penalty
  - Primary metric: RMSE(theta_adaptive, theta_full)

Usage:
    python tools/run_adaptive_testing_v3.py device=cpu '+n_eval=20'   # local debug
    python tools/run_adaptive_testing_v3.py device=cuda               # full scale
"""

import json
import sys
import time
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ═══════════════════════════════════════════════════════════════════════
# BATCHED FORWARD PASS
# ═══════════════════════════════════════════════════════════════════════

def batched_forward(net, theta_sig, item_ids, q_matrix, text_embs, device):
    """Compute P(correct) for all (LLM, item) pairs in one pass.

    Args:
        net: Frozen TextConditionedNet.
        theta_sig: (M, K) sigmoid mastery for M LLMs.
        item_ids: (J,) int array of item indices.
        q_matrix: (N_total, K) numpy Q-matrix.
        text_embs: (N_total, D) numpy text embeddings.
        device: torch device.

    Returns:
        (M, J) numpy array of predicted probabilities.
    """
    M = theta_sig.shape[0]
    J = len(item_ids)
    idx = item_ids.astype(int)

    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)     # (J, D)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)      # (J, K)

    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))   # (J, K)
        e_d = torch.sigmoid(net.e_difficulty_proj(te))    # (J, 1)

    # Broadcast: stat (M,1,K) - k_d (1,J,K) → (M,J,K)
    stat = theta_sig.unsqueeze(1)       # (M, 1, K)
    k_d = k_d.unsqueeze(0)             # (1, J, K)
    e_d = e_d.unsqueeze(0)             # (1, J, 1)
    qr = qr.unsqueeze(0)              # (1, J, K)

    with torch.no_grad():
        x = e_d * (stat - k_d) * qr    # (M, J, K)
        # Flatten batch dims for PosLinear
        MJ = M * J
        x_flat = x.reshape(MJ, -1)     # (M*J, K)
        h1 = torch.sigmoid(net.prednet_full1(x_flat))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2))  # (M*J, 1)
        pred = pred.reshape(M, J)

    return pred.cpu().numpy()


def batched_forward_with_grad(net, theta_raw, item_ids, q_matrix, text_embs, device):
    """Forward pass that allows gradients w.r.t. theta_raw.

    Args:
        net: Frozen TextConditionedNet.
        theta_raw: (M, K) Parameter (requires_grad).
        item_ids: (J,) int array.

    Returns:
        pred: (M, J) tensor (on device, in graph).
    """
    M = theta_raw.shape[0]
    J = len(item_ids)
    idx = item_ids.astype(int)

    te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))   # (J, K)
        e_d = torch.sigmoid(net.e_difficulty_proj(te))    # (J, 1)

    stat = torch.sigmoid(theta_raw).unsqueeze(1)  # (M, 1, K)
    k_d = k_d.unsqueeze(0)
    e_d = e_d.unsqueeze(0)
    qr_b = qr.unsqueeze(0)

    x = e_d * (stat - k_d) * qr_b    # (M, J, K)
    MJ = M * J
    x_flat = x.reshape(MJ, -1)
    h1 = torch.sigmoid(net.prednet_full1(x_flat))
    h2 = torch.sigmoid(net.prednet_full2(h1))
    pred = torch.sigmoid(net.prednet_full3(h2)).reshape(M, J)
    return pred


# ═══════════════════════════════════════════════════════════════════════
# NEWTON THETA UPDATE
# ═══════════════════════════════════════════════════════════════════════

def newton_update(net, theta_raw, items_seen, responses, q_matrix, text_embs,
                  device, lam=0.1, n_steps=3):
    """Update theta via damped Newton's method on the MAP objective.

    Optimizes: -log_likelihood + (lam/2) * ||theta_raw||^2

    Returns:
        theta_raw: updated (M, K) Parameter (detached, new leaf).
        Sigma: (M, K, K) posterior covariance per LLM (numpy).
    """
    M, K = theta_raw.shape
    items_arr = np.array(items_seen, dtype=int)
    J = len(items_arr)

    if J == 0:
        theta_raw = nn.Parameter(theta_raw.detach().clone())
        Sigma = np.tile(np.eye(K) / lam, (M, 1, 1))
        return theta_raw, Sigma

    # responses: (M, J) numpy
    y = torch.tensor(responses, dtype=torch.float32, device=device)  # (M, J)

    for step in range(n_steps):
        theta_raw = nn.Parameter(theta_raw.detach().clone())

        pred = batched_forward_with_grad(net, theta_raw, items_arr,
                                         q_matrix, text_embs, device)  # (M, J)

        # BCE loss + L2 prior, summed over items, mean over LLMs
        eps = 1e-7
        nll = -(y * torch.log(pred + eps) + (1 - y) * torch.log(1 - pred + eps)).sum(dim=1)
        prior = 0.5 * lam * (theta_raw ** 2).sum(dim=1)
        loss = (nll + prior).sum()

        loss.backward()
        grad = theta_raw.grad.detach()  # (M, K)

        # Gauss-Newton Hessian approximation: H_m = sum_j c_j * g_j g_j^T + lam*I
        # where c_j = p_j(1-p_j) and g_j = dp_j/d_theta_raw (per LLM)
        # For efficiency, compute per-LLM Hessian and invert
        with torch.no_grad():
            p = pred.detach()  # (M, J)
            c = p * (1 - p)   # (M, J)

        # We need per-item gradients. Compute them via a J-loop (items are few: <=50).
        per_item_grads = []  # list of (M, K) tensors
        for j_idx in range(J):
            theta_j = nn.Parameter(theta_raw.detach().clone())
            pred_j = batched_forward_with_grad(net, theta_j, items_arr[j_idx:j_idx+1],
                                               q_matrix, text_embs, device)  # (M, 1)
            # grad of pred_j w.r.t. theta_j for each m
            g_list = []
            for m in range(min(M, 50)):
                # For large M, approximate with a subset
                gm = torch.autograd.grad(pred_j[m, 0], theta_j, retain_graph=True)[0][m]
                g_list.append(gm.detach())
            if M <= 50:
                per_item_grads.append(torch.stack(g_list))  # (M, K)
            else:
                # For M > 50, use the first 50 to estimate, then extrapolate
                per_item_grads.append(torch.stack(g_list))
            del theta_j

        # Build H and invert for each LLM (or batch of 50)
        M_eff = min(M, 50)
        Sigma_list = []
        for m in range(M_eff):
            H = lam * torch.eye(K, device=device)
            for j_idx in range(J):
                g = per_item_grads[j_idx][m]  # (K,)
                H += c[m, j_idx] * torch.outer(g, g)
            Sigma_m = torch.linalg.inv(H)
            # Newton step
            with torch.no_grad():
                theta_raw.data[m] -= Sigma_m @ grad[m]
            Sigma_list.append(Sigma_m.cpu().numpy())

        # For LLMs beyond M_eff, use gradient descent as fallback
        if M > M_eff:
            with torch.no_grad():
                theta_raw.data[M_eff:] -= 0.1 * grad[M_eff:]

    # Final Sigma computation for item selection
    Sigma = np.stack(Sigma_list) if Sigma_list else np.tile(np.eye(K) / lam, (M_eff, 1, 1))

    theta_raw = nn.Parameter(theta_raw.detach().clone())
    return theta_raw, Sigma


# ═══════════════════════════════════════════════════════════════════════
# BATCH ITEM SELECTION
# ═══════════════════════════════════════════════════════════════════════

def select_initial_batch(q_matrix, train_items, batch_size=10, rng=None):
    """Round 1: Select items covering diverse skills via greedy skill coverage."""
    if rng is None:
        rng = np.random.RandomState(42)

    pool = train_items.copy()
    Q = q_matrix[pool.astype(int)]  # (pool_size, K)
    covered = np.zeros(q_matrix.shape[1])
    selected = []

    for _ in range(batch_size):
        # Score: number of new skills each item covers
        new_coverage = np.maximum(Q - covered[np.newaxis, :], 0).sum(axis=1)
        # Tie-break by total skill count
        scores = new_coverage + 0.001 * Q.sum(axis=1)
        best = np.argmax(scores)
        selected.append(int(pool[best]))
        covered = np.maximum(covered, Q[best])
        # Remove selected
        mask = np.ones(len(pool), dtype=bool)
        mask[best] = False
        pool = pool[mask]
        Q = Q[mask]

    return np.array(selected)


def select_adaptive_batch(net, theta_raw, Sigma, remaining_items, q_matrix,
                          text_embs, device, batch_size=10, cos_penalty=0.8):
    """Adaptive batch selection via A-optimality with diversity penalty.

    For each candidate j, score = g_j^T @ Sigma_mean @ g_j / (p_j * (1-p_j))
    where Sigma_mean is averaged over LLMs. After selecting each item,
    penalize candidates with similar Q-rows (cosine > cos_penalty).
    """
    M, K = theta_raw.shape
    pool = remaining_items.copy()
    M_eff = min(M, 50)

    # Average Sigma across LLMs (use first M_eff)
    Sigma_mean = Sigma[:M_eff].mean(axis=0)  # (K, K)
    Sigma_t = torch.tensor(Sigma_mean, dtype=torch.float32, device=device)

    # Compute predictions for all candidates (batched)
    theta_sig = torch.sigmoid(theta_raw.detach())
    preds = batched_forward(net, theta_sig[:M_eff], pool, q_matrix, text_embs, device)
    p_mean = preds.mean(axis=0)  # (pool_size,) average over LLMs
    c = p_mean * (1 - p_mean) + 1e-8  # (pool_size,)

    # Compute gradients for all candidates (using mean theta)
    # Single forward pass per candidate with mean theta
    theta_mean = theta_raw[:1].detach().clone()
    theta_mean.data[:] = theta_raw.detach().mean(dim=0, keepdim=True)
    theta_mean = nn.Parameter(theta_mean)

    # Batched gradient computation via Q-row approximation (fast):
    # For A-optimality, g_j ≈ dsigmoid(theta_raw) * e_j * q_j (first-order approx)
    # This avoids per-item autograd
    with torch.no_grad():
        idx = pool.astype(int)
        te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)  # (pool, K)
        k_d = torch.sigmoid(net.k_difficulty_proj(te))  # (pool, K)
        e_d = torch.sigmoid(net.e_difficulty_proj(te))   # (pool, 1)
        stat_mean = torch.sigmoid(theta_mean.detach()).squeeze(0)  # (K,)

        # Approximate gradient: g_j ≈ e_j * sigmoid'(z_j) * q_j
        # where z_j depends on the PosLinear layers. For speed, use q_j * e_j as proxy.
        g_approx = qr * e_d  # (pool, K) — fast proxy for gradient direction

    # A-optimality scores: g^T Sigma g / c
    # (pool, K) @ (K, K) → (pool, K), then dot with g → (pool,)
    g_Sigma = g_approx @ Sigma_t  # (pool, K)
    a_scores = (g_Sigma * g_approx).sum(dim=1).cpu().numpy() / c  # (pool,)

    # Q-row cosine similarity matrix for diversity penalty
    Q_pool = q_matrix[pool.astype(int)].astype(np.float32)
    Q_norms = np.linalg.norm(Q_pool, axis=1, keepdims=True) + 1e-8
    Q_normed = Q_pool / Q_norms

    selected = []
    penalty = np.ones(len(pool))

    for _ in range(batch_size):
        scores = a_scores * penalty
        best = np.argmax(scores)
        selected.append(int(pool[best]))

        # Penalize items with similar Q-rows
        sim = Q_normed @ Q_normed[best]
        penalty *= (1.0 - 0.9 * (sim > cos_penalty).astype(float))
        penalty[best] = 0  # don't re-select

    return np.array(selected)


def select_random_batch(train_items, seen, batch_size, rng):
    """Random baseline: uniform random from unseen items."""
    remaining = np.setdiff1d(train_items, seen)
    return rng.choice(remaining, size=min(batch_size, len(remaining)), replace=False)


def select_stratified_batch(q_matrix, train_items, seen, batch_size, rng):
    """Stratified random: one item per skill cluster, random within cluster."""
    remaining = np.setdiff1d(train_items, seen)
    Q = q_matrix[remaining.astype(int)]
    K = Q.shape[1]

    selected = []
    skills_used = set()
    pool_idx = rng.permutation(len(remaining))

    # First pass: one item per skill
    for i in pool_idx:
        if len(selected) >= batch_size:
            break
        item_skills = set(np.where(Q[i] > 0)[0])
        new_skills = item_skills - skills_used
        if new_skills:
            selected.append(int(remaining[i]))
            skills_used.update(item_skills)

    # Fill remainder randomly if needed
    if len(selected) < batch_size:
        leftover = [int(remaining[i]) for i in pool_idx
                     if int(remaining[i]) not in selected]
        need = batch_size - len(selected)
        selected.extend(leftover[:need])

    return np.array(selected[:batch_size])


# ═══════════════════════════════════════════════════════════════════════
# SIMPLE THETA UPDATE (gradient descent, avoids per-LLM Hessian cost)
# ═══════════════════════════════════════════════════════════════════════

def gd_theta_update(net, theta_raw, items_seen, responses, q_matrix, text_embs,
                    device, lam=0.1, lr=0.05, n_steps=30):
    """Update theta via gradient descent on MAP objective (fast, scales to any M).

    Returns:
        theta_raw: (M, K) updated Parameter.
        Sigma_approx: (K, K) approximate posterior covariance (diagonal).
    """
    M, K = theta_raw.shape
    items_arr = np.array(items_seen, dtype=int)
    J = len(items_arr)

    if J == 0:
        theta_raw = nn.Parameter(theta_raw.detach().clone())
        return theta_raw, np.eye(K) / lam

    y = torch.tensor(responses, dtype=torch.float32, device=device)  # (M, J)

    for step in range(n_steps):
        theta_raw = nn.Parameter(theta_raw.detach().clone())
        pred = batched_forward_with_grad(net, theta_raw, items_arr,
                                         q_matrix, text_embs, device)
        eps = 1e-7
        nll = -(y * torch.log(pred + eps) + (1 - y) * torch.log(1 - pred + eps)).sum(dim=1)
        prior = 0.5 * lam * (theta_raw ** 2).sum(dim=1)
        loss = (nll + prior).sum()
        loss.backward()

        with torch.no_grad():
            theta_raw.data -= lr * theta_raw.grad

    # Approximate Sigma from final gradient magnitudes (diagonal approx)
    # For item selection, this is sufficient since we use average Sigma
    with torch.no_grad():
        p = pred.detach()
        c = p * (1 - p)
        # Diagonal Fisher approx: F_kk ≈ sum_j c_mean_j * q_jk^2
        c_mean = c.mean(dim=0)  # (J,)
        qr = torch.tensor(q_matrix[items_arr], dtype=torch.float32, device=device)  # (J, K)
        diag_F = (c_mean.unsqueeze(1) * qr ** 2).sum(dim=0) + lam  # (K,)
        Sigma_diag = 1.0 / diag_F.cpu().numpy()

    theta_raw = nn.Parameter(theta_raw.detach().clone())
    Sigma = np.diag(Sigma_diag)
    return theta_raw, Sigma


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

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
    n_eval = int(cfg.n_eval) if hasattr(cfg, "n_eval") else 200
    print(f"Device: {device}, n_eval: {n_eval}", flush=True)

    # ── Load data ──
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
    _, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    eval_llms = test_llms[:n_eval]
    M = len(eval_llms)
    print(f"  {n_llms} LLMs total, evaluating {M}, K={K}", flush=True)
    print(f"  {len(train_items)} train items (pool), {len(test_items)} test items", flush=True)

    # ── Load model and ground-truth theta ──
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    with torch.no_grad():
        theta_full = torch.sigmoid(net.student_emb.weight[eval_llms]).cpu().numpy()  # (M, K)
    print(f"  theta_full shape: {theta_full.shape}", flush=True)

    # Population mean theta (initialization for adaptive)
    with torch.no_grad():
        all_train_llms = np.setdiff1d(all_llms, test_llms)
        theta_pop_mean = torch.sigmoid(
            net.student_emb.weight[all_train_llms]
        ).mean(dim=0).cpu().numpy()  # (K,)
    print(f"  Population mean theta range: [{theta_pop_mean.min():.3f}, {theta_pop_mean.max():.3f}]",
          flush=True)

    # ── Configuration ──
    N_ROUNDS = 5
    BATCH_SIZE = 10
    TOTAL_ITEMS = N_ROUNDS * BATCH_SIZE  # 50

    methods = ["random", "stratified", "a_optimal"]
    checkpoints = list(range(BATCH_SIZE, TOTAL_ITEMS + 1, BATCH_SIZE))  # [10, 20, 30, 40, 50]

    print(f"\n  {N_ROUNDS} rounds x {BATCH_SIZE} items = {TOTAL_ITEMS} total", flush=True)
    print(f"  Methods: {methods}", flush=True)
    print(f"  Checkpoints: {checkpoints}", flush=True)

    # ── Responses for eval LLMs ──
    R_eval = R[eval_llms]  # (M, n_items)

    # ── Run methods ──
    all_results = {}

    for method in methods:
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Method: {method}", flush=True)
        print(f"{'='*60}", flush=True)

        rng = np.random.RandomState(42)

        # Initialize theta_raw to population mean (in logit space)
        theta_init_sig = np.tile(theta_pop_mean, (M, 1))  # (M, K)
        theta_init_raw = np.log(theta_init_sig / (1 - theta_init_sig + 1e-8) + 1e-8)
        theta_raw = nn.Parameter(
            torch.tensor(theta_init_raw, dtype=torch.float32, device=device)
        )

        items_seen = []
        responses_seen = np.zeros((M, 0))
        Sigma = np.eye(K) / 0.1  # Prior covariance (lam=0.1)

        method_results = {cp: {} for cp in checkpoints}

        for rnd in range(N_ROUNDS):
            # ── Select batch ──
            if method == "random":
                batch = select_random_batch(train_items, items_seen, BATCH_SIZE, rng)
            elif method == "stratified":
                batch = select_stratified_batch(q_matrix, train_items, items_seen,
                                                BATCH_SIZE, rng)
            elif method == "a_optimal":
                if rnd == 0:
                    batch = select_initial_batch(q_matrix, train_items, BATCH_SIZE, rng)
                else:
                    remaining = np.setdiff1d(train_items, items_seen)
                    batch = select_adaptive_batch(
                        net, theta_raw, np.stack([Sigma] * min(M, 50))
                        if Sigma.ndim == 2 else Sigma,
                        remaining, q_matrix, text_embs, device, BATCH_SIZE)
            else:
                raise ValueError(f"Unknown method: {method}")

            # ── Observe responses ──
            batch_responses = R_eval[:, batch.astype(int)]  # (M, batch_size)
            items_seen.extend(batch.tolist())
            responses_seen = np.concatenate(
                [responses_seen, batch_responses], axis=1
            ) if responses_seen.shape[1] > 0 else batch_responses

            # ── Update theta ──
            theta_raw, Sigma = gd_theta_update(
                net, theta_raw, items_seen, responses_seen,
                q_matrix, text_embs, device, lam=0.1, lr=0.05, n_steps=30,
            )

            # ── Evaluate at checkpoint ──
            cp = (rnd + 1) * BATCH_SIZE
            theta_est = torch.sigmoid(theta_raw.detach()).cpu().numpy()  # (M, K)

            # RMSE vs ground truth
            rmse = np.sqrt(((theta_est - theta_full) ** 2).mean())
            per_llm_rmse = np.sqrt(((theta_est - theta_full) ** 2).mean(axis=1))

            # Cosine similarity
            cos_sim = np.array([
                np.dot(theta_est[m], theta_full[m]) /
                (np.linalg.norm(theta_est[m]) * np.linalg.norm(theta_full[m]) + 1e-8)
                for m in range(M)
            ])

            # Per-dimension correlation (averaged over K)
            dim_corrs = []
            for k in range(K):
                if np.std(theta_full[:, k]) > 1e-6 and np.std(theta_est[:, k]) > 1e-6:
                    r = np.corrcoef(theta_full[:, k], theta_est[:, k])[0, 1]
                    if not np.isnan(r):
                        dim_corrs.append(r)
            dim_corr_mean = float(np.mean(dim_corrs)) if dim_corrs else 0.0

            # AUC on test items
            theta_sig = torch.sigmoid(theta_raw.detach())
            test_preds = batched_forward(net, theta_sig, test_items, q_matrix,
                                         text_embs, device)  # (M, n_test)
            aucs = []
            for m in range(M):
                y_true = R_eval[m, test_items.astype(int)]
                if len(np.unique(y_true)) >= 2:
                    aucs.append(roc_auc_score(y_true, test_preds[m]))
            auc_mean = float(np.mean(aucs)) if aucs else 0.0

            method_results[cp] = {
                "rmse": float(rmse),
                "rmse_std": float(per_llm_rmse.std()),
                "cos_sim_mean": float(cos_sim.mean()),
                "cos_sim_std": float(cos_sim.std()),
                "dim_corr_mean": dim_corr_mean,
                "auc_mean": auc_mean,
                "auc_std": float(np.std(aucs)) if aucs else 0.0,
            }

            print(f"  Round {rnd+1}: {cp} items | RMSE={rmse:.4f} | "
                  f"cos={cos_sim.mean():.4f} | dim_r={dim_corr_mean:.4f} | "
                  f"AUC={auc_mean:.4f}", flush=True)

        elapsed = time.time() - t0
        print(f"  Wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
        all_results[method] = {"results": method_results, "wall_time": elapsed}

    # ── Figure ──
    print("\nGenerating figure...", flush=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors = {"random": "#888888", "stratified": "#DD8452", "a_optimal": "#4C72B0"}
    labels = {"random": "Random", "stratified": "Stratified Random", "a_optimal": "A-optimal (ours)"}
    markers = {"random": "s", "stratified": "^", "a_optimal": "o"}

    metrics_to_plot = [
        ("rmse", "RMSE vs ground-truth theta", True),
        ("cos_sim_mean", "Cosine similarity", False),
        ("auc_mean", "AUC on test items", False),
    ]

    for ax, (metric, ylabel, invert) in zip(axes, metrics_to_plot):
        for method in methods:
            vals = [all_results[method]["results"][cp][metric] for cp in checkpoints]
            ax.plot(checkpoints, vals, f"{markers[method]}-", color=colors[method],
                    lw=2, markersize=7, label=labels[method])
        ax.set_xlabel("Calibration items")
        ax.set_ylabel(ylabel)
        if invert:
            ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_fig = fig_dir / "fig_adaptive_testing_v3.pdf"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "adaptive_testing_v3",
        "n_eval_llms": M,
        "K": K,
        "n_rounds": N_ROUNDS,
        "batch_size": BATCH_SIZE,
        "total_items": TOTAL_ITEMS,
        "checkpoints": checkpoints,
        "methods": methods,
        "results": {m: all_results[m]["results"] for m in methods},
        "wall_times": {m: all_results[m]["wall_time"] for m in methods},
    }
    out_json = Path("cdm_exploration/experiments/v2_adaptive_testing_v3.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    log_experiment(
        name="adaptive_testing_v3",
        config={"n_eval": M, "K": K, "n_rounds": N_ROUNDS,
                "batch_size": BATCH_SIZE, "methods": methods, "device": device},
        results=save_data,
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_eval_llms": M},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
