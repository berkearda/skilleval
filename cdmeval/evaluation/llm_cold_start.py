"""Cold-start evaluation: profile new LLMs from a small number of responses."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# ════════════════════════════════════════════════════════════════════
#  LLM-level split
# ════════════════════════════════════════════════════════════════════


def llm_cold_start_split(
    n_llms: int,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fraction of LLMs entirely from training.

    Returns:
        ``(train_llm_indices, test_llm_indices)``
    """
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(
        all_llms, test_size=test_fraction, random_state=seed
    )
    return train_llms, test_llms


# ════════════════════════════════════════════════════════════════════
#  Fit a new LLM's mastery profile
# ════════════════════════════════════════════════════════════════════


def fit_new_llm_profile(
    trained_net: nn.Module,
    responses: np.ndarray,
    calibration_items: np.ndarray,
    q_matrix: np.ndarray,
    text_embeddings: np.ndarray,
    device: str = "cpu",
    lr: float = 0.01,
    epochs: int = 100,
) -> np.ndarray:
    """Fit a mastery vector for a new LLM by optimising only theta.

    All model parameters (item projections, PosLinear weights) are frozen.
    Only a fresh 1×K embedding is trained on the calibration items.

    Args:
        trained_net: A trained ``TextConditionedNet`` (will NOT be modified).
        responses: ``(n_items,)`` binary responses for this LLM.
        calibration_items: Item indices to use for fitting.
        q_matrix: ``(n_items, K)`` binary Q-matrix.
        text_embeddings: ``(n_items, text_dim)`` SBERT embeddings.
        device: Torch device.
        lr: Learning rate for theta optimisation.
        epochs: Number of gradient steps.

    Returns:
        Mastery vector ``(K,)`` in [0, 1].
    """
    trained_net.eval()
    trained_net = trained_net.to(device)
    K = trained_net.knowledge_dim

    # Prepare calibration data
    cal_items = calibration_items.astype(int)
    y = torch.tensor(responses[cal_items], dtype=torch.float32, device=device)
    text_emb = torch.tensor(
        text_embeddings[cal_items], dtype=torch.float32, device=device
    )
    q_rows = torch.tensor(q_matrix[cal_items], dtype=torch.float32, device=device)

    # Fresh theta for this LLM (raw logits, will be sigmoided)
    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    # Pre-compute item difficulty/discrimination (frozen)
    with torch.no_grad():
        k_difficulty = torch.sigmoid(trained_net.k_difficulty_proj(text_emb))
        e_difficulty = torch.sigmoid(trained_net.e_difficulty_proj(text_emb))

    for _ in range(epochs):
        stat_emb = torch.sigmoid(theta_raw).expand(len(cal_items), -1)
        input_x = e_difficulty * (stat_emb - k_difficulty) * q_rows

        # Forward through frozen PosLinear layers (no dropout at eval)
        with torch.no_grad():
            h1 = torch.sigmoid(trained_net.prednet_full1(input_x))
            h2 = torch.sigmoid(trained_net.prednet_full2(h1))
            # We need gradients through theta, so we can't fully detach.
            # Recompute with grad enabled for theta path only.

        # Full forward with grad for theta
        stat_emb = torch.sigmoid(theta_raw).expand(len(cal_items), -1)
        input_x = e_difficulty * (stat_emb - k_difficulty) * q_rows
        h1 = torch.sigmoid(trained_net.prednet_full1(input_x))
        h2 = torch.sigmoid(trained_net.prednet_full2(h1))
        pred = torch.sigmoid(trained_net.prednet_full3(h2)).squeeze(-1)

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mastery = torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()
    return mastery


def _predict_with_theta(
    trained_net: nn.Module,
    theta_mastery: np.ndarray,
    item_indices: np.ndarray,
    q_matrix: np.ndarray,
    text_embeddings: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Predict P(correct) for a single LLM (given its mastery) on specified items."""
    trained_net.eval()
    trained_net = trained_net.to(device)

    items = item_indices.astype(int)
    text_emb = torch.tensor(text_embeddings[items], dtype=torch.float32, device=device)
    q_rows = torch.tensor(q_matrix[items], dtype=torch.float32, device=device)
    stat_emb = torch.tensor(
        theta_mastery, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(len(items), -1)

    with torch.no_grad():
        k_difficulty = torch.sigmoid(trained_net.k_difficulty_proj(text_emb))
        e_difficulty = torch.sigmoid(trained_net.e_difficulty_proj(text_emb))
        input_x = e_difficulty * (stat_emb - k_difficulty) * q_rows
        h1 = torch.sigmoid(trained_net.prednet_full1(input_x))
        h2 = torch.sigmoid(trained_net.prednet_full2(h1))
        pred = torch.sigmoid(trained_net.prednet_full3(h2)).squeeze(-1)

    return pred.cpu().numpy()


# ════════════════════════════════════════════════════════════════════
#  Full cold-start evaluation
# ════════════════════════════════════════════════════════════════════


def evaluate_llm_cold_start(
    trained_net: nn.Module,
    test_llm_indices: np.ndarray,
    response_matrix: np.ndarray,
    q_matrix: np.ndarray,
    text_embeddings: np.ndarray,
    calibration_pool: np.ndarray,
    eval_items: np.ndarray,
    calibration_sizes: list[int],
    device: str = "cpu",
    n_repeats: int = 5,
    seed: int = 42,
    lr: float = 0.01,
    fit_epochs: int = 100,
) -> pd.DataFrame:
    """Evaluate cold-start LLM profiling across calibration sizes.

    Calibration items are sampled from *calibration_pool* (training items).
    AUC is always evaluated on *eval_items* (held-out test items) to avoid
    data leakage.

    Args:
        trained_net: Trained ``TextConditionedNet`` (frozen for this eval).
        test_llm_indices: Held-out LLM indices.
        response_matrix: ``(n_llms, n_items)`` binary matrix.
        q_matrix: ``(n_items, K)`` binary Q-matrix.
        text_embeddings: ``(n_items, text_dim)`` SBERT embeddings.
        calibration_pool: Item indices to sample calibration items from
            (should be training items only).
        eval_items: Item indices to evaluate AUC on (should be held-out
            test items only, disjoint from calibration_pool).
        calibration_sizes: List of N values to test.
        device: Torch device.
        n_repeats: Number of random calibration samples per LLM per N.
        seed: Base random seed.
        lr: Learning rate for theta fitting.
        fit_epochs: Gradient steps for theta fitting.

    Returns:
        DataFrame with columns
        ``[llm_idx, n_calibration, repeat, auc, acc, n_eval_items]``.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for li, llm_idx in enumerate(test_llm_indices):
        responses = response_matrix[llm_idx]
        print(f"  LLM {li + 1}/{len(test_llm_indices)} (idx={llm_idx})")

        for N in calibration_sizes:
            for rep in range(n_repeats):
                if N == 0:
                    # No calibration — use default theta
                    cal = np.array([], dtype=int)
                elif N >= len(calibration_pool):
                    cal = calibration_pool.copy()
                else:
                    cal = rng.choice(calibration_pool, size=N, replace=False)

                if len(eval_items) < 10:
                    continue

                # Fit theta on calibration set (or use default for N=0)
                if N == 0:
                    K = trained_net.knowledge_dim
                    mastery = np.full(K, 0.5)
                else:
                    mastery = fit_new_llm_profile(
                        trained_net, responses, cal, q_matrix,
                        text_embeddings, device, lr=lr, epochs=fit_epochs,
                    )

                # Evaluate on held-out test items (never seen during cal)
                preds = _predict_with_theta(
                    trained_net, mastery, eval_items,
                    q_matrix, text_embeddings, device,
                )
                y_true = responses[eval_items.astype(int)]

                # Skip if single-class
                if len(np.unique(y_true)) < 2:
                    continue

                auc = roc_auc_score(y_true, preds)
                acc = accuracy_score(y_true, (preds >= 0.5).astype(int))

                rows.append({
                    "llm_idx": int(llm_idx),
                    "n_calibration": N,
                    "repeat": rep,
                    "auc": auc,
                    "acc": acc,
                    "n_eval_items": len(eval_items),
                })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  Visualization
# ════════════════════════════════════════════════════════════════════


def plot_cold_start_curve(
    results_df: pd.DataFrame,
    full_train_auc: float,
    fig_dir: Path,
) -> None:
    """Plot calibration size vs AUC with error bands."""
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()

    # Aggregate: mean/std AUC per calibration size
    agg = results_df.groupby("n_calibration")["auc"].agg(["mean", "std"]).reset_index()
    agg.columns = ["n_calibration", "auc_mean", "auc_std"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Error band
    ax.fill_between(
        agg["n_calibration"],
        agg["auc_mean"] - agg["auc_std"],
        agg["auc_mean"] + agg["auc_std"],
        alpha=0.2, color="#4C72B0",
    )
    ax.plot(
        agg["n_calibration"], agg["auc_mean"],
        "o-", color="#4C72B0", lw=2, markersize=6, label="Cold-start LLM",
    )

    # Reference lines
    ax.axhline(
        full_train_auc, ls="--", color="#55A868", lw=1.5,
        label=f"Full training AUC ({full_train_auc:.3f})",
    )
    ax.axhline(0.5, ls=":", color="#999999", lw=1, label="Random guess")

    # 90% of full training AUC
    threshold = full_train_auc * 0.9
    ax.axhline(
        threshold, ls="--", color="#DD8452", lw=1, alpha=0.7,
        label=f"90% of full ({threshold:.3f})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of calibration items")
    ax.set_ylabel("Mean AUC on remaining items")
    ax.set_title("LLM Cold-Start: Skill Profiling from Few Responses")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(agg["n_calibration"].tolist())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    out = fig_dir / "fig_llm_cold_start.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")
