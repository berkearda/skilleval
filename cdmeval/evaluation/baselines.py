"""Routing baselines for comparison against CDM-based routing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════
#  Shared evaluation
# ════════════════════════════════════════════════════════════════════


def evaluate_rankings(
    rankings: dict[int, np.ndarray],
    response_matrix: np.ndarray,
    ks: list[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Evaluate routing accuracy from per-item LLM rankings.

    Args:
        rankings: ``{item_idx: array of LLM indices sorted by predicted quality}``
        response_matrix: ``(n_llms, n_items)`` binary correctness matrix.
        ks: Values of k for Acc@k.

    Returns:
        Dict with ``accuracy@k`` for each k, plus ``n_items``.
    """
    accs = {k: 0 for k in ks}
    total = 0
    for item_idx, ranking in rankings.items():
        gt = response_matrix[:, int(item_idx)]
        if gt.sum() == 0:
            continue
        total += 1
        for k in ks:
            if gt[ranking[:k]].sum() > 0:
                accs[k] += 1
    if total == 0:
        result = {f"accuracy@{k}": 0.0 for k in ks}
        result["n_items"] = 0
        return result
    result = {f"accuracy@{k}": accs[k] / total for k in ks}
    result["n_items"] = total
    return result


# ════════════════════════════════════════════════════════════════════
#  Naive baselines
# ════════════════════════════════════════════════════════════════════


def random_router(
    test_items: np.ndarray,
    n_llms: int,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Random ranking of LLMs for each test item."""
    rng = np.random.RandomState(seed)
    rankings = {}
    for item_idx in test_items:
        rankings[int(item_idx)] = rng.permutation(n_llms)
    return rankings


def majority_router(
    test_items: np.ndarray,
    response_matrix: np.ndarray,
    train_items: np.ndarray,
) -> dict[int, np.ndarray]:
    """Always pick the globally best LLM (by train-set accuracy)."""
    train_acc = response_matrix[:, train_items.astype(int)].mean(axis=1)
    global_ranking = np.argsort(-train_acc)
    return {int(idx): global_ranking.copy() for idx in test_items}


# ════════════════════════════════════════════════════════════════════
#  Embedding-based baselines
# ════════════════════════════════════════════════════════════════════


def sbert_nearest_neighbor_router(
    test_items: np.ndarray,
    train_items: np.ndarray,
    text_embeddings: np.ndarray,
    response_matrix: np.ndarray,
    k: int = 5,
) -> dict[int, np.ndarray]:
    """k-NN router: find nearest training items, pick most frequent solver.

    For each test item, find k nearest training items by SBERT cosine
    similarity. Rank LLMs by their total correct count on those k items.
    """
    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn_model.fit(text_embeddings[train_items.astype(int)])

    n_llms = response_matrix.shape[0]
    rankings = {}

    for item_idx in test_items:
        emb = text_embeddings[int(item_idx)].reshape(1, -1)
        dists, indices = nn_model.kneighbors(emb)
        neighbor_items = train_items[indices[0]]

        # Count how many of the k neighbors each LLM solved
        scores = response_matrix[:, neighbor_items.astype(int)].sum(axis=1)
        rankings[int(item_idx)] = np.argsort(-scores)

    return rankings


def text_similarity_router(
    test_items: np.ndarray,
    train_items: np.ndarray,
    text_embeddings: np.ndarray,
    response_matrix: np.ndarray,
    k: int = 5,
) -> dict[int, np.ndarray]:
    """Weighted k-NN router using cosine similarity as weights.

    For each test item, find k nearest training items. Compute a weighted
    average of each LLM's correctness on those items (weight = similarity).
    """
    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn_model.fit(text_embeddings[train_items.astype(int)])

    rankings = {}

    for item_idx in test_items:
        emb = text_embeddings[int(item_idx)].reshape(1, -1)
        dists, indices = nn_model.kneighbors(emb)
        neighbor_items = train_items[indices[0]]
        similarities = 1.0 - dists[0]  # cosine distance -> similarity

        # Weighted average correctness per LLM
        correctness = response_matrix[:, neighbor_items.astype(int)]  # (n_llms, k)
        scores = correctness @ similarities  # (n_llms,)
        rankings[int(item_idx)] = np.argsort(-scores)

    return rankings


# ════════════════════════════════════════════════════════════════════
#  IRT 2PL baseline
# ════════════════════════════════════════════════════════════════════


class IRT2PL(nn.Module):
    """Simple 2-parameter logistic IRT model.

    P(correct | m, j) = sigmoid(alpha_j * theta_m - beta_j)

    One scalar ability theta per LLM, one discrimination alpha and
    difficulty beta per item.
    """

    def __init__(self, n_students: int, n_items: int):
        super().__init__()
        self.theta = nn.Embedding(n_students, 1)
        self.alpha = nn.Embedding(n_items, 1)
        self.beta = nn.Embedding(n_items, 1)
        nn.init.normal_(self.theta.weight, 0, 0.1)
        nn.init.ones_(self.alpha.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, student_id, item_id):
        theta = self.theta(student_id)           # (batch, 1)
        alpha = torch.exp(self.alpha(item_id))    # positive discrimination
        beta = self.beta(item_id)                 # (batch, 1)
        logit = alpha * theta - beta
        return torch.sigmoid(logit).squeeze(-1)


def _train_irt(
    model: IRT2PL,
    triplets: np.ndarray,
    val_triplets: np.ndarray,
    epochs: int = 15,
    lr: float = 0.002,
    batch_size: int = 64,
    device: str = "cpu",
) -> IRT2PL:
    """Train IRT 2PL with best-validation-AUC checkpointing."""
    from sklearn.metrics import roc_auc_score

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Build tensors
    def make_tensors(trips):
        s = torch.tensor(trips[:, 0], dtype=torch.int64)
        i = torch.tensor(trips[:, 1], dtype=torch.int64)
        y = torch.tensor(trips[:, 2], dtype=torch.float32)
        return s, i, y

    s_tr, i_tr, y_tr = make_tensors(triplets)
    s_va, i_va, y_va = make_tensors(val_triplets)

    n_train = len(triplets)
    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            s_b = s_tr[idx].to(device)
            i_b = i_tr[idx].to(device)
            y_b = y_tr[idx].to(device)

            pred = model(s_b, i_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            va_pred = []
            for start in range(0, len(val_triplets), batch_size):
                end = min(start + batch_size, len(val_triplets))
                p = model(
                    s_va[start:end].to(device), i_va[start:end].to(device)
                )
                va_pred.extend(p.cpu().tolist())
            va_auc = roc_auc_score(y_va.numpy(), np.array(va_pred))

        avg_loss = np.mean(losses)
        print(
            f"    loss={avg_loss:.4f}, val_auc={va_auc:.4f}"
        )
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best IRT model (val_auc={best_auc:.4f})")
    return model


def irt_2pl_router(
    response_matrix: np.ndarray,
    train_items: np.ndarray,
    test_items: np.ndarray,
    n_llms: int,
    n_items: int,
    epochs: int = 15,
    lr: float = 0.002,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Train IRT 2PL on training items, route test items by predicted P(correct).

    Args:
        response_matrix: ``(n_llms, n_items)`` binary matrix.
        train_items: Item indices used for training.
        test_items: Held-out item indices for routing.
        n_llms: Number of LLMs.
        n_items: Total number of items.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        device: Torch device.

    Returns:
        Rankings dict ``{item_idx: sorted LLM indices}``.
    """
    from sklearn.model_selection import train_test_split

    # Build triplets from training items only
    student_ids = np.repeat(np.arange(n_llms), len(train_items))
    item_ids = np.tile(train_items.astype(int), n_llms)
    scores = response_matrix[
        np.repeat(np.arange(n_llms), len(train_items)),
        np.tile(train_items.astype(int), n_llms),
    ].astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])

    # Split train/val
    idx = np.arange(len(triplets))
    tr_idx, va_idx = train_test_split(idx, test_size=0.1, random_state=42)

    print("  Training IRT 2PL...")
    model = IRT2PL(n_llms, n_items)
    model = _train_irt(
        model, triplets[tr_idx], triplets[va_idx],
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
    )

    # Route: for each test item, predict P(correct) for all LLMs
    model.eval()
    model = model.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    rankings = {}
    with torch.no_grad():
        for item_idx in test_items:
            item_t = torch.full((n_llms,), int(item_idx), dtype=torch.int64, device=device)
            preds = model(all_llm_ids, item_t).cpu().numpy()
            rankings[int(item_idx)] = np.argsort(-preds)

    return rankings


# ════════════════════════════════════════════════════════════════════
#  CDM router (wraps existing text-conditioned model)
# ════════════════════════════════════════════════════════════════════


def cdm_router(
    net: nn.Module,
    test_items: np.ndarray,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    n_llms: int,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Route using the text-conditioned NCDM (our method)."""
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    rankings = {}
    with torch.no_grad():
        for item_idx in test_items:
            emb = torch.tensor(
                text_embeddings[int(item_idx)], dtype=torch.float32, device=device
            )
            emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
            q_row = torch.tensor(
                q_matrix[int(item_idx)], dtype=torch.float32, device=device
            )
            q_batch = q_row.unsqueeze(0).expand(n_llms, -1)
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()
            rankings[int(item_idx)] = np.argsort(-preds)

    return rankings
