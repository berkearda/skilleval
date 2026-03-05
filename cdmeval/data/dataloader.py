"""PyTorch DataLoader factories for CDM training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(
    triplets: np.ndarray,
    q_matrix: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for the ID-based NCDM.

    Each sample is ``(user_id, item_id, knowledge_emb, score)``.

    Args:
        triplets: Array of shape ``(N, 3)`` with columns
            ``[student_id, item_id, score]``.
        q_matrix: Binary array of shape ``(n_items, n_skills)``.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples.

    Returns:
        A :class:`~torch.utils.data.DataLoader`.
    """
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    item_ids = torch.tensor(triplets[:, 1], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    item_indices = triplets[:, 1].astype(int)
    knowledge_embs = torch.tensor(q_matrix[item_indices], dtype=torch.float32)

    dataset = TensorDataset(user_ids, item_ids, knowledge_embs, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_text_dataloader(
    triplets: np.ndarray,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for the text-conditioned NCDM.

    Each sample is ``(user_id, text_emb, knowledge_emb, score)``.

    Args:
        triplets: Array of shape ``(N, 3)`` with columns
            ``[student_id, item_id, score]``.
        text_embeddings: Array of shape ``(n_items, text_dim)`` with
            pre-computed SBERT embeddings.
        q_matrix: Binary array of shape ``(n_items, n_skills)``.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples.

    Returns:
        A :class:`~torch.utils.data.DataLoader`.
    """
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    item_indices = triplets[:, 1].astype(int)
    text_embs = torch.tensor(text_embeddings[item_indices], dtype=torch.float32)
    knowledge_embs = torch.tensor(q_matrix[item_indices], dtype=torch.float32)

    dataset = TensorDataset(user_ids, text_embs, knowledge_embs, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
