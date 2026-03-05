"""Evaluation helpers: metrics computation and model evaluation loops."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Compute AUC, accuracy, and RMSE.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted probabilities in [0, 1].

    Returns:
        A tuple of ``(auc, accuracy, rmse)``.
    """
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return auc, acc, rmse


def eval_id_model(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Evaluate a standard ID-based NCDM (EduCDM wrapper).

    Expects *model* to expose ``model.ncdm_net``.

    Args:
        model: An :class:`EduCDM.NCDM` instance.
        dataloader: DataLoader yielding ``(user_id, item_id, knowledge_emb, y)``.
        device: Torch device string.

    Returns:
        ``(auc, accuracy, rmse)``
    """
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, item_id, knowledge_emb, y in dataloader:
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            pred = model.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())

    return compute_metrics(np.array(y_true), np.array(y_pred))


def eval_text_model(
    net: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Evaluate a :class:`~cdmeval.modeling.text_conditioned.TextConditionedNet`.

    Args:
        net: A ``TextConditionedNet`` instance.
        dataloader: DataLoader yielding ``(user_id, text_emb, knowledge_emb, y)``.
        device: Torch device string.

    Returns:
        ``(auc, accuracy, rmse)``
    """
    net.eval()
    net = net.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, text_emb, knowledge_emb, y in dataloader:
            user_id = user_id.to(device)
            text_emb = text_emb.to(device)
            knowledge_emb = knowledge_emb.to(device)
            pred = net(user_id, text_emb, knowledge_emb)
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())

    return compute_metrics(np.array(y_true), np.array(y_pred))
