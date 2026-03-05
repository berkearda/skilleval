"""Training loops for ID-based and text-conditioned NCDM models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import eval_id_model, eval_text_model

# Lazy import: EduCDM is an external repo added to sys.path by scripts.
# We import it at call time to avoid hard-wiring the path here.


def train_id_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_skills: int,
    n_items: int,
    n_students: int,
    epochs: int = 15,
    lr: float = 0.002,
    device: str = "cpu",
):
    """Train a standard ID-based NCDM and return the model.

    Uses best-validation-AUC checkpointing.

    Args:
        train_loader: Training DataLoader (``user_id, item_id, knowledge_emb, y``).
        val_loader: Validation DataLoader.
        n_skills: Number of skills (Q-matrix columns).
        n_items: Total number of items.
        n_students: Total number of students / LLMs.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        device: Torch device string.

    Returns:
        Trained :class:`EduCDM.NCDM` model with best-validation weights restored.
    """
    from EduCDM import NCDM

    model = NCDM(n_skills, n_items, n_students)
    model.ncdm_net = model.ncdm_net.to(device)
    optimizer = torch.optim.Adam(model.ncdm_net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.ncdm_net.train()
        losses = []

        for user_id, item_id, knowledge_emb, y in tqdm(
            train_loader, desc=f"  Epoch {epoch + 1}/{epochs}", leave=False
        ):
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)

            pred = model.ncdm_net(user_id, item_id, knowledge_emb)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        val_auc, val_acc, val_rmse = eval_id_model(model, val_loader, device)
        print(
            f"    loss={avg_loss:.4f}, val_auc={val_auc:.4f}, "
            f"val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.ncdm_net.state_dict().items()}

    if best_state:
        model.ncdm_net.load_state_dict(best_state)
        print(f"  Restored best model (val_auc={best_auc:.4f})")

    return model


def train_text_model(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.002,
    device: str = "cpu",
) -> nn.Module:
    """Train a :class:`~cdmeval.modeling.text_conditioned.TextConditionedNet`.

    Uses best-validation-AUC checkpointing.

    Args:
        net: A ``TextConditionedNet`` instance (uninitialised is fine).
        train_loader: Training DataLoader (``user_id, text_emb, knowledge_emb, y``).
        val_loader: Validation DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        device: Torch device string.

    Returns:
        The same *net* with best-validation weights restored.
    """
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        net.train()
        losses = []

        for user_id, text_emb, knowledge_emb, y in tqdm(
            train_loader, desc=f"  Epoch {epoch + 1}/{epochs}", leave=False
        ):
            user_id = user_id.to(device)
            text_emb = text_emb.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)

            pred = net(user_id, text_emb, knowledge_emb)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        val_auc, val_acc, val_rmse = eval_text_model(net, val_loader, device)
        print(
            f"    loss={avg_loss:.4f}, val_auc={val_auc:.4f}, "
            f"val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict(best_state)
        print(f"  Restored best model (val_auc={best_auc:.4f})")

    return net
