"""Device selection and reproducibility utilities."""

import random

import numpy as np
import torch


def resolve_device(device: str) -> str:
    """Resolve requested device to an available one, falling back to CPU.

    Args:
        device: One of ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        The validated device string (may be ``"cpu"`` if the requested
        accelerator is unavailable).
    """
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return device


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
