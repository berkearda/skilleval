"""Experiment verification, logging, and checkpoint utilities."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════════════
#  Split verification
# ════════════════════════════════════════════════════════════════════


def verify_splits(
    train_items: np.ndarray,
    test_items: np.ndarray,
    eval_items: np.ndarray | None = None,
    calibration_items: np.ndarray | None = None,
    expected_seed: int = 42,
    label: str = "experiment",
) -> bool:
    """Verify that train/test splits are correct and non-overlapping.

    Prints PASS/FAIL for each check. Returns True if all checks pass.

    Args:
        train_items: Training item indices.
        test_items: Test item indices.
        eval_items: Items used for evaluation (should be subset of test_items).
        calibration_items: Items used for calibration (should be subset of train_items).
        expected_seed: Expected random_state for the split.
        label: Name of the experiment for logging.
    """
    print(f"\n  Split verification [{label}]:")
    all_pass = True

    # Check 1: no overlap
    overlap = np.intersect1d(train_items, test_items)
    if len(overlap) == 0:
        print(f"    PASS: train ({len(train_items)}) and test ({len(test_items)}) do not overlap")
    else:
        print(f"    FAIL: {len(overlap)} items in both train and test!")
        all_pass = False

    # Check 2: eval items are subset of test
    if eval_items is not None:
        leak = np.setdiff1d(eval_items, test_items)
        if len(leak) == 0:
            print(f"    PASS: all {len(eval_items)} eval items are in test set")
        else:
            print(f"    FAIL: {len(leak)} eval items NOT in test set (data leakage!)")
            all_pass = False

    # Check 3: calibration items are subset of train
    if calibration_items is not None:
        leak = np.setdiff1d(calibration_items, train_items)
        if len(leak) == 0:
            print(f"    PASS: all {len(calibration_items)} calibration items are in train set")
        else:
            print(f"    FAIL: {len(leak)} calibration items NOT in train set (data leakage!)")
            all_pass = False

    # Check 4: reproducibility (verify the split matches expected seed)
    from sklearn.model_selection import train_test_split
    n_total = len(train_items) + len(test_items)
    expected_train, expected_test = train_test_split(
        np.arange(n_total), test_size=len(test_items) / n_total,
        random_state=expected_seed,
    )
    if np.array_equal(np.sort(train_items), np.sort(expected_train)):
        print(f"    PASS: split matches random_state={expected_seed}")
    else:
        print(f"    WARN: split does not match random_state={expected_seed} (may use different n_total)")

    if all_pass:
        print(f"    ALL CHECKS PASSED")
    else:
        print(f"    SOME CHECKS FAILED")

    return all_pass


# ════════════════════════════════════════════════════════════════════
#  Experiment logging
# ════════════════════════════════════════════════════════════════════


def _get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def log_experiment(
    name: str,
    config: dict,
    results: dict,
    split_info: dict,
    verified: bool,
    log_dir: str | Path = "cdm_exploration/experiments",
) -> None:
    """Append an experiment entry to the experiment log.

    Args:
        name: Experiment name (e.g. "baseline_comparison", "cold_start").
        config: Dict with K, epochs, lr, device, seed, etc.
        results: Dict with AUC, Acc@k, etc.
        split_info: Dict with n_train_items, n_test_items, n_train_llms, etc.
        verified: Whether verify_splits() passed.
        log_dir: Directory for the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "experiment_log.json"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": name,
        "git_commit": _get_git_hash(),
        "config": config,
        "split_info": split_info,
        "results": results,
        "verified": verified,
    }

    # Load existing log or create new
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    status = "VERIFIED" if verified else "UNVERIFIED"
    print(f"\n  Logged [{status}]: {name} -> {log_path}")


# ════════════════════════════════════════════════════════════════════
#  Checkpoint saving / loading
# ════════════════════════════════════════════════════════════════════


def save_checkpoint(
    model,
    path: str | Path,
    config: dict,
    train_items: np.ndarray,
    test_items: np.ndarray,
    val_auc: float,
    epoch: int,
) -> None:
    """Save a model checkpoint with metadata.

    Args:
        model: PyTorch model (nn.Module).
        path: Output path for the .pt file.
        config: Training config dict.
        train_items: Training item indices.
        test_items: Test item indices.
        val_auc: Best validation AUC.
        epoch: Epoch number of best checkpoint.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "train_items": train_items.tolist(),
        "test_items": test_items.tolist(),
        "val_auc": val_auc,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_hash(),
    }, str(path))
    print(f"  Checkpoint saved: {path} (val_auc={val_auc:.4f}, epoch={epoch})")


def load_checkpoint(path: str | Path, model=None, device: str = "cpu"):
    """Load a checkpoint. Optionally load weights into a model.

    Args:
        path: Path to .pt checkpoint.
        model: If provided, load state_dict into this model.
        device: Device to map tensors to.

    Returns:
        Dict with all checkpoint metadata. If model is provided,
        weights are loaded in-place.
    """
    import torch

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if model is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Loaded checkpoint: {path} (val_auc={ckpt.get('val_auc', '?')})")
    return ckpt
