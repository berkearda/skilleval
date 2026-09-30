"""Skill-level mastery scores that do not depend on where theta sits on its scale.

The NCDM predicts from theta - d only, so shifting theta_k and every item's d_k by the same amount leaves every
prediction unchanged: "theta_k > 0.5" is not identified (an external review, 2026-09-22). The two scores here are
unaffected by such a shift:

- predicted: mean predicted probability of a correct answer over the items assigned to skill k
  (written by tools/predict_skill_accuracy.py);
- observed: the LLM's observed accuracy over the same items (no model involved).
"""

from pathlib import Path

import numpy as np

PREDICTED_PATH = Path("cdm_exploration/experiments/v2_predicted_skill_accuracy.npz")
SOURCES = ("theta", "predicted", "observed")


def skill_accuracy(values: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Mean of ``values`` (n_llms, n_items) over each skill's items, giving (n_llms, K)."""
    q = (q_matrix > 0).astype(np.float64)
    n_per_skill = q.sum(axis=0)
    assert (n_per_skill > 0).all(), "every skill needs at least one item"
    return (values.astype(np.float64) @ q) / n_per_skill


def skill_scores(source: str, theta: np.ndarray, R: np.ndarray, q_matrix: np.ndarray,
                 predicted_path: Path = PREDICTED_PATH) -> np.ndarray:
    """Per-LLM, per-skill score used for mastery: theta itself, predicted skill accuracy or observed skill accuracy."""
    assert source in SOURCES, f"mastery source must be one of {SOURCES}, got {source!r}"
    if source == "theta":
        return theta
    if source == "observed":
        return skill_accuracy(R, q_matrix)
    pred = np.load(predicted_path)["pred_skill_acc"]
    assert pred.shape == theta.shape, f"{predicted_path} has shape {pred.shape}, expected {theta.shape}"
    return pred
