"""Response matrix and Q-matrix loading, plus triplet construction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_response_matrix(
    path: str | Path,
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Load the binary response matrix CSV.

    Args:
        path: Path to ``response_matrix.csv`` (index_col=0 expected).

    Returns:
        A tuple of ``(response_df, n_students, n_items, student_names)``.
    """
    response_df = pd.read_csv(path, index_col=0)
    n_students = response_df.shape[0]
    n_items = response_df.shape[1]
    student_names = list(response_df.index)
    return response_df, n_students, n_items, student_names


def load_q_matrix(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a Q-matrix CSV, separating metadata columns from skill columns.

    Metadata columns (``item_idx``, ``source``) are dropped; only binary
    skill columns are returned.

    Args:
        path: Path to a Q-matrix CSV (e.g. ``q_matrix_hac50.csv``).

    Returns:
        A tuple of ``(q_matrix, skill_names)`` where *q_matrix* is a
        float array of shape ``(n_items, n_skills)``.
    """
    q_df = pd.read_csv(path)
    meta_cols = [c for c in ["item_idx", "source"] if c in q_df.columns]
    skill_cols = [c for c in q_df.columns if c not in meta_cols]
    q_matrix = q_df[skill_cols].values.astype(float)
    return q_matrix, skill_cols


def build_triplets(response_df: pd.DataFrame) -> np.ndarray:
    """Convert a response matrix into ``(student_id, item_id, score)`` triplets.

    Args:
        response_df: DataFrame with students as rows, items as columns.

    Returns:
        Array of shape ``(n_students * n_items, 3)``.
    """
    n_students = response_df.shape[0]
    n_items = response_df.shape[1]
    values = response_df.values  # (n_students, n_items)

    student_ids = np.repeat(np.arange(n_students), n_items)
    item_ids = np.tile(np.arange(n_items), n_students)
    scores = values.ravel().astype(float)

    return np.column_stack([student_ids, item_ids, scores])
