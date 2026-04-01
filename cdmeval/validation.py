"""Core validation framework for CDMEval experiments.

Every experiment should call these checks. Failures raise ValueError
with detailed messages so bugs are caught early, not after hours of training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════════════
#  Data validation
# ════════════════════════════════════════════════════════════════════


def validate_data(
    response_matrix: np.ndarray,
    q_matrix: np.ndarray,
    text_embeddings: np.ndarray,
    item_metadata: list | None = None,
    llm_names: list | None = None,
) -> None:
    """Validate data consistency before training.

    Raises ``ValueError`` with a detailed message if any check fails.
    """
    n_llms, n_items = response_matrix.shape
    errors = []

    # Shapes
    if q_matrix.shape[0] != n_items:
        errors.append(f"Q-matrix rows ({q_matrix.shape[0]}) != items ({n_items})")
    if text_embeddings.shape[0] != n_items:
        errors.append(f"Text embeddings rows ({text_embeddings.shape[0]}) != items ({n_items})")
    if item_metadata is not None and len(item_metadata) != n_items:
        errors.append(f"Item metadata length ({len(item_metadata)}) != items ({n_items})")
    if llm_names is not None and len(llm_names) != n_llms:
        errors.append(f"LLM names length ({len(llm_names)}) != LLMs ({n_llms})")

    # Binary response matrix
    unique_vals = np.unique(response_matrix)
    if not np.all(np.isin(unique_vals, [0, 1])):
        errors.append(f"Response matrix not binary: unique values = {unique_vals[:10]}")

    # No NaN
    if np.isnan(response_matrix.astype(float)).any():
        errors.append("Response matrix contains NaN")
    if np.isnan(q_matrix.astype(float)).any():
        errors.append("Q-matrix contains NaN")
    if np.isnan(text_embeddings).any():
        errors.append("Text embeddings contain NaN")

    # Every item should have at least one skill
    zero_skill = (q_matrix.sum(axis=1) == 0).sum()
    if zero_skill > 0:
        errors.append(f"{zero_skill} items have zero skills in Q-matrix")

    # Q-matrix should be binary
    q_unique = np.unique(q_matrix)
    if not np.all(np.isin(q_unique, [0, 1])):
        errors.append(f"Q-matrix not binary: unique values = {q_unique[:10]}")

    if errors:
        raise ValueError("Data validation failed:\n  " + "\n  ".join(errors))

    print(f"  [VALIDATE] Data OK: {n_llms} LLMs x {n_items} items, "
          f"K={q_matrix.shape[1]}, emb_dim={text_embeddings.shape[1]}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  Split validation
# ════════════════════════════════════════════════════════════════════


def validate_split(
    train_items: np.ndarray,
    test_items: np.ndarray,
    n_items: int,
    random_state: int = 42,
) -> None:
    """Validate train/test split is correct and reproducible."""
    from sklearn.model_selection import train_test_split

    errors = []

    # No overlap
    overlap = np.intersect1d(train_items, test_items)
    if len(overlap) > 0:
        errors.append(f"{len(overlap)} items in both train and test!")

    # Completeness
    combined = np.union1d(train_items, test_items)
    if len(combined) != n_items:
        errors.append(f"Train+test = {len(combined)} items, expected {n_items}")

    # Reproducibility
    expected_train, expected_test = train_test_split(
        np.arange(n_items), test_size=len(test_items) / n_items,
        random_state=random_state,
    )
    if not np.array_equal(np.sort(train_items), np.sort(expected_train)):
        errors.append(f"Split does not match random_state={random_state}")

    if errors:
        raise ValueError("Split validation failed:\n  " + "\n  ".join(errors))

    print(f"  [VALIDATE] Split OK: {len(train_items)} train, {len(test_items)} test, "
          f"seed={random_state}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  Prediction validation
# ════════════════════════════════════════════════════════════════════


def validate_predictions(
    predictions: np.ndarray,
    n_llms: int,
    n_test_items: int,
) -> None:
    """Validate model predictions before computing metrics."""
    errors = []

    if predictions.shape != (n_llms, n_test_items):
        errors.append(f"Prediction shape {predictions.shape} != expected ({n_llms}, {n_test_items})")

    if np.isnan(predictions).any():
        n_nan = np.isnan(predictions).sum()
        errors.append(f"{n_nan} NaN values in predictions")

    if predictions.min() < -0.01 or predictions.max() > 1.01:
        errors.append(f"Predictions out of [0,1]: min={predictions.min():.4f}, max={predictions.max():.4f}")

    if errors:
        raise ValueError("Prediction validation failed:\n  " + "\n  ".join(errors))

    print(f"  [VALIDATE] Predictions OK: {predictions.shape}, "
          f"range=[{predictions.min():.4f}, {predictions.max():.4f}]", flush=True)


# ════════════════════════════════════════════════════════════════════
#  Metrics validation
# ════════════════════════════════════════════════════════════════════


def validate_metrics(
    results: dict,
    n_llms: int | None = None,
    random_baseline: float | None = None,
) -> None:
    """Validate experiment results are reasonable."""
    warnings = []

    # AUC should be > 0.5
    auc = results.get("test_auc")
    if auc is not None and auc <= 0.5:
        warnings.append(f"AUC={auc:.4f} <= 0.5 (worse than chance)")

    # Routing @1 should beat random
    acc1 = results.get("acc@1")
    if acc1 is not None:
        if random_baseline and acc1 <= random_baseline:
            warnings.append(f"@1={acc1:.4f} <= random={random_baseline:.4f}")
        if acc1 < 0 or acc1 > 1:
            warnings.append(f"@1={acc1:.4f} out of [0,1]")

    # All metric values in [0,1]
    for key in ["test_auc", "test_acc", "acc@1", "acc@3", "acc@5", "acc@10"]:
        val = results.get(key)
        if val is not None and (val < 0 or val > 1):
            warnings.append(f"{key}={val:.4f} out of [0,1]")

    # Routing should be monotonic: @1 <= @3 <= @5 <= @10
    prev = 0
    for key in ["acc@1", "acc@3", "acc@5", "acc@10"]:
        val = results.get(key)
        if val is not None:
            if val < prev - 0.001:
                warnings.append(f"@k not monotonic: {key}={val:.4f} < previous={prev:.4f}")
            prev = val

    if warnings:
        print("  [VALIDATE] Metrics WARNINGS:", flush=True)
        for w in warnings:
            print(f"    WARN: {w}", flush=True)
    else:
        print(f"  [VALIDATE] Metrics OK", flush=True)

    return len(warnings) == 0


# ════════════════════════════════════════════════════════════════════
#  Checkpoint validation
# ════════════════════════════════════════════════════════════════════


def validate_checkpoint(
    checkpoint_path: str | Path,
    expected_K: int | None = None,
    expected_n_llms: int | None = None,
) -> dict:
    """Validate a saved checkpoint is loadable and correct."""
    import torch

    path = Path(checkpoint_path)
    errors = []

    if not path.exists():
        raise ValueError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        errors.append("Missing model_state_dict")
    else:
        sd = ckpt["model_state_dict"]
        # Check for NaN in weights
        for name, param in sd.items():
            if torch.isnan(param).any():
                errors.append(f"NaN in weight: {name}")

        # Check shapes if expected values given
        if expected_K is not None and "k_difficulty_proj.weight" in sd:
            actual_K = sd["k_difficulty_proj.weight"].shape[0]
            if actual_K != expected_K:
                errors.append(f"K mismatch: checkpoint has {actual_K}, expected {expected_K}")

        if expected_n_llms is not None and "student_emb.weight" in sd:
            actual_n = sd["student_emb.weight"].shape[0]
            if actual_n != expected_n_llms:
                errors.append(f"n_llms mismatch: checkpoint has {actual_n}, expected {expected_n_llms}")

    if errors:
        raise ValueError(f"Checkpoint validation failed ({path}):\n  " + "\n  ".join(errors))

    config = ckpt.get("config", {})
    print(f"  [VALIDATE] Checkpoint OK: {path.name} "
          f"(K={config.get('K', '?')}, n_llms={config.get('n_llms', '?')})", flush=True)
    return ckpt


# ════════════════════════════════════════════════════════════════════
#  Full validation of all existing results
# ════════════════════════════════════════════════════════════════════


def run_full_validation(
    experiments_dir: str | Path = "cdm_exploration/experiments",
    checkpoints_dir: str | Path = "cdm_exploration/checkpoints",
) -> None:
    """Run ALL validation checks on all existing experiments and checkpoints."""
    import json

    experiments_dir = Path(experiments_dir)
    checkpoints_dir = Path(checkpoints_dir)

    print("=" * 70, flush=True)
    print("FULL VALIDATION", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    failed = 0

    # 1. Validate experiment log
    log_path = experiments_dir / "experiment_log.json"
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        print(f"\n1. Experiment log: {len(log)} entries", flush=True)
        unverified = [e for e in log if not e.get("verified", False)]
        if unverified:
            print(f"   FAIL: {len(unverified)} unverified entries", flush=True)
            failed += 1
        else:
            print(f"   PASS: all entries verified", flush=True)
            passed += 1
    else:
        print(f"\n1. Experiment log: NOT FOUND", flush=True)
        failed += 1

    # 2. Validate result JSON files
    print(f"\n2. Result files:", flush=True)
    for f in sorted(experiments_dir.glob("v2_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        ok = validate_metrics(data)
        if ok:
            passed += 1
        else:
            failed += 1
        # Check routing monotonicity
        has_routing = any(k.startswith("acc@") for k in data)
        if has_routing:
            vals = [data.get(f"acc@{k}", 0) for k in [1, 3, 5, 10] if f"acc@{k}" in data]
            if vals == sorted(vals):
                print(f"   {f.name}: routing monotonic PASS", flush=True)
                passed += 1
            else:
                print(f"   {f.name}: routing monotonic FAIL: {vals}", flush=True)
                failed += 1

    # 3. Validate checkpoints
    print(f"\n3. Checkpoints:", flush=True)
    for f in sorted(checkpoints_dir.rglob("*.pt")):
        try:
            validate_checkpoint(f, expected_n_llms=3811)
            passed += 1
        except ValueError as e:
            print(f"   FAIL: {e}", flush=True)
            failed += 1

    # 4. Cross-check K ablation consistency
    print(f"\n4. K ablation cross-check:", flush=True)
    k_results = {}
    for k in [50, 100, 150, 200, 300]:
        path = experiments_dir / f"v2_K{k}_results.json"
        if path.exists():
            with open(path) as fh:
                k_results[k] = json.load(fh)

    if len(k_results) >= 2:
        aucs = {k: r["test_auc"] for k, r in k_results.items()}
        auc_range = max(aucs.values()) - min(aucs.values())
        if auc_range < 0.05:
            print(f"   PASS: AUC stable across K (range={auc_range:.4f})", flush=True)
            passed += 1
        else:
            print(f"   WARN: AUC varies significantly (range={auc_range:.4f})", flush=True)

        # @1 should generally increase with K
        acc1s = {k: r["acc@1"] for k, r in k_results.items()}
        ks_sorted = sorted(acc1s.keys())
        if acc1s[ks_sorted[-1]] >= acc1s[ks_sorted[0]]:
            print(f"   PASS: @1 trends upward with K ({acc1s[ks_sorted[0]]:.4f} -> {acc1s[ks_sorted[-1]]:.4f})", flush=True)
            passed += 1
        else:
            print(f"   WARN: @1 decreases with K", flush=True)

    # 5. CDMEval vs IRT consistency
    print(f"\n5. CDMEval vs IRT:", flush=True)
    irt_path = experiments_dir / "v2_irt_baseline.json"
    baselines_path = experiments_dir / "v2_baselines.json"
    if irt_path.exists() and baselines_path.exists():
        with open(irt_path) as f:
            irt = json.load(f)
        with open(baselines_path) as f:
            bl = json.load(f)
        cdm_acc1 = bl.get("cdmeval", {}).get("acc1", 0)
        irt_acc1 = irt.get("routing_acc@1", 0)
        if cdm_acc1 > irt_acc1:
            print(f"   PASS: CDMEval @1 ({cdm_acc1:.4f}) > IRT @1 ({irt_acc1:.4f})", flush=True)
            passed += 1
        else:
            print(f"   WARN: IRT beats CDMEval", flush=True)

        # Both should beat random
        random_acc = bl.get("random_acc1", 0)
        if cdm_acc1 > random_acc and irt_acc1 > random_acc:
            print(f"   PASS: both beat random ({random_acc:.4f})", flush=True)
            passed += 1
        else:
            print(f"   FAIL: model doesn't beat random", flush=True)
            failed += 1

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed", flush=True)
    print(f"{'=' * 70}", flush=True)

    if failed > 0:
        print("ACTION REQUIRED: fix failures before submitting", flush=True)
    else:
        print("ALL CHECKS PASSED — results are consistent", flush=True)
