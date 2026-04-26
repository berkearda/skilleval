"""Compare θ-stability across seeds for IrtNet vs CDMEval.

Request (2026-04-26): "Could you also check the stability of the θ learned by
IRTNet across runs? Not the performance, but the variance of each dimension
of θ across different runs. I suspect the variance would be relatively high,
since the θ is non-identifiable in their model. This is a key comparison to
demonstrate that our θ is more interpretable and meaningful."

What this measures
==================

For each method's multi-seed θ matrices (shape n_llms x d), three Pearson
stability metrics, computed pairwise over seeds and averaged:

  1. Flat Pearson — flatten θ to one (n_llms * d) vector per seed,
     correlate across seed pairs.
  2. Per-LLM Pearson — for each LLM i, correlate θ_s1[i] vs θ_s2[i]
     (length-d vectors), average over LLMs and pairs.
  3. Per-dimension Pearson — for each dimension k, correlate θ_s1[:, k]
     vs θ_s2[:, k] (length-n_llms vectors), average over dimensions and
     pairs. **This is the headline metric.**

For CDMEval, dimensions are anchored to named Q-matrix skills, so per-dim
Pearson should be high (different seeds learn similar mastery on the same
named skill). For IrtNet, the d-dim latent θ has rotation/sign-flip
invariance, so per-dim Pearson is expected to be near zero — the model
relearns a different dimension assignment every seed.

We also report Procrustes-aligned per-dim Pearson: for each pair we solve
for the orthogonal R that best aligns θ_s2 R ≈ θ_s1, then re-measure.
This separates "structure unstable" from "structure stable but rotated".

Inputs
======

  - IrtNet checkpoints at cdm_exploration/checkpoints/irtnet/d{D}_s{S}.pt
        Three seeds per d_model in {232, 512, 1024}.
        Loads `model_embedder.weight` (the θ matrix, shape n_llms x d).
  - CDMEval K=100 stability already aggregated in
    cdm_exploration/experiments/v2_multi_seed.json.
        We reuse the precomputed flat / per-LLM / per-skill Pearson means.

Output
======

  cdm_exploration/experiments/v2_theta_stability_comparison.json

Run on Euler (checkpoints live there):

  $SCRATCH/cdmeval_pro6000_venv/bin/python3 tools/compare_theta_stability.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr


REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
CKPT = REPO / "cdm_exploration" / "checkpoints" / "irtnet"

D_MODEL_GRID = [232, 512, 1024]
SEEDS = [42, 43, 44]


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────

def load_irtnet_theta(d_model: int, seed: int) -> np.ndarray | None:
    path = CKPT / f"d{d_model}_s{seed}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if "model_embedder.weight" not in sd:
        raise KeyError(
            f"Expected 'model_embedder.weight' in {path}; "
            f"got {list(sd.keys())[:5]}",
        )
    return sd["model_embedder.weight"].numpy()  # (n_llms, d_model)


# ──────────────────────────────────────────────────────────────────────
# Stability metrics
# ──────────────────────────────────────────────────────────────────────

def flat_pearson(thetas: list[np.ndarray]) -> tuple[float, float]:
    """Mean and std of pairwise Pearson on flattened θ across seeds."""
    pairs = []
    for a, b in combinations(thetas, 2):
        r, _ = pearsonr(a.flatten(), b.flatten())
        pairs.append(r)
    return float(np.mean(pairs)), float(np.std(pairs))


def per_llm_pearson(thetas: list[np.ndarray]) -> tuple[float, float]:
    """Per-LLM Pearson averaged across LLMs, then averaged across seed pairs."""
    pair_means = []
    n_llms = thetas[0].shape[0]
    for a, b in combinations(thetas, 2):
        per_llm = np.zeros(n_llms)
        for i in range(n_llms):
            r, _ = pearsonr(a[i], b[i])
            per_llm[i] = r if not np.isnan(r) else 0.0
        pair_means.append(per_llm.mean())
    return float(np.mean(pair_means)), float(np.std(pair_means))


def per_dim_pearson(thetas: list[np.ndarray]) -> tuple[float, float, np.ndarray]:
    """Per-dim Pearson averaged across dims, then across seed pairs.

    Returns (mean, std, per_dim_mean_array). The per-dim array averages each
    dimension's pair-wise correlation across all seed pairs, useful for
    plotting the variance distribution.
    """
    pair_means = []
    d = thetas[0].shape[1]
    accum = np.zeros(d)
    n_pairs = 0
    for a, b in combinations(thetas, 2):
        per_dim = np.zeros(d)
        for k in range(d):
            r, _ = pearsonr(a[:, k], b[:, k])
            per_dim[k] = r if not np.isnan(r) else 0.0
        accum += per_dim
        pair_means.append(per_dim.mean())
        n_pairs += 1
    return float(np.mean(pair_means)), float(np.std(pair_means)), accum / n_pairs


def procrustes_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find orthogonal R to minimize ||a - b @ R||_F, return b @ R."""
    # SVD on b^T a
    U, _, Vt = np.linalg.svd(b.T @ a, full_matrices=False)
    R = U @ Vt
    return b @ R


def per_dim_pearson_procrustes(thetas: list[np.ndarray]) -> tuple[float, float]:
    """Per-dim Pearson after Procrustes-aligning each pair."""
    pair_means = []
    d = thetas[0].shape[1]
    for a, b in combinations(thetas, 2):
        b_aligned = procrustes_align(a, b)
        per_dim = np.zeros(d)
        for k in range(d):
            r, _ = pearsonr(a[:, k], b_aligned[:, k])
            per_dim[k] = r if not np.isnan(r) else 0.0
        pair_means.append(per_dim.mean())
    return float(np.mean(pair_means)), float(np.std(pair_means))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def analyse_irtnet() -> dict:
    """Compute IrtNet θ stability per d_model across the 3 seeds."""
    results = {}
    for d in D_MODEL_GRID:
        print(f"\n[ IrtNet d_model={d} ]")
        thetas = []
        seeds_loaded = []
        for s in SEEDS:
            t = load_irtnet_theta(d, s)
            if t is None:
                print(f"  WARN: missing checkpoint for d={d}, seed={s}")
                continue
            thetas.append(t)
            seeds_loaded.append(s)
        if len(thetas) < 2:
            print(f"  Skipping d={d} (only {len(thetas)} seeds)")
            continue
        print(f"  Loaded {len(thetas)} seeds: {seeds_loaded}")
        print(f"  θ shape per seed: {thetas[0].shape}")

        flat_m, flat_s = flat_pearson(thetas)
        print(f"  Flat Pearson:               {flat_m:.4f} ± {flat_s:.4f}")

        llm_m, llm_s = per_llm_pearson(thetas)
        print(f"  Per-LLM Pearson:            {llm_m:.4f} ± {llm_s:.4f}")

        dim_m, dim_s, _ = per_dim_pearson(thetas)
        print(f"  Per-dim Pearson (raw):      {dim_m:.4f} ± {dim_s:.4f}")

        proc_m, proc_s = per_dim_pearson_procrustes(thetas)
        print(f"  Per-dim Pearson (Procrustes): {proc_m:.4f} ± {proc_s:.4f}")

        results[f"d_model={d}"] = {
            "d_model": d,
            "n_seeds": len(thetas),
            "seeds": seeds_loaded,
            "flat_pearson_mean": flat_m,
            "flat_pearson_std": flat_s,
            "per_llm_pearson_mean": llm_m,
            "per_llm_pearson_std": llm_s,
            "per_dim_pearson_raw_mean": dim_m,
            "per_dim_pearson_raw_std": dim_s,
            "per_dim_pearson_procrustes_mean": proc_m,
            "per_dim_pearson_procrustes_std": proc_s,
        }
    return results


def load_cdmeval_reference() -> dict:
    """Pull CDMEval K=100 stability from the existing multi-seed JSON."""
    p = EXP / "v2_multi_seed.json"
    if not p.exists():
        return {"error": f"missing {p}"}
    d = json.load(open(p))
    return {
        "K": d.get("K"),
        "n_seeds": len(d.get("seeds", [])),
        "seeds": d.get("seeds", []),
        "flat_pearson_mean": d.get("theta_pairwise_pearson_flat_mean"),
        "flat_pearson_std": d.get("theta_pairwise_pearson_flat_std"),
        "per_llm_pearson_mean": d.get("theta_per_llm_pearson_mean"),
        "per_llm_pearson_std": d.get("theta_per_llm_pearson_std"),
        "per_dim_pearson_raw_mean": d.get("theta_per_skill_pearson_mean"),
        "per_dim_pearson_raw_std": d.get("theta_per_skill_pearson_std"),
        "note": (
            "CDMEval K=100 dimensions are anchored to named Q-matrix skills, "
            "so per-dim correlation is computed without Procrustes alignment "
            "(rotation invariance does not apply)."
        ),
    }


def print_comparison(cdmeval: dict, irtnet: dict) -> None:
    print("\n" + "=" * 96)
    print("θ stability across seeds — CDMEval K=100 vs IrtNet")
    print("=" * 96)
    print(
        f"{'Method':<28} {'Flat Pearson':>20} {'Per-LLM':>20} "
        f"{'Per-dim (raw)':>20}"
    )
    print("-" * 96)

    if cdmeval.get("flat_pearson_mean") is not None:
        print(
            f"{'CDMEval K=100 (5 seeds)':<28} "
            f"{cdmeval['flat_pearson_mean']:.4f} ± {cdmeval['flat_pearson_std']:.4f}   "
            f"{cdmeval['per_llm_pearson_mean']:.4f} ± {cdmeval['per_llm_pearson_std']:.4f}   "
            f"{cdmeval['per_dim_pearson_raw_mean']:.4f} ± {cdmeval['per_dim_pearson_raw_std']:.4f}"
        )

    for d in D_MODEL_GRID:
        key = f"d_model={d}"
        if key not in irtnet:
            continue
        r = irtnet[key]
        n_seeds = r['n_seeds']
        label = f"IrtNet d={d} ({n_seeds} seeds)"
        print(
            f"{label:<28} "
            f"{r['flat_pearson_mean']:.4f} ± {r['flat_pearson_std']:.4f}   "
            f"{r['per_llm_pearson_mean']:.4f} ± {r['per_llm_pearson_std']:.4f}   "
            f"{r['per_dim_pearson_raw_mean']:.4f} ± {r['per_dim_pearson_raw_std']:.4f}"
        )

    print("\n" + "-" * 96)
    print("IrtNet per-dim Pearson AFTER Procrustes alignment (handles rotation invariance):")
    for d in D_MODEL_GRID:
        key = f"d_model={d}"
        if key not in irtnet:
            continue
        r = irtnet[key]
        print(
            f"  IrtNet d={d}:  raw {r['per_dim_pearson_raw_mean']:.4f}  →  "
            f"aligned {r['per_dim_pearson_procrustes_mean']:.4f} ± "
            f"{r['per_dim_pearson_procrustes_std']:.4f}"
        )
    print("=" * 96)


def main() -> int:
    print("Computing IrtNet θ stability across seeds...")
    irtnet = analyse_irtnet()
    print("\nLoading CDMEval K=100 reference from v2_multi_seed.json...")
    cdmeval = load_cdmeval_reference()

    summary = {
        "cdmeval_K100": cdmeval,
        "irtnet": irtnet,
        "_meta": {
            "d_model_grid": D_MODEL_GRID,
            "seeds": SEEDS,
            "metrics": [
                "flat_pearson",
                "per_llm_pearson",
                "per_dim_pearson_raw",
                "per_dim_pearson_procrustes (IrtNet only)",
            ],
            "interpretation": (
                "Per-dim Pearson is the headline metric for non-identifiability. "
                "CDMEval's named-skill θ should show high per-dim stability; "
                "IrtNet's latent θ should show low raw per-dim stability that "
                "recovers under Procrustes alignment if the underlying ability "
                "ordering is preserved across seeds."
            ),
        },
    }

    out = EXP / "v2_theta_stability_comparison.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {out}")

    print_comparison(cdmeval, irtnet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
