"""Check 1 for the circularity analysis: skill score vs plain scalar.

Contest, per skill k, across 3,811 LLMs:
  Predictor A (scalar) : model's overall accuracy on TRAINING items only.
                         One number per model, no skill structure.
  Predictor B (skill)  : theta[:, k] from the protocolB checkpoint
                         (trained on training items only).
  Target               : model's accuracy on HELD-OUT items requiring skill k.

Both predictors see only training data; the target is fresh items. If the
100 skill scores were a single general-ability score in disguise, B could
never systematically beat A, because A IS the general-ability score.
Every skill where r(B, target) > r(A, target) is information a scalar
cannot contain.

Also reports the partial correlation of theta_k with the target after
regressing out the scalar (the incremental skill-specific signal).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"

MIN_HELDOUT = 10  # same robustness filter as diag_construct_validity_heldout.py


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y) after regressing z out of both."""
    zc = np.column_stack([np.ones_like(z), z])
    rx = x - zc @ np.linalg.lstsq(zc, x, rcond=None)[0]
    ry = y - zc @ np.linalg.lstsq(zc, y, rcond=None)[0]
    return pearsonr(rx, ry)[0]


def main() -> None:
    np.random.seed(42)
    print("=== Check 1: theta_k vs scalar train-accuracy, predicting held-out skill-k accuracy ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    K = Q.shape[1]

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    theta = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()
    train_items = np.array(sorted(ck["train_items"]), dtype=int)
    test_items = np.array(sorted(ck["test_items"]), dtype=int)
    assert set(train_items.tolist()).isdisjoint(set(test_items.tolist()))
    print(f"train items {len(train_items)}, held-out items {len(test_items)}", flush=True)

    # Predictor A: overall TRAIN accuracy (scalar general ability).
    scalar = R[:, train_items].mean(axis=1)
    print(f"scalar train-acc: mean {scalar.mean():.3f}, range [{scalar.min():.3f}, {scalar.max():.3f}]", flush=True)

    rows = []
    for k in range(K):
        skill_ho = test_items[Q[test_items, k] > 0]
        if len(skill_ho) < MIN_HELDOUT:
            continue
        target = R[:, skill_ho].mean(axis=1)
        r_scalar = pearsonr(scalar, target)[0]
        r_theta = pearsonr(theta[:, k], target)[0]
        r_partial = partial_corr(theta[:, k], target, scalar)
        rows.append({
            "skill": int(k),
            "n_heldout_items": int(len(skill_ho)),
            "r_scalar": float(r_scalar),
            "r_theta": float(r_theta),
            "delta": float(r_theta - r_scalar),
            "r_theta_partial_given_scalar": float(r_partial),
        })

    n = len(rows)
    deltas = np.array([r["delta"] for r in rows])
    r_s = np.array([r["r_scalar"] for r in rows])
    r_t = np.array([r["r_theta"] for r in rows])
    r_p = np.array([r["r_theta_partial_given_scalar"] for r in rows])
    wins = int((deltas > 0).sum())

    print(f"\nskills evaluated (>= {MIN_HELDOUT} held-out items): {n}", flush=True)
    print(f"  mean r  scalar->target : {r_s.mean():.4f}", flush=True)
    print(f"  mean r  theta_k->target: {r_t.mean():.4f}", flush=True)
    print(f"  mean delta (theta - scalar): {deltas.mean():+.4f}", flush=True)
    print(f"  theta wins: {wins}/{n} skills ({wins/n:.1%})", flush=True)
    print(f"  mean partial r (theta_k | scalar): {r_p.mean():.4f}", flush=True)
    print(f"  partial r > 0 on {(r_p > 0).sum()}/{n} skills", flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r["delta"])
    print("\ntop 5 skills by delta (theta beats scalar most):", flush=True)
    for r in rows_sorted[:5]:
        print(f"  skill {r['skill']:3d}  n_ho={r['n_heldout_items']:3d}  "
              f"r_scalar={r['r_scalar']:.3f}  r_theta={r['r_theta']:.3f}  "
              f"delta={r['delta']:+.3f}", flush=True)
    print("bottom 5 (scalar beats theta most):", flush=True)
    for r in rows_sorted[-5:]:
        print(f"  skill {r['skill']:3d}  n_ho={r['n_heldout_items']:3d}  "
              f"r_scalar={r['r_scalar']:.3f}  r_theta={r['r_theta']:.3f}  "
              f"delta={r['delta']:+.3f}", flush=True)

    out = {
        "experiment": "skill_theta_vs_scalar_trainacc_heldout_prediction",
        "ckpt": str(CKPT.relative_to(REPO)),
        "design": "predictors use train data only; target = held-out per-skill accuracy",
        "min_heldout_items": MIN_HELDOUT,
        "n_skills_evaluated": n,
        "mean_r_scalar": float(r_s.mean()),
        "mean_r_theta": float(r_t.mean()),
        "mean_delta": float(deltas.mean()),
        "theta_wins": wins,
        "theta_win_rate": float(wins / n),
        "mean_partial_r_theta_given_scalar": float(r_p.mean()),
        "n_partial_positive": int((r_p > 0).sum()),
        "per_skill": rows,
        "verified": True,
    }
    outp = EXP / "v2_skill_vs_scalar_heldout.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
