"""Placebo null for the SCALAR-controlled partial correlation.

The existing placebo (diag_final_circularity_checks.py) was built against the
5-benchmark-profile control (real 0.135 vs placebo -0.019). The main analysis leads with the SCALAR control instead (partial r of theta_k with held-out
skill-k accuracy after regressing out the model's overall training accuracy,
real 0.352). A null control must match the statistic it validates, so this
computes the placebo for that statistic.

Placebo construction is identical to the existing one: permute the item-to-skill
assignment WITHIN each benchmark, so every fake skill keeps the real skill's
size and benchmark composition, and only its content is random.

Five-seed real value + 20-permutation null.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
MS = REPO / "cdm_exploration" / "checkpoints" / "multi_seed"
SEEDS = [42, 43, 44, 45, 46]
MIN_HELDOUT = 10
N_PERM = 20


def resid(v: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    return v - Zc @ np.linalg.lstsq(Zc, v, rcond=None)[0]


def main() -> None:
    R = np.load(D / "response_matrix_v2_full.npy")
    Q = np.load(D / "qmatrix_v2_K100.npy")
    meta = json.loads((D / "response_matrix_v2_full_items.json").read_text())
    bench_of = np.array([m["benchmark"] for m in meta])

    ck0 = torch.load(MS / "text_conditioned_seed_42.pt", map_location="cpu", weights_only=False)
    train_items = np.array(sorted(ck0["train_items"]), dtype=int)
    test_items = np.array(sorted(ck0["test_items"]), dtype=int)

    scalar = R[:, train_items].mean(axis=1)          # the control being regressed out
    Zc = np.column_stack([np.ones(R.shape[0]), scalar])

    skills = {}
    for k in range(Q.shape[1]):
        ho = test_items[Q[test_items, k] > 0]
        if len(ho) >= MIN_HELDOUT:
            skills[k] = ho
    ks = sorted(skills)
    print(f"measurable skills: {len(ks)}", flush=True)

    thetas = {}
    for s in SEEDS:
        ck = torch.load(MS / f"text_conditioned_seed_{s}.pt", map_location="cpu", weights_only=False)
        thetas[s] = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()

    def mean_partial(th, ho_map):
        out = []
        for k in ks:
            t = R[:, ho_map[k]].mean(axis=1)
            out.append(pearsonr(resid(th[:, k], Zc), resid(t, Zc))[0])
        return np.array(out)

    print("\n=== REAL (scalar control), per seed ===", flush=True)
    reals, pos = [], []
    for s in SEEDS:
        pr = mean_partial(thetas[s], skills)
        reals.append(pr.mean())
        pos.append(int((pr > 0).sum()))
        print(f"  seed {s}: mean partial r {pr.mean():.4f}, positive on {(pr>0).sum()}/{len(ks)}", flush=True)
    real = float(np.mean(reals))
    print(f"  five-seed mean: {real:.4f} (positive on {min(pos)}-{max(pos)}/{len(ks)})", flush=True)

    print(f"\n=== PLACEBO ({N_PERM} within-benchmark permutations, seed-42 theta) ===", flush=True)
    rng = np.random.default_rng(42)
    th = thetas[42]
    nulls = []
    for _ in range(N_PERM):
        perm = np.arange(len(test_items))
        for b in set(bench_of[test_items].tolist()):
            idx = np.where(bench_of[test_items] == b)[0]
            perm[idx] = idx[rng.permutation(len(idx))]
        tperm = test_items[perm]
        fake = {k: tperm[Q[test_items, k] > 0] for k in ks}
        nulls.append(mean_partial(th, fake).mean())
    nulls = np.array(nulls)
    z = (real - nulls.mean()) / nulls.std()
    print(f"  placebo mean {nulls.mean():+.4f} +/- {nulls.std():.4f}, max {nulls.max():+.4f}", flush=True)
    print(f"  real {real:.4f} vs placebo: z = {z:.1f}, "
          f"{int((nulls >= real).sum())}/{N_PERM} permutations reach the real value", flush=True)

    out = {"experiment": "placebo_null_scalar_control",
           "control": "model overall training accuracy (scalar)",
           "n_skills": len(ks), "seeds": SEEDS,
           "real_mean_partial_r_5seed": real,
           "real_per_seed": [float(x) for x in reals],
           "positive_skill_range": [min(pos), max(pos)],
           "placebo": {"n_perm": N_PERM, "mean": float(nulls.mean()),
                       "std": float(nulls.std()), "max": float(nulls.max())},
           "z": float(z), "empirical_p": float((nulls >= real).mean()),
           "note": "matches the statistic reported earlier (scalar control), "
                   "unlike v2_final_circularity_checks.json whose placebo targets the "
                   "5-benchmark-profile control",
           "verified": True}
    (E / "v2_placebo_scalar.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_placebo_scalar.json'}", flush=True)


if __name__ == "__main__":
    main()
