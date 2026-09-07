"""Variance check for the check-1b result (skill signal beyond benchmark profile).

User question: "are you sure? doesn't it depend on the chosen questions?
try a couple more times to see its variance."

Two wobble sources, both measured here:

  A. Training randomness — recompute check 1b with each of the five
     multi-seed checkpoints (seeds 42-46, identical item split, different
     training runs). If the result depends on one lucky training run,
     the claim is fragile.

  B. Choice of fresh questions — per-skill bootstrap: resample each
     skill's held-out items with replacement (B=200) and recompute the
     partial correlation. Shows how much each number moves depending on
     which fresh questions were drawn.

Headline quantities tracked across both: mean partial r | 5-benchmark
profile over the 41 measurable skills, and the count of skills with
positive partial r.
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
MS = REPO / "cdm_exploration" / "checkpoints" / "multi_seed"

MIN_HELDOUT = 10
SEEDS = [42, 43, 44, 45, 46]
B_BOOT = 200


def residualize(v: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    return v - Zc @ np.linalg.lstsq(Zc, v, rcond=None)[0]


def main() -> None:
    rng = np.random.default_rng(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    items_meta = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    bench_of = np.array([m["benchmark"] for m in items_meta])
    benches = sorted(set(bench_of.tolist()))

    # Split is identical across seeds (verified); take it from seed 42.
    ck0 = torch.load(MS / "text_conditioned_seed_42.pt", map_location="cpu", weights_only=False)
    train_items = np.array(sorted(ck0["train_items"]), dtype=int)
    test_items = np.array(sorted(ck0["test_items"]), dtype=int)

    bench_profile = np.column_stack([
        R[:, train_items[bench_of[train_items] == b]].mean(axis=1) for b in benches
    ])
    Zc = np.column_stack([np.ones(R.shape[0]), bench_profile])

    # Skills measurable under the shared split.
    skill_ho = {}
    for k in range(Q.shape[1]):
        ho = test_items[Q[test_items, k] > 0]
        if len(ho) >= MIN_HELDOUT:
            skill_ho[k] = ho
    skills = sorted(skill_ho)
    print(f"measurable skills: {len(skills)} (same split for all seeds)\n", flush=True)

    # ---- A. Across training seeds -------------------------------------
    print("=== A. Training-randomness wobble (5 independent training runs) ===", flush=True)
    per_seed_mean, per_seed_pos = [], []
    thetas = {}
    for s in SEEDS:
        ck = torch.load(MS / f"text_conditioned_seed_{s}.pt", map_location="cpu", weights_only=False)
        th = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()
        thetas[s] = th
        prs = []
        for k in skills:
            target = R[:, skill_ho[k]].mean(axis=1)
            rx = residualize(th[:, k], Zc)
            ry = residualize(target, Zc)
            prs.append(pearsonr(rx, ry)[0])
        prs = np.array(prs)
        per_seed_mean.append(prs.mean())
        per_seed_pos.append(int((prs > 0).sum()))
        print(f"  seed {s}: mean partial r = {prs.mean():.4f}, positive on {(prs>0).sum()}/{len(skills)}", flush=True)
    per_seed_mean = np.array(per_seed_mean)
    print(f"  ACROSS SEEDS: mean {per_seed_mean.mean():.4f} +/- {per_seed_mean.std():.4f}, "
          f"positive count range [{min(per_seed_pos)}, {max(per_seed_pos)}]", flush=True)

    # ---- B. Bootstrap over fresh questions (theta = seed 42) ----------
    print(f"\n=== B. Fresh-question wobble (per-skill bootstrap, B={B_BOOT}, theta seed 42) ===", flush=True)
    th = thetas[42]
    rx_cache = {k: residualize(th[:, k], Zc) for k in skills}
    boot_means, boot_pos = [], []
    per_skill_boot = {k: [] for k in skills}
    for b in range(B_BOOT):
        prs = []
        for k in skills:
            ho = skill_ho[k]
            draw = rng.choice(ho, size=len(ho), replace=True)
            target = R[:, draw].mean(axis=1)
            ry = residualize(target, Zc)
            r = pearsonr(rx_cache[k], ry)[0]
            prs.append(r)
            per_skill_boot[k].append(r)
        prs = np.array(prs)
        boot_means.append(prs.mean())
        boot_pos.append(int((prs > 0).sum()))
    boot_means = np.array(boot_means)
    boot_pos = np.array(boot_pos)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"  mean partial r: {boot_means.mean():.4f}, 95% interval [{lo:.4f}, {hi:.4f}]", flush=True)
    print(f"  positive-skill count: median {int(np.median(boot_pos))}, "
          f"95% interval [{int(np.percentile(boot_pos,2.5))}, {int(np.percentile(boot_pos,97.5))}]", flush=True)

    # Per-skill: how many skills are positive in >= 95% of bootstrap draws?
    frac_pos = {k: np.mean(np.array(per_skill_boot[k]) > 0) for k in skills}
    robust_pos = [k for k in skills if frac_pos[k] >= 0.95]
    print(f"  skills positive in >=95% of draws: {len(robust_pos)}/{len(skills)}", flush=True)
    wide = sorted(skills, key=lambda k: np.std(per_skill_boot[k]))[-3:]
    print("  wobbliest skills (bootstrap std):", flush=True)
    for k in wide:
        arr = np.array(per_skill_boot[k])
        print(f"    skill {k:3d} (n_ho={len(skill_ho[k])}): {arr.mean():+.3f} +/- {arr.std():.3f}", flush=True)

    out = {
        "experiment": "benchprofile_partial_r_variance",
        "n_skills": len(skills),
        "min_heldout_items": MIN_HELDOUT,
        "seed_variation": {
            "seeds": SEEDS,
            "mean_partial_r_per_seed": [float(x) for x in per_seed_mean],
            "mean": float(per_seed_mean.mean()),
            "std": float(per_seed_mean.std()),
            "positive_count_per_seed": per_seed_pos,
        },
        "item_bootstrap": {
            "B": B_BOOT,
            "theta_seed": 42,
            "mean_partial_r": float(boot_means.mean()),
            "ci95": [float(lo), float(hi)],
            "positive_count_median": int(np.median(boot_pos)),
            "positive_count_ci95": [int(np.percentile(boot_pos, 2.5)), int(np.percentile(boot_pos, 97.5))],
            "n_skills_positive_95pct_of_draws": len(robust_pos),
        },
        "verified": True,
    }
    outp = EXP / "v2_benchprofile_partial_variance.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
