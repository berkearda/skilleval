"""Aggregate multi-seed results from train_expanded.py runs.

Loads 5 checkpoints (seeds 42-46), extracts theta matrices, computes
pairwise correlations and routing variance.

Usage:
    python tools/aggregate_multi_seed.py

Expects checkpoints at:
    cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt  (seed=42)
    cdm_exploration/checkpoints/multi_seed/seed_43.pt
    cdm_exploration/checkpoints/multi_seed/seed_44.pt
    cdm_exploration/checkpoints/multi_seed/seed_45.pt
    cdm_exploration/checkpoints/multi_seed/seed_46.pt

And experiment log entries named train_expanded_seed{42..46}.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

SEEDS = [42, 43, 44, 45, 46]

CKPT_PATHS = {
    42: Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
    43: Path("cdm_exploration/checkpoints/multi_seed/seed_43.pt"),
    44: Path("cdm_exploration/checkpoints/multi_seed/seed_44.pt"),
    45: Path("cdm_exploration/checkpoints/multi_seed/seed_45.pt"),
    46: Path("cdm_exploration/checkpoints/multi_seed/seed_46.pt"),
}


def main():
    print("=" * 60, flush=True)
    print("MULTI-SEED AGGREGATION", flush=True)
    print("=" * 60, flush=True)

    # ── Check all checkpoints exist ──
    missing = [s for s, p in CKPT_PATHS.items() if not p.exists()]
    if missing:
        print(f"\nMISSING checkpoints for seeds: {missing}", flush=True)
        print("Run the remaining seeds first. Available:", flush=True)
        for s, p in CKPT_PATHS.items():
            print(f"  seed={s}: {'OK' if p.exists() else 'MISSING'} ({p})", flush=True)
        sys.exit(1)

    # ── Load thetas ──
    thetas = {}
    for seed in SEEDS:
        ckpt = torch.load(CKPT_PATHS[seed], map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        theta = torch.sigmoid(sd["student_emb.weight"]).numpy()
        thetas[seed] = theta
        print(f"  seed={seed}: theta {theta.shape}, "
              f"range [{theta.min():.4f}, {theta.max():.4f}]", flush=True)

    n_llms, K = thetas[SEEDS[0]].shape
    theta_stack = np.stack([thetas[s] for s in SEEDS], axis=0)  # (5, n_llms, K)

    # ── Collect per-seed metrics from experiment log ──
    log_path = Path("cdm_exploration/experiments/experiment_log.json")
    log = json.load(open(log_path))

    per_seed_results = {}
    for entry in log:
        name = entry.get("experiment", "")
        for seed in SEEDS:
            if name == f"train_expanded_seed{seed}":
                per_seed_results[seed] = entry["results"]

    # Also check the original train_expanded entry for seed=42 if
    # train_expanded_seed42 doesn't exist yet.
    if 42 not in per_seed_results:
        for entry in log:
            if entry.get("experiment") == "train_expanded":
                cfg = entry.get("config", {})
                if cfg.get("n_llms") == n_llms and cfg.get("K") == K:
                    per_seed_results[42] = entry["results"]
                    break

    found_seeds = sorted(per_seed_results.keys())
    print(f"\nPer-seed results found for: {found_seeds}", flush=True)

    # ── Pairwise Pearson r (flattened theta) ──
    print("\nPairwise Pearson r (flattened theta):", flush=True)
    pair_r_flat = {}
    for i, j in combinations(range(len(SEEDS)), 2):
        si, sj = SEEDS[i], SEEDS[j]
        r, _ = pearsonr(thetas[si].ravel(), thetas[sj].ravel())
        pair_r_flat[f"{si}-{sj}"] = float(r)
        print(f"  {si}-{sj}: r = {r:.6f}", flush=True)
    flat_vals = list(pair_r_flat.values())
    flat_mean = float(np.mean(flat_vals))
    flat_std = float(np.std(flat_vals))
    print(f"  Mean: {flat_mean:.6f} +/- {flat_std:.6f}", flush=True)

    # ── Per-LLM Pearson r (averaged over LLMs and pairs) ──
    print("\nPer-LLM Pearson r:", flush=True)
    per_llm_pair_means = []
    for i, j in combinations(range(len(SEEDS)), 2):
        si, sj = SEEDS[i], SEEDS[j]
        rs = np.array([
            pearsonr(thetas[si][l], thetas[sj][l])[0]
            for l in range(n_llms)
        ])
        rs = rs[~np.isnan(rs)]
        per_llm_pair_means.append(float(np.mean(rs)))
    per_llm_mean = float(np.mean(per_llm_pair_means))
    per_llm_std = float(np.std(per_llm_pair_means))
    print(f"  Mean: {per_llm_mean:.6f} +/- {per_llm_std:.6f}", flush=True)

    # ── Per-skill Pearson r ──
    print("\nPer-skill Pearson r:", flush=True)
    per_skill_r = np.zeros(K)
    for k in range(K):
        rs = []
        for i, j in combinations(range(len(SEEDS)), 2):
            si, sj = SEEDS[i], SEEDS[j]
            r, _ = pearsonr(thetas[si][:, k], thetas[sj][:, k])
            if not np.isnan(r):
                rs.append(r)
        per_skill_r[k] = np.mean(rs) if rs else float("nan")
    skill_mean = float(np.nanmean(per_skill_r))
    skill_std = float(np.nanstd(per_skill_r))
    skill_min = float(np.nanmin(per_skill_r))
    skill_max = float(np.nanmax(per_skill_r))
    print(f"  Mean: {skill_mean:.4f} +/- {skill_std:.4f} "
          f"(min={skill_min:.4f}, max={skill_max:.4f})", flush=True)

    # ── Routing Acc@1 and AUC across seeds ──
    acc1s = []
    aucs = []
    for seed in SEEDS:
        if seed in per_seed_results:
            r = per_seed_results[seed]
            a1 = r.get("acc@1", r.get("acc1", None))
            auc = r.get("test_auc", None)
            if a1 is not None:
                acc1s.append(float(a1))
            if auc is not None:
                aucs.append(float(auc))
            print(f"  seed={seed}: AUC={auc}, Acc@1={a1}", flush=True)
        else:
            print(f"  seed={seed}: no log entry found", flush=True)

    acc1s = np.array(acc1s) if acc1s else np.array([])
    aucs = np.array(aucs) if aucs else np.array([])

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    if len(acc1s) > 0:
        print(f"  Acc@1: {np.mean(acc1s):.4f} +/- {np.std(acc1s):.4f} "
              f"(n={len(acc1s)} seeds)", flush=True)
    if len(aucs) > 0:
        print(f"  AUC:   {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} "
              f"(n={len(aucs)} seeds)", flush=True)
    print(f"  Theta flat r: {flat_mean:.4f} +/- {flat_std:.4f}", flush=True)
    print(f"  Theta per-LLM r: {per_llm_mean:.4f} +/- {per_llm_std:.4f}", flush=True)
    print(f"  Theta per-skill r: {skill_mean:.4f} +/- {skill_std:.4f}", flush=True)

    # ── Save ──
    out_dir = Path("cdm_exploration/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "multi_seed_stability",
        "seeds": SEEDS,
        "K": int(K),
        "n_llms": int(n_llms),
        "theta_pairwise_pearson_flat": pair_r_flat,
        "theta_pairwise_pearson_flat_mean": flat_mean,
        "theta_pairwise_pearson_flat_std": flat_std,
        "theta_per_llm_pearson_mean": per_llm_mean,
        "theta_per_llm_pearson_std": per_llm_std,
        "theta_per_skill_pearson_mean": skill_mean,
        "theta_per_skill_pearson_std": skill_std,
        "theta_per_skill_pearson_min": skill_min,
        "theta_per_skill_pearson_max": skill_max,
        "theta_per_skill_pearson": per_skill_r.tolist(),
        "routing_acc1_per_seed": {str(s): float(acc1s[i]) for i, s in enumerate(SEEDS) if i < len(acc1s)},
        "routing_acc1_mean": float(np.mean(acc1s)) if len(acc1s) > 0 else None,
        "routing_acc1_std": float(np.std(acc1s)) if len(acc1s) > 0 else None,
        "test_auc_per_seed": {str(s): float(aucs[i]) for i, s in enumerate(SEEDS) if i < len(aucs)},
        "test_auc_mean": float(np.mean(aucs)) if len(aucs) > 0 else None,
        "test_auc_std": float(np.std(aucs)) if len(aucs) > 0 else None,
        "per_seed_results": {str(s): per_seed_results.get(s) for s in SEEDS},
    }

    out_path = out_dir / "v2_multi_seed.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
