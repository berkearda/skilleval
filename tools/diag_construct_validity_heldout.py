"""Held-out construct validity for SkillEval theta.

The published construct-validity number (run_construct_validity.py -> 0.837)
computes per-skill accuracy over the FULL response matrix. The paper (L377)
calls it "held-out accuracy", which is imprecise: theta was fit with the
protocolB item-wise split holding out test items, but the accuracy target
averaged over all items (train + test).

This script recomputes the same diagonal / off-diagonal construct-validity
control using ONLY the held-out test items (the items theta never saw during
training). If theta_k still tracks its own skill's held-out accuracy far
better than other skills' held-out accuracy, the validity is out-of-sample
and the internal-consistency charge is answered.

Diagonal    r = corr(theta[:,k], heldout_acc[:,k])   across 3,811 LLMs
Off-diagonal r = corr(theta[:,i], heldout_acc[:,j])  i != j   (control)
Gap = diagonal - off-diagonal.  A single global ability factor would give
gap ~ 0.
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

MIN_HELDOUT_ITEMS = 10   # skills with fewer held-out items are too noisy; reported separately


def main() -> None:
    np.random.seed(42)
    print("=== Held-out construct validity: theta vs HELD-OUT per-skill accuracy ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    n_llms, n_items = R.shape
    K = Q.shape[1]

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    theta = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()  # (n_llms, K)

    # Held-out items = the checkpoint's own recorded test_items (what theta never saw).
    test_items = np.array(sorted(ck["test_items"]), dtype=int)
    train_items = np.array(sorted(ck["train_items"]), dtype=int)
    print(f"checkpoint: train_items={len(train_items)}, test_items={len(test_items)}", flush=True)

    # Cross-check against the canonical calibration split.
    split = json.load(open(DATA / "calibration_split.json"))
    cal_test = set(split["test_idx"])
    overlap = len(set(test_items.tolist()) & cal_test)
    print(f"cross-check: checkpoint test_items vs calibration_split.test_idx "
          f"-> {overlap}/{len(test_items)} match ({len(cal_test)} in cal split)", flush=True)
    assert set(test_items.tolist()).isdisjoint(set(train_items.tolist())), "train/test overlap!"

    # Per-skill accuracy computed over HELD-OUT items only.
    heldout_acc = np.full((n_llms, K), np.nan, dtype=np.float64)
    n_ho_items = np.zeros(K, dtype=int)
    for k in range(K):
        skill_ho = test_items[Q[test_items, k] > 0]
        n_ho_items[k] = len(skill_ho)
        if len(skill_ho) > 0:
            heldout_acc[:, k] = R[:, skill_ho].mean(axis=1)
    print(f"held-out items per skill: min={n_ho_items.min()}, "
          f"median={int(np.median(n_ho_items))}, max={n_ho_items.max()}", flush=True)
    print(f"skills with >= {MIN_HELDOUT_ITEMS} held-out items: "
          f"{(n_ho_items >= MIN_HELDOUT_ITEMS).sum()}/{K}", flush=True)

    # Diagonal: theta_k vs its own skill's held-out accuracy.
    diag = np.full(K, np.nan)
    for k in range(K):
        if n_ho_items[k] == 0:
            continue
        y = heldout_acc[:, k]
        m = ~np.isnan(y)
        if m.sum() < 50:
            continue
        diag[k], _ = pearsonr(theta[m, k], y[m])

    def summarize(mask, label):
        rs = diag[mask & ~np.isnan(diag)]
        print(f"\n[{label}] n_skills={len(rs)}", flush=True)
        print(f"  diagonal mean r   = {rs.mean():.4f}", flush=True)
        print(f"  diagonal median r = {np.median(rs):.4f}", flush=True)
        print(f"  frac r>0.5 = {(rs>0.5).mean():.3f}   frac r>0.7 = {(rs>0.7).mean():.3f}", flush=True)
        return rs

    rs_all = summarize(n_ho_items > 0, "all skills with >=1 held-out item")
    rs_rob = summarize(n_ho_items >= MIN_HELDOUT_ITEMS, f"robust subset (>= {MIN_HELDOUT_ITEMS} held-out items)")

    # Off-diagonal control: theta_i vs OTHER skill j's held-out accuracy.
    valid_k = np.where((n_ho_items >= MIN_HELDOUT_ITEMS))[0]
    pairs = [(i, j) for i in valid_k for j in valid_k if i != j]
    idx = np.random.choice(len(pairs), min(1000, len(pairs)), replace=False)
    off = []
    for p in idx:
        i, j = pairs[p]
        y = heldout_acc[:, j]
        m = ~np.isnan(y)
        if m.sum() < 50:
            continue
        r, _ = pearsonr(theta[m, i], y[m])
        off.append(r)
    off = np.array(off)
    print(f"\n=== CONTROL (robust subset) ===", flush=True)
    print(f"  off-diagonal mean r = {off.mean():.4f}", flush=True)
    print(f"  diagonal mean r     = {rs_rob.mean():.4f}", flush=True)
    print(f"  construct-validity GAP = {rs_rob.mean() - off.mean():+.4f}", flush=True)
    print(f"\n(full-matrix reference: diag 0.837, off 0.253, gap 0.584)", flush=True)

    out = {
        "experiment": "construct_validity_heldout_items",
        "ckpt": str(CKPT.relative_to(REPO)),
        "held_out_source": "checkpoint test_items (item-wise OOD split, seed 42)",
        "n_held_out_items": int(len(test_items)),
        "min_heldout_items_per_skill_filter": MIN_HELDOUT_ITEMS,
        "all_skills": {
            "n": int(len(rs_all)),
            "diagonal_mean_r": float(rs_all.mean()),
            "diagonal_median_r": float(np.median(rs_all)),
        },
        "robust_subset": {
            "n": int(len(rs_rob)),
            "diagonal_mean_r": float(rs_rob.mean()),
            "diagonal_median_r": float(np.median(rs_rob)),
            "off_diagonal_mean_r": float(off.mean()),
            "gap": float(rs_rob.mean() - off.mean()),
        },
        "full_matrix_reference": {"diagonal": 0.8372, "off_diagonal": 0.2533, "gap": 0.5838},
        "verified": True,
    }
    outp = EXP / "v2_construct_validity_heldout.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
