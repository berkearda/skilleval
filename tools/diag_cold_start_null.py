"""F9 null-baseline diagnostic for cold-start.

Question: is the pop-mean cold-start AUC (0.678) coming from CDMEval's
model machinery, or just from the trivial fact that "easy items are easy
for everyone, hard items are hard for everyone"?

Three baselines compared on the SAME 763 held-out LLMs and 1905 test items
that the cold-start used:

  N1. CDMEval pop-mean θ init (the result we're verifying)
  N2. Trivial null: per-item mean correctness across the 3,048 train LLMs.
      No model. For each test item i, predict P_i = mean(R[train_llms, i]).
      Same prediction across every held-out LLM.
  N3. Random: shuffled per-item predictions.

If N2 ≈ N1, the cold-start "win" is item-difficulty structure, not skill
decomposition. If N1 >> N2, CDMEval's skill machinery is contributing.

Also reports per-LLM AUC distribution (mean, std, min, max, percentiles).
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    data_dir = REPO / "cdm_exploration" / "data" / "cdm_ready"
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    n_llms, n_items = R.shape

    # SAME splits as cold_start_v2_popinit
    train_items, test_items = train_test_split(np.arange(n_items),
                                                  test_size=0.2,
                                                  random_state=42)
    train_llms, test_llms = train_test_split(np.arange(n_llms),
                                                test_size=0.2,
                                                random_state=42)
    print(f"R: {R.shape}")
    print(f"  test_items: {len(test_items)}, test_llms: {len(test_llms)}")
    print(f"  train_llms: {len(train_llms)}")

    R_test = R[:, test_items.astype(int)]  # (3811, 1905)

    # ── N2: trivial null = per-item mean correctness across train LLMs ──
    print("\n[N2] computing trivial null (per-item mean across 3048 train LLMs)")
    # For each test item, the predicted P is the mean correctness across
    # train LLMs on that item. Same prediction for every held-out LLM.
    item_difficulty = R[train_llms.astype(int)][:, test_items.astype(int)].mean(axis=0)  # (1905,)
    print(f"  item_difficulty range: [{item_difficulty.min():.4f}, "
            f"{item_difficulty.max():.4f}], mean={item_difficulty.mean():.4f}")
    print(f"  fraction items where >50% of train LLMs got it right: "
            f"{(item_difficulty > 0.5).mean():.4f}")

    # Compute per-LLM AUC using item_difficulty as the prediction
    null_aucs = []
    for llm_idx in test_llms:
        y_true = R_test[llm_idx]
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, item_difficulty)
        null_aucs.append(auc)
    null_aucs = np.array(null_aucs)
    print(f"\n[N2 result] trivial null AUC across {len(null_aucs)} held-out LLMs:")
    print(f"  mean={null_aucs.mean():.4f}  std={null_aucs.std():.4f}")
    print(f"  min={null_aucs.min():.4f}  p10={np.percentile(null_aucs, 10):.4f}  "
            f"p50={np.percentile(null_aucs, 50):.4f}  "
            f"p90={np.percentile(null_aucs, 90):.4f}  max={null_aucs.max():.4f}")

    # ── N1 reference (loaded from JSON) ──
    print("\n[N1 reference] CDMEval pop-init at N=0 (from v2_cold_start_popinit.json):")
    popinit = json.load(open(
        REPO / "cdm_exploration" / "experiments" / "v2_cold_start_popinit.json"))
    n0 = next(s for s in popinit["summary"] if s["N"] == 0)
    print(f"  AUC mean = {n0['auc_mean']:.4f}  std = {n0['auc_std']:.4f}")

    # Original zero-init baseline
    print("\n[N0 reference] CDMEval zero-init at N=0 (original, broken):")
    zeroinit = json.load(open(
        REPO / "cdm_exploration" / "experiments" / "v2_cold_start.json"))
    n0_zero = next(s for s in zeroinit["summary"] if s["N"] == 0)
    print(f"  AUC mean = {n0_zero['auc_mean']:.4f}  std = {n0_zero['auc_std']:.4f}")

    # ── Comparison ──
    print(f"\n{'='*70}")
    print(f"COLD-START NULL-CONTROL DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"  zero-init (original)           : AUC {n0_zero['auc_mean']:.4f} ± {n0_zero['auc_std']:.4f}")
    print(f"  trivial null (per-item mean)   : AUC {null_aucs.mean():.4f} ± {null_aucs.std():.4f}")
    print(f"  pop-init (T-035, 'CDMEval')    : AUC {n0['auc_mean']:.4f} ± {n0['auc_std']:.4f}")
    print(f"  full training (per-LLM theta)  : AUC {popinit['full_training_auc']:.4f}")
    print()
    print(f"  Δ (pop-init − trivial null)    : {n0['auc_mean'] - null_aucs.mean():+.4f}")
    print(f"  Δ (full training − pop-init)   : {popinit['full_training_auc'] - n0['auc_mean']:+.4f}")
    print(f"  Δ (full training − trivial)    : {popinit['full_training_auc'] - null_aucs.mean():+.4f}")
    print()
    print("INTERPRETATION:")
    print("  If pop-init ≈ trivial null → CDMEval's machinery contributes nothing.")
    print("  If pop-init > trivial null → CDMEval skill decomposition adds value.")
    print(f"  Observed delta: {(n0['auc_mean'] - null_aucs.mean()):+.4f}")
    if abs(n0['auc_mean'] - null_aucs.mean()) < 0.005:
        print("  → essentially tied; cold-start gain is item-difficulty structure")
    elif n0['auc_mean'] > null_aucs.mean() + 0.005:
        print(f"  → CDMEval adds {(n0['auc_mean'] - null_aucs.mean())*100:+.1f} AUC points "
                "above the trivial null")
    else:
        print(f"  → CDMEval LOSES {(null_aucs.mean() - n0['auc_mean'])*100:+.1f} AUC points "
                "below the trivial null (broken)")

    # Save
    out = {
        "experiment": "cold_start_null_diagnostic",
        "n_train_llms": int(len(train_llms)),
        "n_test_llms": int(len(test_llms)),
        "n_test_items": int(len(test_items)),
        "trivial_null_auc": {
            "mean": float(null_aucs.mean()),
            "std": float(null_aucs.std()),
            "p10": float(np.percentile(null_aucs, 10)),
            "p50": float(np.percentile(null_aucs, 50)),
            "p90": float(np.percentile(null_aucs, 90)),
            "min": float(null_aucs.min()),
            "max": float(null_aucs.max()),
        },
        "cdmeval_pop_init_n0_auc": n0["auc_mean"],
        "cdmeval_zero_init_n0_auc": n0_zero["auc_mean"],
        "cdmeval_full_training_auc": popinit["full_training_auc"],
    }
    out_path = REPO / "cdm_exploration" / "experiments" / "v2_cold_start_null_diag.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
