"""Controls for the weak-beats-strong statistic and for Acc@k (refs issue #1).

The reported statistic counts held-out items where the strongest LLM (by training accuracy) is
wrong and *any* of the ~2,360 LLMs with <=13B parameters is right. With that many models, some
small model is usually right by chance on a multiple-choice item, so the count approaches the
strongest model's error count. This script reports:

1. the original count next to the strongest model's error count (the denominator it approaches);
2. a fixed-model version: the single best <=13B LLM, chosen on training items, against the
   strongest LLM on held-out items, in both directions;
3. a majority version: items where the strongest LLM is wrong and more than half of the
   <=13B LLMs are right;
4. a chance baseline for the "any small model" rule: the expected count if each small model
   answered each item independently at its own held-out accuracy;
5. Acc@k baselines for k in (1, 3, 5, 10): k random LLMs (averaged over draws) and the k
   strongest LLMs by training accuracy, scored with the same at-least-one-correct rule.

It uses the same data, size parser and item split (test_size=0.2, random_state=42) as
fig_weak_beats_strong_v2.py, and writes cdm_exploration/experiments/weak_beats_strong_controls.json.

    python tools/diag_weak_beats_strong_controls.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

SMALL_MAX_B = 13.0
KS = (1, 3, 5, 10)
RANDOM_DRAWS = 200
PHI_SIZES = [("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
             ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3)]


def parse_size(name: str) -> float:
    """Same parser as fig_weak_beats_strong_v2.parse_size."""
    low = name.lower()
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return float(m.group(1))
    for kw, sz in PHI_SIZES:
        if kw in low:
            return sz
    return np.nan


def weak_beats_strong_controls(R: np.ndarray, sizes: np.ndarray, train_items: np.ndarray,
                               test_items: np.ndarray) -> dict:
    """R is the LLM x item 0/1 correctness matrix."""
    train_acc = R[:, train_items].mean(axis=1)
    strongest = int(np.argmax(train_acc))
    small = np.where((sizes <= SMALL_MAX_B) & ~np.isnan(sizes))[0]
    best_small = int(small[np.argmax(train_acc[small])])

    T = R[:, test_items]
    strong_wrong = T[strongest] == 0
    small_correct_frac = T[small].mean(axis=0)
    any_small = T[small].max(axis=0) > 0

    # Expected count under independence at each small model's held-out accuracy.
    p_small = T[small].mean(axis=1)
    p_none = float(np.exp(np.log1p(-np.clip(p_small, 0, 1 - 1e-12)).sum()))
    n_test = len(test_items)

    return {
        "n_test_items": n_test,
        "n_small_models": int(len(small)),
        "strongest_model_index": strongest,
        "strongest_test_errors": int(strong_wrong.sum()),
        "original_any_small": {
            "items": int((strong_wrong & any_small).sum()),
            "share_of_strongest_errors": float((strong_wrong & any_small).sum() / max(strong_wrong.sum(), 1)),
            "chance_items_if_independent": float(strong_wrong.sum() * (1 - p_none)),
        },
        "fixed_best_small_model": {
            "model_index": best_small,
            "small_right_strong_wrong": int(((T[best_small] == 1) & strong_wrong).sum()),
            "strong_right_small_wrong": int(((T[best_small] == 0) & ~strong_wrong).sum()),
            "small_test_acc": float(T[best_small].mean()),
            "strong_test_acc": float(T[strongest].mean()),
        },
        "majority_of_small_models": {
            "items": int((strong_wrong & (small_correct_frac > 0.5)).sum()),
        },
    }


def acc_at_k_baselines(R: np.ndarray, train_items: np.ndarray, test_items: np.ndarray,
                       seed: int = 0) -> dict:
    train_acc = R[:, train_items].mean(axis=1)
    order = np.argsort(-train_acc)
    T = R[:, test_items]
    rng = np.random.default_rng(seed)
    out = {}
    for k in KS:
        strongest_k = float(T[order[:k]].max(axis=0).mean())
        draws = [T[rng.choice(R.shape[0], size=k, replace=False)].max(axis=0).mean()
                 for _ in range(RANDOM_DRAWS)]
        out[f"acc@{k}"] = {"strongest_k": strongest_k, "random_k_mean": float(np.mean(draws)),
                           "random_k_sd": float(np.std(draws, ddof=1))}
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.cdm_ready)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    sizes = np.array([parse_size(n) for n in llm_names])
    train_items, test_items = train_test_split(np.arange(R.shape[1]), test_size=0.2, random_state=42)

    result = weak_beats_strong_controls(R, sizes, train_items, test_items)
    result["strongest_model"] = llm_names[result["strongest_model_index"]]
    result["fixed_best_small_model"]["model"] = llm_names[result["fixed_best_small_model"]["model_index"]]
    result["acc_at_k_baselines"] = acc_at_k_baselines(R, train_items, test_items)

    out = Path("cdm_exploration/experiments/weak_beats_strong_controls.json")
    out.write_text(json.dumps(result, indent=2))
    o, f, m = result["original_any_small"], result["fixed_best_small_model"], result["majority_of_small_models"]
    print(f"strongest errors on test: {result['strongest_test_errors']} / {result['n_test_items']}")
    print(f"any-small rule: {o['items']} ({o['share_of_strongest_errors']:.1%} of strongest errors; "
          f"chance under independence {o['chance_items_if_independent']:.0f})")
    print(f"best small model ({f['model']}): beats strongest on {f['small_right_strong_wrong']}, "
          f"loses on {f['strong_right_small_wrong']}")
    print(f"majority of small models right, strongest wrong: {m['items']}")
    for k, v in result["acc_at_k_baselines"].items():
        print(f"{k}: strongest-k {v['strongest_k']:.4f}, random-k {v['random_k_mean']:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
