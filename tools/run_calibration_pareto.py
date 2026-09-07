"""Apply post-hoc isotonic calibration and re-run cost-aware Pareto.

Reads val + test prediction matrices from the calibrated retraining of
CDMEval (T-033) and EmbedLLM (T-034). For each method:
  1. Fit IsotonicRegression on (val_preds.flatten(), val_true.flatten()).
  2. Apply mapping to test_preds.flatten() -> calibrated test predictions.
  3. Compute ECE / AUC (uncalibrated and calibrated) for sanity.
  4. Run cheapest-above-threshold sweep on calibrated test predictions.
  5. Merge curves into v2_pareto_multibaseline.json under new method keys.

Then call fig_pareto_v4.py to render.

T-033 / T-034. the project log 2026-04-30.

Usage:
    python tools/run_calibration_pareto.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
LLMS = json.load(open(DATA / "response_matrix_v2_full_llms.json"))


def ece(preds: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    pf = preds.flatten()
    tf = true.flatten()
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pf >= lo) & (pf < hi) if i < n_bins - 1 else (pf >= lo) & (pf <= hi)
        if mask.sum() == 0:
            continue
        out += (mask.sum() / len(pf)) * abs(pf[mask].mean() - tf[mask].mean())
    return float(out)


def threshold_sweep(preds: np.ndarray, R_test: np.ndarray, flops: np.ndarray,
                     thresholds: list[float]) -> list[dict]:
    n_llms, n_test = preds.shape
    out = []
    for t in thresholds:
        correct = 0
        total_cost = 0.0
        picks = []
        n_fallback = 0
        for j in range(n_test):
            p = preds[:, j]
            above = np.where(p > t)[0]
            if len(above) > 0:
                chosen = above[np.argmin(flops[above])]
            else:
                chosen = int(np.argmax(p))
                n_fallback += 1
            picks.append(chosen)
            total_cost += flops[chosen]
            if R_test[chosen, j] > 0:
                correct += 1
        out.append({
            "threshold": float(t),
            "accuracy": correct / n_test,
            "mean_cost": total_cost / n_test,
            "n_unique_picks": int(len(np.unique(picks))),
            "fallback_rate": n_fallback / n_test,
        })
    return out


def calibrate_and_sweep(npz_path: Path, method_name: str, flops: np.ndarray,
                          strongest_acc: float, strongest_cost: float,
                          thresholds: list[float]) -> dict:
    """Fit isotonic on val, apply to test, run threshold_sweep on both."""
    print(f"\n=== {method_name} ===", flush=True)
    print(f"  loading {npz_path}", flush=True)
    d = np.load(npz_path)
    val_preds = d["val_preds"]      # (n_llms, n_val)
    R_val = d["R_val"]              # (n_llms, n_val)
    test_preds = d["test_preds"]    # (n_llms, n_test)
    R_test = d["R_test"]            # (n_llms, n_test)
    print(f"  val_preds: {val_preds.shape}  test_preds: {test_preds.shape}",
            flush=True)

    # ── Uncalibrated metrics ──
    val_auc_uncal = float(roc_auc_score(R_val.flatten(), val_preds.flatten()))
    test_auc_uncal = float(roc_auc_score(R_test.flatten(), test_preds.flatten()))
    val_ece_uncal = ece(val_preds, R_val)
    test_ece_uncal = ece(test_preds, R_test)
    print(f"  uncalibrated: val AUC={val_auc_uncal:.4f}, ECE={val_ece_uncal:.4f} "
            f"| test AUC={test_auc_uncal:.4f}, ECE={test_ece_uncal:.4f}",
            flush=True)

    # ── Fit isotonic on val ──
    print(f"  fitting IsotonicRegression on val ({val_preds.size:,} pairs)...",
            flush=True)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_preds.flatten(), R_val.flatten())

    # ── Apply to test ──
    test_preds_cal = iso.predict(test_preds.flatten()).reshape(test_preds.shape)
    val_preds_cal = iso.predict(val_preds.flatten()).reshape(val_preds.shape)
    test_auc_cal = float(roc_auc_score(R_test.flatten(), test_preds_cal.flatten()))
    val_ece_cal = ece(val_preds_cal, R_val)
    test_ece_cal = ece(test_preds_cal, R_test)
    print(f"  calibrated  : val ECE={val_ece_cal:.4f} (was {val_ece_uncal:.4f}) "
            f"| test AUC={test_auc_cal:.4f}, ECE={test_ece_cal:.4f} "
            f"(was {test_ece_uncal:.4f})", flush=True)

    # ── Threshold sweep on calibrated test ──
    print(f"  running threshold sweep on calibrated test predictions...",
            flush=True)
    sweep = threshold_sweep(test_preds_cal, R_test, flops, thresholds)
    normalized = [{**p,
                    "cost_pct": p["mean_cost"] / strongest_cost * 100,
                    "acc_pct": p["accuracy"] / strongest_acc * 100}
                   for p in sweep]
    return {
        "method_name": method_name,
        "val_auc_uncalibrated": val_auc_uncal,
        "val_ece_uncalibrated": val_ece_uncal,
        "test_auc_uncalibrated": test_auc_uncal,
        "test_ece_uncalibrated": test_ece_uncal,
        "val_ece_calibrated": val_ece_cal,
        "test_auc_calibrated": test_auc_cal,
        "test_ece_calibrated": test_ece_cal,
        "sweep": sweep,
        "normalized": normalized,
    }


def main() -> int:
    from cdmeval.evaluation.cost_analysis import compute_flops_cost

    # ── Load existing pareto JSON ──
    pareto_path = EXP / "v2_pareto_multibaseline.json"
    if not pareto_path.exists():
        raise FileNotFoundError(f"{pareto_path} missing — run "
                                  "tools/run_pareto_multibaseline.py first")
    pareto = json.load(open(pareto_path))
    strongest_acc = pareto["strongest"]["test_acc"]
    strongest_cost = pareto["strongest"]["cost_gflops"]
    thresholds = pareto["thresholds"]
    print(f"loaded pareto: strongest_acc={strongest_acc:.4f}, "
            f"cost={strongest_cost:.0f}, thresholds={len(thresholds)}",
            flush=True)

    # ── Compute FLOPs (must match pareto multibaseline) ──
    flops_dict = compute_flops_cost(LLMS, seq_length=512)
    flops = np.array([flops_dict[n] for n in LLMS])

    # ── Calibrate CDMEval ──
    cdm_npz = EXP / "v2_cdmeval_calibrated_predictions_s42.npz"
    if cdm_npz.exists():
        cdm_result = calibrate_and_sweep(cdm_npz,
                                            "CDMEval (K=100, 60% train, iso-cal)",
                                            flops, strongest_acc, strongest_cost,
                                            thresholds)
        pareto["methods"]["cdmeval_calibrated"] = {
            "sweep": cdm_result["sweep"],
            "normalized": cdm_result["normalized"],
            "val_ece_uncalibrated": cdm_result["val_ece_uncalibrated"],
            "val_ece_calibrated": cdm_result["val_ece_calibrated"],
            "test_auc_uncalibrated": cdm_result["test_auc_uncalibrated"],
            "test_auc_calibrated": cdm_result["test_auc_calibrated"],
            "test_ece_uncalibrated": cdm_result["test_ece_uncalibrated"],
            "test_ece_calibrated": cdm_result["test_ece_calibrated"],
        }
    else:
        print(f"\n  SKIP: {cdm_npz} not found yet (CDMEval calibrated training "
                "not done)", flush=True)

    # ── Calibrate EmbedLLM ──
    el_npz = EXP / "v2_embedllm_calibrated_d232_s42_predictions.npz"
    if el_npz.exists():
        el_result = calibrate_and_sweep(el_npz,
                                          "EmbedLLM (d=232, 60% train, iso-cal)",
                                          flops, strongest_acc, strongest_cost,
                                          thresholds)
        pareto["methods"]["embedllm_calibrated"] = {
            "sweep": el_result["sweep"],
            "normalized": el_result["normalized"],
            "val_ece_uncalibrated": el_result["val_ece_uncalibrated"],
            "val_ece_calibrated": el_result["val_ece_calibrated"],
            "test_auc_uncalibrated": el_result["test_auc_uncalibrated"],
            "test_auc_calibrated": el_result["test_auc_calibrated"],
            "test_ece_uncalibrated": el_result["test_ece_uncalibrated"],
            "test_ece_calibrated": el_result["test_ece_calibrated"],
        }
    else:
        print(f"\n  SKIP: {el_npz} not found yet (EmbedLLM calibrated training "
                "not done)", flush=True)

    # ── Save merged pareto JSON ──
    out_path = EXP / "v2_pareto_multibaseline_v4.json"
    with open(out_path, "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
