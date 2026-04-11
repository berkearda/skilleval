"""Calibration analysis for CDMEval's NCDM predictions.

Checks whether predicted P(correct) matches actual correctness rates
via reliability diagrams, ECE, MCE, and Brier score — overall, per
benchmark, per model quartile, per skill, and train vs test.

Usage:
    python tools/run_calibration_analysis.py device=cpu
    python tools/run_calibration_analysis.py device=cuda  # faster for 36M preds
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

N_BINS = 20


def calibration_metrics(preds, labels, n_bins=N_BINS):
    """Compute ECE, MCE, Brier, and per-bin stats."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == 0:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds > lo) & (preds <= hi)
        n = mask.sum()
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "mean_pred": float((lo + hi) / 2), "mean_actual": 0.0,
                         "gap": 0.0})
            continue
        mp = preds[mask].mean()
        ma = labels[mask].mean()
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(n),
                     "mean_pred": float(mp), "mean_actual": float(ma),
                     "gap": float(abs(mp - ma))})

    total = len(preds)
    ece = sum(b["count"] / total * b["gap"] for b in bins if b["count"] > 0)
    mce = max((b["gap"] for b in bins if b["count"] > 0), default=0.0)
    brier = float(((preds - labels) ** 2).mean())
    return {"ece": float(ece), "mce": float(mce), "brier": float(brier),
            "bins": bins, "n": int(total)}


def plot_reliability(bins, ax, label="", color="#4C72B0"):
    """Draw a reliability diagram on the given axis."""
    mp = [b["mean_pred"] for b in bins if b["count"] > 0]
    ma = [b["mean_actual"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.bar(mp, ma, width=1.0 / N_BINS * 0.85, alpha=0.6, color=color,
           edgecolor="white", linewidth=0.5, label=label if label else None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import setup_style
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        skill_labels = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)
    skill_names = [skill_labels[str(i)] for i in range(K)]

    # Validate R is binary
    assert set(np.unique(R)).issubset({0, 1, 0.0, 1.0}), "R is not binary!"

    # ── Load model ──
    print("\nLoading model...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, device,
    )
    net.eval()

    # ── Generate ALL predictions in item batches ──
    print("\nGenerating predictions for all 36M+ pairs...", flush=True)
    all_llm_ids = torch.arange(n_llms, device=device)
    ITEM_BATCH = 200  # items per batch to control memory

    all_preds = np.zeros((n_llms, n_items), dtype=np.float32)
    for start in range(0, n_items, ITEM_BATCH):
        end = min(start + ITEM_BATCH, n_items)
        idx = np.arange(start, end)
        te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            k_d = torch.sigmoid(net.k_difficulty_proj(te))
            e_d = torch.sigmoid(net.e_difficulty_proj(te))
            theta_sig = torch.sigmoid(net.student_emb.weight)  # (M, K)

            # (M, J, K) broadcast
            stat = theta_sig.unsqueeze(1)        # (M, 1, K)
            x = e_d.unsqueeze(0) * (stat - k_d.unsqueeze(0)) * qr.unsqueeze(0)
            MJ = n_llms * len(idx)
            x_flat = x.reshape(MJ, K)
            h1 = torch.sigmoid(net.prednet_full1(x_flat))
            h2 = torch.sigmoid(net.prednet_full2(h1))
            pred = torch.sigmoid(net.prednet_full3(h2)).reshape(n_llms, len(idx))
            all_preds[:, start:end] = pred.cpu().numpy()

        if (start // ITEM_BATCH) % 10 == 0:
            print(f"  items {start}-{end}/{n_items}", flush=True)

    # Validate predictions
    assert all_preds.min() >= 0 and all_preds.max() <= 1, "Predictions out of [0,1]!"
    print(f"\n  Predictions: shape={all_preds.shape}, "
          f"mean={all_preds.mean():.4f}, std={all_preds.std():.4f}, "
          f"min={all_preds.min():.4f}, max={all_preds.max():.4f}", flush=True)
    print(f"  Actual mean: {R.mean():.4f}", flush=True)
    print(f"  Sanity: |pred_mean - actual_mean| = "
          f"{abs(all_preds.mean() - R.mean()):.4f}", flush=True)

    # Flatten for overall metrics
    preds_flat = all_preds.ravel()
    labels_flat = R.ravel().astype(np.float32)

    # ── Train/test split ──
    all_items_idx = np.arange(n_items)
    train_items, test_items = train_test_split(all_items_idx, test_size=0.2,
                                                random_state=42)

    # ════════════════════════════════════════════════════════════════
    # 2. OVERALL CALIBRATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("OVERALL CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    overall = calibration_metrics(preds_flat, labels_flat)
    print(f"  ECE  = {overall['ece']:.4f}", flush=True)
    print(f"  MCE  = {overall['mce']:.4f}", flush=True)
    print(f"  Brier = {overall['brier']:.4f}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. PER-BENCHMARK CALIBRATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PER-BENCHMARK CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    benchmarks = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
    bench_items = {b: np.array([i for i, it in enumerate(items_data)
                                 if it.get("benchmark") == b], dtype=int)
                   for b in benchmarks}
    per_bench = {}
    for b in benchmarks:
        idx = bench_items[b]
        p = all_preds[:, idx].ravel()
        l = R[:, idx].ravel().astype(np.float32)
        m = calibration_metrics(p, l)
        per_bench[b] = m
        print(f"  {b:<8}: ECE={m['ece']:.4f}, MCE={m['mce']:.4f}, "
              f"Brier={m['brier']:.4f}, n={m['n']:,}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. PER-MODEL-QUARTILE CALIBRATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PER-MODEL-QUARTILE CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    llm_acc = R.mean(axis=1)
    quartile_bounds = np.percentile(llm_acc, [25, 50, 75])
    quartile_labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
    quartile_masks = [
        llm_acc <= quartile_bounds[0],
        (llm_acc > quartile_bounds[0]) & (llm_acc <= quartile_bounds[1]),
        (llm_acc > quartile_bounds[1]) & (llm_acc <= quartile_bounds[2]),
        llm_acc > quartile_bounds[2],
    ]
    per_quartile = {}
    for ql, qm in zip(quartile_labels, quartile_masks):
        p = all_preds[qm].ravel()
        l = R[qm].ravel().astype(np.float32)
        m = calibration_metrics(p, l)
        per_quartile[ql] = m
        print(f"  {ql:<16}: ECE={m['ece']:.4f}, n_llms={qm.sum()}, "
              f"mean_acc={llm_acc[qm].mean():.3f}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 5. PER-SKILL CALIBRATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PER-SKILL CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    skill_eces = []
    per_skill = {}
    for k in range(K):
        items_with_skill = np.where(q_matrix[:, k] > 0)[0]
        if len(items_with_skill) < 10:
            skill_eces.append(float("nan"))
            continue
        p = all_preds[:, items_with_skill].ravel()
        l = R[:, items_with_skill].ravel().astype(np.float32)
        m = calibration_metrics(p, l, n_bins=10)  # fewer bins for smaller samples
        skill_eces.append(m["ece"])
        per_skill[str(k)] = {"name": skill_names[k], "ece": m["ece"],
                              "n_items": len(items_with_skill)}

    valid_eces = [e for e in skill_eces if not np.isnan(e)]
    print(f"  Skills with enough items: {len(valid_eces)}/100", flush=True)
    print(f"  Skill ECE: mean={np.mean(valid_eces):.4f}, "
          f"median={np.median(valid_eces):.4f}", flush=True)

    sorted_skills = sorted(per_skill.items(), key=lambda x: x[1]["ece"])
    print(f"\n  Top 10 best-calibrated skills:", flush=True)
    for sk, info in sorted_skills[:10]:
        print(f"    ECE={info['ece']:.4f} ({info['n_items']:>4} items): "
              f"{info['name'][:60]}", flush=True)
    print(f"\n  Top 10 worst-calibrated skills:", flush=True)
    for sk, info in sorted_skills[-10:]:
        print(f"    ECE={info['ece']:.4f} ({info['n_items']:>4} items): "
              f"{info['name'][:60]}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 6. TRAIN VS TEST SPLIT
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("TRAIN VS TEST CALIBRATION", flush=True)
    print(f"{'='*60}", flush=True)
    train_p = all_preds[:, train_items].ravel()
    train_l = R[:, train_items].ravel().astype(np.float32)
    test_p = all_preds[:, test_items].ravel()
    test_l = R[:, test_items].ravel().astype(np.float32)

    train_cal = calibration_metrics(train_p, train_l)
    test_cal = calibration_metrics(test_p, test_l)
    print(f"  Train: ECE={train_cal['ece']:.4f}, Brier={train_cal['brier']:.4f}, "
          f"n={train_cal['n']:,}", flush=True)
    print(f"  Test:  ECE={test_cal['ece']:.4f}, Brier={test_cal['brier']:.4f}, "
          f"n={test_cal['n']:,}", flush=True)
    if test_cal["ece"] > train_cal["ece"] + 0.01:
        print(f"  NOTE: test ECE is notably higher than train — some overfitting",
              flush=True)
    else:
        print(f"  PASS: train/test calibration is comparable", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Fig 1: Overall reliability diagram ──
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_reliability(overall["bins"], ax, color="#4C72B0")
    ax.set_xlabel("Mean predicted P(correct)", fontsize=12)
    ax.set_ylabel("Observed fraction correct", fontsize=12)
    ax.text(0.05, 0.90, f"ECE = {overall['ece']:.4f}\nBrier = {overall['brier']:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f3ff", alpha=0.8))
    plt.tight_layout()
    out1 = fig_dir / "fig_calibration_reliability.pdf"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}", flush=True)

    # ── Fig 2: Per-benchmark reliability ──
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bench_colors = {"MATH": "#4C72B0", "BBH": "#DD8452", "GPQA": "#55A868",
                    "MuSR": "#C44E52", "IFEval": "#8172B3"}
    for ax, b in zip(axes, benchmarks):
        plot_reliability(per_bench[b]["bins"], ax, label=b, color=bench_colors[b])
        ax.set_title(f"{b}\nECE={per_bench[b]['ece']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        if b == benchmarks[0]:
            ax.set_ylabel("Observed", fontsize=9)
    plt.tight_layout()
    out2 = fig_dir / "fig_calibration_benchmark.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Fig 3: Per-quartile reliability ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    q_colors = ["#888888", "#DD8452", "#55A868", "#4C72B0"]
    for ax, (ql, col) in zip(axes, zip(quartile_labels, q_colors)):
        plot_reliability(per_quartile[ql]["bins"], ax, color=col)
        ax.set_title(f"{ql}\nECE={per_quartile[ql]['ece']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        if ql == quartile_labels[0]:
            ax.set_ylabel("Observed", fontsize=9)
    plt.tight_layout()
    out3 = fig_dir / "fig_calibration_quartile.pdf"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}", flush=True)

    # ── Fig 4: Skill ECE histogram ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid_eces, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(valid_eces), color="#C44E52", ls="--", lw=2,
               label=f"Mean ECE = {np.mean(valid_eces):.4f}")
    ax.set_xlabel("ECE per skill", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out4 = fig_dir / "fig_calibration_skill_ece.pdf"
    fig.savefig(out4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out4}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "calibration_analysis",
        "n_llms": int(n_llms), "n_items": int(n_items), "K": int(K),
        "n_bins": N_BINS,
        "prediction_stats": {
            "mean": float(all_preds.mean()), "std": float(all_preds.std()),
            "min": float(all_preds.min()), "max": float(all_preds.max()),
        },
        "actual_mean": float(R.mean()),
        "overall": {"ece": overall["ece"], "mce": overall["mce"],
                     "brier": overall["brier"]},
        "per_benchmark": {b: {"ece": per_bench[b]["ece"], "mce": per_bench[b]["mce"],
                               "brier": per_bench[b]["brier"]}
                          for b in benchmarks},
        "per_quartile": {ql: {"ece": per_quartile[ql]["ece"],
                               "n_llms": int(qm.sum()),
                               "mean_acc": float(llm_acc[qm].mean())}
                         for ql, qm in zip(quartile_labels, quartile_masks)},
        "per_skill_ece": {k: {"name": info["name"], "ece": info["ece"],
                               "n_items": info["n_items"]}
                          for k, info in per_skill.items()},
        "skill_ece_mean": float(np.mean(valid_eces)),
        "skill_ece_median": float(np.median(valid_eces)),
        "train_test": {
            "train_ece": train_cal["ece"], "train_brier": train_cal["brier"],
            "test_ece": test_cal["ece"], "test_brier": test_cal["brier"],
        },
    }

    out_json = Path("cdm_exploration/experiments/v2_calibration_analysis.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="calibration_analysis",
        config={"K": K, "n_bins": N_BINS, "device": device},
        results=save_data,
        split_info={"n_items": n_items, "n_llms": n_llms,
                    "n_train": len(train_items), "n_test": len(test_items)},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
