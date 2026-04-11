"""Contamination detection via person-fit statistics (lz).

Flags LLMs whose response patterns are inconsistent with their CDM
skill profiles. Models that memorised benchmark items will show
aberrant lz because they get hard items right and easy items wrong
relative to their estimated abilities.

Usage:
    python tools/run_contamination_detection.py device=cpu
"""

import json
import sys
from collections import defaultdict
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

LZ_THRESHOLD = -2.0
EPS = 1e-7


def compute_lz(R_subset, P_subset):
    """Compute lz person-fit statistic for each LLM.

    Args:
        R_subset: (M, J) binary response matrix.
        P_subset: (M, J) predicted probabilities (clipped away from 0/1).

    Returns:
        lz: (M,) standardised person-fit statistic.
    """
    P = np.clip(P_subset, EPS, 1 - EPS)
    log_p = np.log(P)
    log_1mp = np.log(1 - P)
    log_odds = log_p - log_1mp  # log(p/(1-p))

    # Observed log-likelihood per LLM
    l_obs = (R_subset * log_p + (1 - R_subset) * log_1mp).sum(axis=1)  # (M,)

    # Expected log-likelihood under the model
    E_l = (P * log_p + (1 - P) * log_1mp).sum(axis=1)  # (M,)

    # Variance of log-likelihood
    Var_l = (P * (1 - P) * log_odds ** 2).sum(axis=1)  # (M,)

    std_l = np.sqrt(np.maximum(Var_l, EPS))
    lz = (l_obs - E_l) / std_l
    return lz


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
    R = np.load(data_dir / "response_matrix_v2_full.npy").astype(np.float32)
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

    # ── Load model & generate predictions ──
    print("\nLoading model and generating predictions...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, device,
    )
    net.eval()

    ITEM_BATCH = 200
    all_preds = np.zeros((n_llms, n_items), dtype=np.float32)
    for start in range(0, n_items, ITEM_BATCH):
        end = min(start + ITEM_BATCH, n_items)
        idx = np.arange(start, end)
        te = torch.tensor(text_embs[idx], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            k_d = torch.sigmoid(net.k_difficulty_proj(te))
            e_d = torch.sigmoid(net.e_difficulty_proj(te))
            theta_sig = torch.sigmoid(net.student_emb.weight)
            stat = theta_sig.unsqueeze(1)
            x = e_d.unsqueeze(0) * (stat - k_d.unsqueeze(0)) * qr.unsqueeze(0)
            x_flat = x.reshape(n_llms * len(idx), K)
            h1 = torch.sigmoid(net.prednet_full1(x_flat))
            h2 = torch.sigmoid(net.prednet_full2(h1))
            pred = torch.sigmoid(net.prednet_full3(h2)).reshape(n_llms, len(idx))
            all_preds[:, start:end] = pred.cpu().numpy()
        if (start // ITEM_BATCH) % 10 == 0:
            print(f"  items {start}-{end}/{n_items}", flush=True)

    print(f"  Predictions: mean={all_preds.mean():.4f}, range=[{all_preds.min():.4f}, "
          f"{all_preds.max():.4f}]", flush=True)

    # Load theta for consistency analysis
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).cpu().numpy()

    # Train/test split
    all_items_idx = np.arange(n_items)
    train_items, test_items = train_test_split(all_items_idx, test_size=0.2,
                                                random_state=42)

    # ════════════════════════════════════════════════════════════════
    # 1. OVERALL lz
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("1. OVERALL PERSON-FIT (lz)", flush=True)
    print(f"{'='*60}", flush=True)

    lz_all = compute_lz(R, all_preds)
    lz_train = compute_lz(R[:, train_items], all_preds[:, train_items])
    lz_test = compute_lz(R[:, test_items], all_preds[:, test_items])

    n_flagged_all = (lz_all < LZ_THRESHOLD).sum()
    n_flagged_test = (lz_test < LZ_THRESHOLD).sum()

    print(f"  All items: mean={lz_all.mean():.3f}, std={lz_all.std():.3f}, "
          f"flagged (lz<{LZ_THRESHOLD}): {n_flagged_all}/{n_llms} "
          f"({100*n_flagged_all/n_llms:.1f}%)", flush=True)
    print(f"  Train items: mean={lz_train.mean():.3f}, std={lz_train.std():.3f}, "
          f"flagged: {(lz_train<LZ_THRESHOLD).sum()}", flush=True)
    print(f"  Test items: mean={lz_test.mean():.3f}, std={lz_test.std():.3f}, "
          f"flagged: {n_flagged_test}", flush=True)

    # Sanity: mean should be near 0, std near 1
    if abs(lz_all.mean()) > 1.0:
        print(f"  WARNING: mean lz far from 0 — predictions may be miscalibrated",
              flush=True)

    # ════════════════════════════════════════════════════════════════
    # 2. PER-BENCHMARK lz
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("2. PER-BENCHMARK lz", flush=True)
    print(f"{'='*60}", flush=True)

    benchmarks = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
    bench_items = {b: np.array([i for i, it in enumerate(items_data)
                                 if it.get("benchmark") == b], dtype=int)
                   for b in benchmarks}

    bench_lz = {}
    for b in benchmarks:
        idx = bench_items[b]
        lz_b = compute_lz(R[:, idx], all_preds[:, idx])
        bench_lz[b] = lz_b
        n_flag = (lz_b < LZ_THRESHOLD).sum()
        print(f"  {b:<8}: mean={lz_b.mean():.3f}, std={lz_b.std():.3f}, "
              f"flagged={n_flag} ({100*n_flag/n_llms:.1f}%)", flush=True)

    # Benchmark-specific outliers: flagged on 1 benchmark but not others
    print(f"\n  Benchmark-specific aberrance:", flush=True)
    bench_specific = defaultdict(list)
    for i in range(n_llms):
        flagged_benches = [b for b in benchmarks if bench_lz[b][i] < LZ_THRESHOLD]
        normal_benches = [b for b in benchmarks if bench_lz[b][i] >= LZ_THRESHOLD]
        if len(flagged_benches) == 1 and len(normal_benches) >= 3:
            bench_specific[flagged_benches[0]].append(i)

    for b in benchmarks:
        n = len(bench_specific[b])
        if n > 0:
            print(f"    Aberrant ONLY on {b}: {n} models", flush=True)
            for idx in bench_specific[b][:3]:
                print(f"      {llm_names[idx]} (lz={bench_lz[b][idx]:.2f})", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. WITHIN-SKILL CONSISTENCY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("3. WITHIN-SKILL CONSISTENCY", flush=True)
    print(f"{'='*60}", flush=True)

    mastery = (theta > 0.5).astype(int)
    inconsistency_scores = np.zeros(n_llms)
    n_skills_checked = np.zeros(n_llms, dtype=int)

    for k in range(K):
        items_with_skill = np.where(q_matrix[:, k] > 0)[0]
        if len(items_with_skill) < 10:
            continue
        # LLMs mastering this skill
        masters = np.where(mastery[:, k] == 1)[0]
        if len(masters) < 20:
            continue

        # Sort items by predicted difficulty (mean P across masters)
        mean_p = all_preds[masters][:, items_with_skill].mean(axis=0)
        sorted_items = items_with_skill[np.argsort(-mean_p)]  # easy first
        half = len(sorted_items) // 2
        easy_items = sorted_items[:half]
        hard_items = sorted_items[half:]

        for m in masters:
            acc_easy = R[m, easy_items].mean()
            acc_hard = R[m, hard_items].mean()
            incon = acc_hard - acc_easy  # should be negative
            inconsistency_scores[m] += incon
            n_skills_checked[m] += 1

    # Normalise
    valid = n_skills_checked > 0
    inconsistency_scores[valid] /= n_skills_checked[valid]

    print(f"  LLMs with enough skill checks: {valid.sum()}", flush=True)
    print(f"  Mean inconsistency: {inconsistency_scores[valid].mean():.4f} "
          f"(should be negative)", flush=True)
    n_positive = (inconsistency_scores[valid] > 0).sum()
    print(f"  LLMs with positive inconsistency (suspicious): {n_positive} "
          f"({100*n_positive/valid.sum():.1f}%)", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("4. TOP ABERRANT MODELS", flush=True)
    print(f"{'='*60}", flush=True)

    sorted_by_lz = np.argsort(lz_all)
    print(f"  Top 20 most aberrant (lowest lz on all items):", flush=True)
    top_aberrant = []
    for rank, idx in enumerate(sorted_by_lz[:20]):
        acc = R[idx].mean()
        bench_flags = [b for b in benchmarks if bench_lz[b][idx] < LZ_THRESHOLD]
        top_aberrant.append({
            "rank": rank + 1, "name": llm_names[idx], "lz_all": float(lz_all[idx]),
            "accuracy": float(acc), "flagged_benchmarks": bench_flags,
        })
        print(f"    {rank+1:>2}. lz={lz_all[idx]:>7.2f} acc={acc:.3f} "
              f"bench_flags={bench_flags} {llm_names[idx][:50]}", flush=True)

    # Model size correlation
    import re
    sizes = []
    for name in llm_names:
        m = re.search(r"(\d+\.?\d*)[bB]", name)
        sizes.append(float(m.group(1)) if m else None)

    valid_size = np.array([s is not None for s in sizes])
    if valid_size.sum() > 50:
        from scipy.stats import pearsonr
        log_s = np.array([np.log10(s) if s else 0 for s in sizes])
        r, p = pearsonr(log_s[valid_size], lz_all[valid_size])
        print(f"\n  Correlation(log_size, lz): r={r:.3f}, p={p:.2e}", flush=True)
    else:
        r = 0.0

    # Family distribution of flagged models
    print(f"\n  Flagged models by family:", flush=True)
    flagged_mask = lz_all < LZ_THRESHOLD
    family_flagged = defaultdict(int)
    family_total = defaultdict(int)
    for i, name in enumerate(llm_names):
        nl = name.lower()
        fam = "other"
        for f in ["llama", "qwen", "mistral", "gemma", "phi", "falcon", "yi"]:
            if f in nl:
                fam = f
                break
        family_total[fam] += 1
        if flagged_mask[i]:
            family_flagged[fam] += 1
    for fam in sorted(family_total.keys()):
        ft = family_total[fam]
        ff = family_flagged[fam]
        if ff > 0:
            print(f"    {fam:<10}: {ff}/{ft} ({100*ff/ft:.1f}%)", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Fig 1: lz distribution histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lz_all, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3,
            alpha=0.8)
    ax.axvline(LZ_THRESHOLD, color="#C44E52", ls="--", lw=2,
               label=f"Threshold = {LZ_THRESHOLD}")
    ax.text(LZ_THRESHOLD - 0.3, ax.get_ylim()[1] * 0.9,
            f"{n_flagged_all} flagged\n({100*n_flagged_all/n_llms:.1f}%)",
            ha="right", fontsize=10, color="#C44E52")
    ax.set_xlabel("Person-fit statistic (lz)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out1 = fig_dir / "fig_contamination_lz_dist.pdf"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}", flush=True)

    # ── Fig 2: Per-benchmark lz scatter (pick 2 most interesting pairs) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter_pairs = [("MATH", "BBH"), ("MATH", "IFEval"), ("GPQA", "IFEval")]
    for ax, (b1, b2) in zip(axes, scatter_pairs):
        c = np.where(flagged_mask, "#C44E52", "#4C72B0")
        ax.scatter(bench_lz[b1], bench_lz[b2], s=4, alpha=0.3, c=c)
        ax.axhline(LZ_THRESHOLD, color="#ccc", ls="--", lw=0.8)
        ax.axvline(LZ_THRESHOLD, color="#ccc", ls="--", lw=0.8)
        ax.set_xlabel(f"lz ({b1})", fontsize=10)
        ax.set_ylabel(f"lz ({b2})", fontsize=10)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out2 = fig_dir / "fig_contamination_benchmark.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Fig 3: Top 20 aberrant bar chart ──
    fig, ax = plt.subplots(figsize=(8, 7))
    names_short = [a["name"].split("__")[-1][:35] for a in top_aberrant]
    lz_vals = [a["lz_all"] for a in top_aberrant]
    y = np.arange(len(top_aberrant))
    ax.barh(y, lz_vals, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_short, fontsize=8)
    ax.axvline(LZ_THRESHOLD, color="black", ls="--", lw=0.8)
    ax.set_xlabel("lz statistic", fontsize=11)
    ax.invert_yaxis()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out3 = fig_dir / "fig_contamination_top_aberrant.pdf"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}", flush=True)

    # ── Save JSON ──
    save_data = {
        "experiment": "contamination_detection",
        "n_llms": int(n_llms), "n_items": int(n_items), "K": int(K),
        "lz_threshold": LZ_THRESHOLD,
        "overall": {
            "mean_lz": float(lz_all.mean()), "std_lz": float(lz_all.std()),
            "n_flagged": int(n_flagged_all),
            "pct_flagged": float(100 * n_flagged_all / n_llms),
        },
        "train_items": {
            "mean_lz": float(lz_train.mean()), "std_lz": float(lz_train.std()),
            "n_flagged": int((lz_train < LZ_THRESHOLD).sum()),
        },
        "test_items": {
            "mean_lz": float(lz_test.mean()), "std_lz": float(lz_test.std()),
            "n_flagged": int(n_flagged_test),
        },
        "per_benchmark": {
            b: {"mean_lz": float(bench_lz[b].mean()),
                "std_lz": float(bench_lz[b].std()),
                "n_flagged": int((bench_lz[b] < LZ_THRESHOLD).sum()),
                "n_benchmark_specific": len(bench_specific[b])}
            for b in benchmarks
        },
        "inconsistency": {
            "mean": float(inconsistency_scores[valid].mean()),
            "n_positive": int(n_positive),
            "pct_positive": float(100 * n_positive / max(valid.sum(), 1)),
        },
        "size_lz_correlation": float(r),
        "top_aberrant": top_aberrant,
        "family_flagged": dict(family_flagged),
    }

    out_json = Path("cdm_exploration/experiments/v2_contamination_detection.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="contamination_detection",
        config={"K": K, "lz_threshold": LZ_THRESHOLD, "device": device},
        results={"n_flagged": int(n_flagged_all),
                 "pct_flagged": float(100 * n_flagged_all / n_llms),
                 "mean_lz": float(lz_all.mean())},
        split_info={"n_llms": n_llms, "n_items": n_items},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
