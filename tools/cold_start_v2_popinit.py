"""Cold-start v2 with population-mean theta initialization (T-035).

Mirror of tools/cold_start_v2.py with one fix: replace the zero
initialization of held-out theta with the empirical mean theta computed
across the 3,048 training LLMs. Sigmoid(zero) = 0.5 mastery on every
skill is the maximum-uncertainty prior; population-mean is a much
stronger prior because the LLM population has a structured shape (some
skills are universally easy, some universally hard).

Audit-first plan in the project log 2026-05-01 (T-035).

Usage:
    python tools/cold_start_v2_popinit.py device=cpu
    python tools/cold_start_v2_popinit.py device=cuda
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Reuse the batched_predict and most of the calibration logic from the
# original cold_start_v2 — we only replace the initialization.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cold_start_v2 import batch_predict


def batch_calibrate_popinit(net, R, llm_indices, cal_items_per_llm, q_matrix,
                              text_embs, device, K, pop_logit, lr=0.01, steps=50):
    """Like batch_calibrate, but initialize theta_raw from population mean
    in raw (logit) space instead of zero.

    pop_logit: (K,) torch tensor on device, the logit of the population
    mean mastery per skill.
    """
    net.eval()
    n_cal = len(llm_indices)

    # Population-mean initialization in raw space
    init = pop_logit.unsqueeze(0).expand(n_cal, -1).clone()
    theta_raw = nn.Parameter(init)
    optimizer = torch.optim.Adam([theta_raw], lr=lr)
    loss_fn = nn.BCELoss()

    all_llm_local = []
    all_items_flat = []
    all_scores = []
    for local_idx, (llm_idx, cal_items) in enumerate(zip(llm_indices, cal_items_per_llm)):
        cal = cal_items.astype(int)
        n = len(cal)
        all_llm_local.extend([local_idx] * n)
        all_items_flat.extend(cal.tolist())
        all_scores.extend(R[llm_idx, cal].tolist())

    llm_local_t = torch.tensor(all_llm_local, dtype=torch.int64, device=device)
    scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device)

    unique_items = np.unique(all_items_flat)
    item_to_local = {int(it): i for i, it in enumerate(unique_items)}
    te_unique = torch.tensor(text_embs[unique_items], dtype=torch.float32, device=device)
    qr_unique = torch.tensor(q_matrix[unique_items], dtype=torch.float32, device=device)
    with torch.no_grad():
        k_diff_unique = torch.sigmoid(net.k_difficulty_proj(te_unique))
        e_diff_unique = torch.sigmoid(net.e_difficulty_proj(te_unique))
    items_local = torch.tensor([item_to_local[int(it)] for it in all_items_flat],
                                dtype=torch.int64, device=device)

    for step in range(steps):
        stat = torch.sigmoid(theta_raw[llm_local_t])
        k_d = k_diff_unique[items_local]
        e_d = e_diff_unique[items_local]
        qr_d = qr_unique[items_local]

        x = e_d * (stat - k_d) * qr_d
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)

        loss = loss_fn(pred, scores_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(theta_raw).detach().cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={K}", flush=True)

    # Splits — IDENTICAL to cold_start_v2.py
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    print(f"  Items: {len(train_items)} train, {len(test_items)} test", flush=True)
    print(f"  LLMs: {len(train_llms)} train, {len(test_llms)} held-out", flush=True)

    # F5 sniff: confirm train/test LLM disjoint
    assert len(set(train_llms.tolist()) & set(test_llms.tolist())) == 0, \
        "F5 LEAK: train_llms and test_llms overlap"
    print(f"  [F5 OK] train_llms ∩ test_llms == ∅", flush=True)

    # Model
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device)
    net.eval()

    # ─── NEW: compute population-mean theta from train LLMs ───
    print("\n[T-035] computing population-mean theta over train LLMs ...",
            flush=True)
    with torch.no_grad():
        train_theta_raw = net.student_emb(
            torch.tensor(train_llms, device=device))  # (n_train, K)
        train_theta = torch.sigmoid(train_theta_raw)
    pop_mean = train_theta.mean(dim=0)  # (K,)
    pop_logit = torch.logit(pop_mean.clamp(0.01, 0.99))  # back to raw space
    pop_mean_np = pop_mean.cpu().numpy()
    # Sanity prints (B4)
    print(f"  pop_mean: min={pop_mean_np.min():.4f}, "
            f"p10={np.percentile(pop_mean_np, 10):.4f}, "
            f"median={np.median(pop_mean_np):.4f}, "
            f"p90={np.percentile(pop_mean_np, 90):.4f}, "
            f"max={pop_mean_np.max():.4f}", flush=True)
    deviation = np.abs(pop_mean_np - 0.5).max()
    print(f"  max deviation from 0.5: {deviation:.4f}  "
            f"(must be > 0.01 for fix to be meaningful)", flush=True)
    assert deviation > 0.01, "pop_mean is essentially 0.5 — fix has no effect"

    # Full-training baseline (same as original)
    print("\nComputing full-training baseline (50 train LLMs)...", flush=True)
    sample_train = train_llms[:50]
    with torch.no_grad():
        raw = net.student_emb(torch.tensor(sample_train, device=device)).cpu().numpy()
    full_mastery = 1.0 / (1.0 + np.exp(-raw))
    full_preds = batch_predict(net, full_mastery, test_items, q_matrix, text_embs, device)

    full_aucs = []
    for i, llm_idx in enumerate(sample_train):
        y_true = R[llm_idx, test_items.astype(int)]
        if len(np.unique(y_true)) >= 2:
            full_aucs.append(roc_auc_score(y_true, full_preds[i]))
    full_auc = float(np.mean(full_aucs))
    print(f"  Full-training AUC: {full_auc:.4f}", flush=True)

    # Cold-start sweep with pop-init
    cal_sizes = [0, 1, 5, 10, 50, 100, 500]
    n_repeats = 3
    rng = np.random.RandomState(42)
    print(f"\nCold-start (pop-init): {len(test_llms)} LLMs, sizes={cal_sizes}, "
            f"repeats={n_repeats}", flush=True)

    summary = []
    for N in cal_sizes:
        print(f"\n  N={N}...", end=" ", flush=True)
        reps = 1 if N == 0 else n_repeats
        all_aucs = []
        all_accs = []
        for rep in range(reps):
            if N == 0:
                # NEW: predict from population-mean θ for every test LLM
                theta_mat = np.tile(pop_mean_np, (len(test_llms), 1))
            else:
                cal_per_llm = [rng.choice(train_items, size=min(N, len(train_items)),
                                            replace=False) for _ in test_llms]
                theta_mat = batch_calibrate_popinit(
                    net, R, test_llms, cal_per_llm, q_matrix, text_embs,
                    device, K, pop_logit, lr=0.01, steps=50,
                )
            preds = batch_predict(net, theta_mat, test_items, q_matrix, text_embs, device)
            for i, llm_idx in enumerate(test_llms):
                y_true = R[llm_idx, test_items.astype(int)]
                if len(np.unique(y_true)) < 2:
                    continue
                auc = roc_auc_score(y_true, preds[i])
                acc = accuracy_score(y_true, (preds[i] >= 0.5).astype(int))
                all_aucs.append(auc)
                all_accs.append(acc)

        auc_m, auc_s = np.mean(all_aucs), np.std(all_aucs)
        acc_m = np.mean(all_accs)
        pct = auc_m / full_auc * 100
        print(f"AUC={auc_m:.4f}±{auc_s:.4f}, Acc={acc_m:.4f}, {pct:.1f}% of full",
                flush=True)
        summary.append({"N": N, "auc_mean": float(auc_m), "auc_std": float(auc_s),
                          "acc_mean": float(acc_m), "pct_of_full": float(pct)})

    # Print before/after comparison vs the original v2_cold_start.json
    original_path = Path("cdm_exploration/experiments/v2_cold_start.json")
    if original_path.exists():
        original = json.load(open(original_path))
        print(f"\n{'='*80}", flush=True)
        print(f"COLD-START COMPARISON: zero-init (original) vs pop-init (T-035)",
                flush=True)
        print(f"{'='*80}", flush=True)
        print(f"{'N':>6} {'zero-init AUC':>18} {'pop-init AUC':>18} {'Δ AUC':>10}",
                flush=True)
        print("-" * 80, flush=True)
        orig_by_N = {s["N"]: s for s in original["summary"]}
        for s in summary:
            o = orig_by_N.get(s["N"], {})
            o_auc = o.get("auc_mean", float("nan"))
            delta = s["auc_mean"] - o_auc
            print(f"{s['N']:>6} {o_auc:>18.4f} {s['auc_mean']:>18.4f} "
                    f"{delta:>+10.4f}", flush=True)

    save_data = {
        "dataset": "v2_full",
        "experiment": "cold_start_v2_popinit",
        "n_llms": n_llms, "n_test_llms": len(test_llms),
        "n_items": n_items, "n_test_items": len(test_items),
        "K": K,
        "full_training_auc": full_auc,
        "calibration_sizes": cal_sizes, "n_repeats": n_repeats,
        "init_method": "population_mean_theta_from_train_llms",
        "pop_mean_stats": {
            "min": float(pop_mean_np.min()),
            "p10": float(np.percentile(pop_mean_np, 10)),
            "median": float(np.median(pop_mean_np)),
            "p90": float(np.percentile(pop_mean_np, 90)),
            "max": float(pop_mean_np.max()),
        },
        "summary": summary,
        "verified": True,
    }
    out_json = Path("cdm_exploration/experiments/v2_cold_start_popinit.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    verified = verify_splits(train_items, test_items, label="cold_start_v2_popinit")
    log_experiment(
        name="cold_start_v2_popinit",
        config={"n_test_llms": len(test_llms), "K": K,
                "cal_sizes": cal_sizes, "n_repeats": n_repeats,
                "init": "population_mean_theta", "device": device},
        results={"full_auc": full_auc, "summary": summary},
        split_info={"n_train_items": len(train_items),
                      "n_test_items": len(test_items),
                      "n_train_llms": len(train_llms),
                      "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
