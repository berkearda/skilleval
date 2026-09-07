"""CD-CAT v1 with Fix A: periodic full re-fit of inner theta (T-039).

Mirror of tools/run_adaptive_testing_fixed.py with EXACTLY ONE change:
the SELECTOR's inner theta is re-fit periodically (every K_REFIT=10
items) using the cold-start T-036 recipe (lr=0.05, steps=500), instead
of being maintained by single-step Adam updates.

Background: the diagnostic in tools/diag_cdcat_heuristic.py showed the
v1-fixed inner theta stays at |raw|=0.33 (vs trained ref 1.87, only
17% of scale) because single-step Adam at lr=0.05 cannot keep up with
item arrivals. Result: heuristic computes p(1-p) at theta~=0 and picks
near-identical items for every LLM (203 unique items out of 500 picks
across 5 LLMs). Adaptive degenerates to a one-size-fits-all calibration
set.

Fix A re-fits theta every K_REFIT items so the heuristic computes
uncertainty at a converged theta, enabling per-LLM personalization.

Final-fit recipe (both random and adaptive): unchanged — cold-start
T-036 (lr=0.05, steps=500) fresh from zero on the items collected.

Usage:
    python tools/run_adaptive_testing_fixedA.py device=cuda
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_adaptive_testing import (
    predict_with_theta,
    compute_fisher_information,
)
from tools.run_adaptive_testing_fixed import fit_theta_fixed, FINAL_FIT_LR, FINAL_FIT_STEPS


# T-039 Fix A: re-fit inner theta every K_REFIT items
K_REFIT = 10


def fit_theta_raw_fixed(net, responses, cal_items, q_matrix, text_embs, device, K):
    """Same as fit_theta_fixed but returns theta_raw (pre-sigmoid) parameter
    so the selector can use it directly."""
    cal = cal_items.astype(int)
    y = torch.tensor(responses[cal], dtype=torch.float32, device=device)
    te = torch.tensor(text_embs[cal], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[cal], dtype=torch.float32, device=device)

    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
    optimizer = torch.optim.Adam([theta_raw], lr=FINAL_FIT_LR)
    loss_fn = nn.BCELoss()
    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))
        e_d = torch.sigmoid(net.e_difficulty_proj(te))

    for _ in range(FINAL_FIT_STEPS):
        stat = torch.sigmoid(theta_raw).expand(len(cal), -1)
        x = e_d * (stat - k_d) * qr
        h1 = torch.sigmoid(net.prednet_full1(x))
        h2 = torch.sigmoid(net.prednet_full2(h1))
        pred = torch.sigmoid(net.prednet_full3(h2)).squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return theta_raw


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"[T-039 Fix A] FINAL_FIT_LR={FINAL_FIT_LR}  STEPS={FINAL_FIT_STEPS}  "
          f"K_REFIT={K_REFIT}", flush=True)

    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)

    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device); net.eval()
    for p in net.parameters():
        p.requires_grad = False

    cal_sizes = [int(x) for x in str(getattr(cfg, "cal_sizes",
                  "5,10,20,50,100,200,500")).split(",")]
    n_repeats = int(getattr(cfg, "n_repeats", 3))
    n_eval_llms = min(int(getattr(cfg, "n_eval", 100)), len(test_llms))
    eval_llms = test_llms[:n_eval_llms]
    print(f"\nCD-CAT FIX-A: {n_eval_llms} LLMs, sizes={cal_sizes}, repeats={n_repeats}",
          flush=True)
    print(f"  Selector inner theta: re-fit every {K_REFIT} items "
          f"(lr={FINAL_FIT_LR}, steps={FINAL_FIT_STEPS})", flush=True)
    print(f"  Final fit (both methods): same recipe, fresh from zero", flush=True)

    results_random = {N: [] for N in cal_sizes}
    results_adaptive = {N: [] for N in cal_sizes}

    for li, llm_idx in enumerate(eval_llms):
        if (li + 1) % 10 == 0 or li == 0:
            print(f"  LLM {li+1}/{n_eval_llms}", flush=True)
        responses = R[llm_idx]
        y_test = responses[test_items.astype(int)]
        if len(np.unique(y_test)) < 2:
            continue

        for N in cal_sizes:
            for rep in range(n_repeats):
                rng = np.random.RandomState(42 + rep)

                # Random: pick N at random, then fixed-recipe fit
                cal_random = rng.choice(train_items, size=min(N, len(train_items)),
                                          replace=False)
                mastery_random = fit_theta_fixed(net, responses, cal_random,
                                                  q_matrix, text_embs, device, K)
                preds_random = predict_with_theta(net, mastery_random, test_items,
                                                    q_matrix, text_embs, device)
                results_random[N].append(roc_auc_score(y_test, preds_random))

                # Adaptive Fix A: heuristic pick with periodic re-fit of inner theta
                theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
                remaining_set = set(train_items.tolist())
                remaining_arr = train_items.copy()
                items_chosen = []

                for step in range(min(N, len(train_items))):
                    # Re-fit theta on items collected so far, every K_REFIT steps
                    if step > 0 and step % K_REFIT == 0:
                        theta_raw = fit_theta_raw_fixed(
                            net, responses, np.array(items_chosen),
                            q_matrix, text_embs, device, K)

                    info = compute_fisher_information(
                        net, theta_raw, remaining_arr, q_matrix, text_embs, device)
                    best_local = int(np.argmax(info))
                    best_item = int(remaining_arr[best_local])
                    items_chosen.append(best_item)
                    remaining_set.discard(best_item)
                    remaining_arr = np.array(sorted(remaining_set))

                cal_adaptive = np.array(items_chosen)
                mastery_adaptive = fit_theta_fixed(net, responses, cal_adaptive,
                                                    q_matrix, text_embs, device, K)
                preds_adaptive = predict_with_theta(net, mastery_adaptive, test_items,
                                                      q_matrix, text_embs, device)
                results_adaptive[N].append(roc_auc_score(y_test, preds_adaptive))

    print(f"\n{'='*70}", flush=True)
    print(f"CD-CAT FIX-A RESULTS ({n_eval_llms} LLMs, {n_repeats} seeds)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>6} {'Random':>14} {'Adaptive (Fix A)':>20} {'Gain':>8}", flush=True)
    print("-" * 70, flush=True)

    summary = []
    for N in cal_sizes:
        r_mean = float(np.mean(results_random[N])); r_std = float(np.std(results_random[N]))
        a_mean = float(np.mean(results_adaptive[N])); a_std = float(np.std(results_adaptive[N]))
        gain = a_mean - r_mean
        print(f"{N:>6} {r_mean:>8.4f}±{r_std:.4f} {a_mean:>14.4f}±{a_std:.4f} {gain:>+8.4f}",
              flush=True)
        summary.append({"N": N, "random_mean": r_mean, "random_std": r_std,
                        "adaptive_mean": a_mean, "adaptive_std": a_std, "gain": gain})

    # Reference: print v1 fixed (no fix A) for comparison
    fixed_path = Path("cdm_exploration/experiments/v2_adaptive_testing_fixed.json")
    if fixed_path.exists():
        ref = json.load(open(fixed_path))
        print(f"\n  Reference (v1 fixed, no Fix A — degenerate inner theta):", flush=True)
        for s in ref.get("summary", []):
            print(f"    N={s['N']:>3}: random={s['random_mean']:.4f}, "
                  f"adaptive={s['adaptive_mean']:.4f}, gain={s['gain']:+.4f}", flush=True)

    save_data = {
        "experiment": "adaptive_testing_fixedA",
        "t_id": "T-039",
        "fix": "periodic re-fit of inner theta every K_REFIT items (T-036 recipe)",
        "selector": "p(1-p) * |q|_1, theta computed at converged inner theta",
        "n_eval_llms": n_eval_llms, "n_repeats": n_repeats, "K": K,
        "cal_sizes": cal_sizes,
        "k_refit": K_REFIT,
        "final_fit_lr": FINAL_FIT_LR, "final_fit_steps": FINAL_FIT_STEPS,
        "summary": summary,
    }
    out_json = Path("cdm_exploration/experiments/v2_adaptive_testing_fixedA.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    verified = verify_splits(train_items, test_items, label="adaptive_testing_fixedA")
    log_experiment(
        name="adaptive_testing_fixedA",
        config={"n_eval_llms": n_eval_llms, "K": K, "n_repeats": n_repeats,
                "cal_sizes": cal_sizes, "k_refit": K_REFIT,
                "final_fit_lr": FINAL_FIT_LR, "final_fit_steps": FINAL_FIT_STEPS,
                "device": device},
        results={"summary": summary},
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_train_llms": len(train_llms), "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
