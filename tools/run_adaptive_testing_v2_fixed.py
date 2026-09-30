"""CD-CAT v2 with symmetric final-fit (T-038).

Mirror of tools/run_adaptive_testing_v2.py with EXACTLY the bug-fix:
both random AND adaptive (heuristic / trace / D-optimal) end with a
fresh fit using the cold-start T-036 recipe (lr=0.05, steps=500) on
the items collected. The selector still uses warm-started 15-step
MAP internally to guide gradient computation; the FINAL theta used
for AUC comes from a fresh fit.

Also: scaled to 100 LLMs (was 20) and 3 seeds (was 1) to match v1.

Trace and D-optimal use EXACT autograd gradients on the top-200
heuristic-prefiltered subset (line 84 of v2: torch.autograd.grad).
This is real Fisher math, not the broken Q-row proxy from v3.

Usage:
    python tools/run_adaptive_testing_v2_fixed.py device=cuda
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tools.run_adaptive_testing_v2 import (
    predict_with_theta,
    map_update,
    prefilter_and_select,
    TOP_K_PREFILTER,
)

# T-038: symmetric final-fit recipe (T-036 cold-start fix)
FINAL_FIT_LR = 0.05
FINAL_FIT_STEPS = 500


def fit_theta_fixed(net, responses, cal_items, q_matrix, text_embs, device, K):
    """Fresh-from-zero fit with cold-start T-036 hyperparameters."""
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

    return torch.sigmoid(theta_raw).detach().cpu().numpy().squeeze()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"[T-038] FINAL_FIT_LR={FINAL_FIT_LR}  STEPS={FINAL_FIT_STEPS}", flush=True)

    print("\nLoading...", flush=True)
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

    # +ckpt=<file> selects the frozen network; +out_tag=<suffix> keeps the result file apart (T-123).
    ckpt_path = Path(str(cfg.ckpt)) if hasattr(cfg, "ckpt") else Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    out_tag = str(cfg.out_tag) if hasattr(cfg, "out_tag") else ""
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device); net.eval()
    for p in net.parameters():
        p.requires_grad = False

    cal_sizes = [int(x) for x in str(getattr(cfg, "cal_sizes",
                  "10,50,100,200,500")).split(",")]
    n_eval = int(getattr(cfg, "n_eval", 100))  # was 20 in v2
    n_repeats = int(getattr(cfg, "n_repeats", 3))  # was 1 in v2
    eval_llms = test_llms[:n_eval]
    criteria_str = str(getattr(cfg, "criteria", "random,heuristic,trace,doptimal"))
    criteria = [c.strip() for c in criteria_str.split(",")]

    print(f"\nCD-CAT v2 FIXED: {n_eval} LLMs, N={cal_sizes}, repeats={n_repeats}",
          flush=True)
    print(f"  Selector: warm-start MAP (lr=0.01, 15 steps) for guidance only",
          flush=True)
    print(f"  Final fit (both methods): lr={FINAL_FIT_LR}, steps={FINAL_FIT_STEPS}, "
          f"fresh from zero", flush=True)
    print(f"  Pre-filter: top {TOP_K_PREFILTER} by heuristic, then exact autograd",
          flush=True)

    results = {crit: {N: [] for N in cal_sizes} for crit in criteria}

    for li, llm_idx in enumerate(eval_llms):
        t0 = time.time()
        responses = R[llm_idx]
        y_test = responses[test_items.astype(int)]
        if len(np.unique(y_test)) < 2:
            continue

        for N in cal_sizes:
            for rep in range(n_repeats):
                rng = np.random.RandomState(42 + rep)

                # Random branch
                cal = rng.choice(train_items, size=min(N, len(train_items)),
                                  replace=False)
                mastery = fit_theta_fixed(net, responses, cal, q_matrix, text_embs,
                                            device, K)
                preds = predict_with_theta(net, mastery, test_items, q_matrix,
                                             text_embs, device)
                results["random"][N].append(roc_auc_score(y_test, preds))

                # Adaptive branches share the same selector path
                for crit in [c for c in criteria if c != "random"]:
                    theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
                    remaining = train_items.copy()
                    items_seen, responses_seen = [], []
                    I_cum_inv = ((1.0 / 0.01) * np.eye(K)
                                 if crit == "doptimal" else None)

                    for step in range(min(N, len(train_items))):
                        chosen, I_cum_inv = prefilter_and_select(
                            net, theta_raw, remaining, q_matrix, text_embs,
                            device, crit, I_cum_inv)
                        items_seen.append(chosen)
                        responses_seen.append(float(responses[chosen]))
                        remaining = remaining[remaining != chosen]
                        # Guide-only MAP for selector context (warm-start, cheap)
                        _, theta_raw = map_update(
                            net, items_seen, responses_seen,
                            q_matrix, text_embs, device, K,
                            init_theta_raw=theta_raw)

                    cal_chosen = np.array(items_seen)
                    mastery = fit_theta_fixed(net, responses, cal_chosen,
                                                q_matrix, text_embs, device, K)
                    preds = predict_with_theta(net, mastery, test_items,
                                                 q_matrix, text_embs, device)
                    results[crit][N].append(roc_auc_score(y_test, preds))

        elapsed = time.time() - t0
        if (li + 1) % 5 == 0 or li == 0:
            est = elapsed * (n_eval - li - 1) / 60.0
            print(f"  LLM {li+1}/{n_eval}: {elapsed:.1f}s, est. remaining: {est:.0f}min",
                  flush=True)

    print(f"\n{'='*90}", flush=True)
    print(f"CD-CAT v2 FIXED RESULTS ({n_eval} LLMs, {n_repeats} seeds)", flush=True)
    print(f"{'='*90}", flush=True)
    hdr = f"{'N':>6}"
    for c in criteria:
        hdr += f"  {c:>14}"
    print(hdr, flush=True)
    print("-" * 90, flush=True)

    summary = []
    for N in cal_sizes:
        row = {"N": N}
        line = f"{N:>6}"
        for c in criteria:
            vals = results[c][N]
            m, s = float(np.mean(vals)), float(np.std(vals))
            line += f"  {m:>8.4f}±{s:.3f}"
            row[f"{c}_mean"] = m; row[f"{c}_std"] = s
        print(line, flush=True)
        summary.append(row)

    save_data = {
        "experiment": "adaptive_testing_v2_fixed",
        "t_id": "T-038",
        "fix": "symmetric final-fit (T-036 recipe), 100 LLMs (was 20), 3 seeds (was 1)",
        "checkpoint": str(ckpt_path),
        "n_eval_llms": n_eval, "n_repeats": n_repeats, "K": K,
        "cal_sizes": cal_sizes, "criteria": criteria,
        "top_k_prefilter": TOP_K_PREFILTER,
        "final_fit_lr": FINAL_FIT_LR, "final_fit_steps": FINAL_FIT_STEPS,
        "selector_uses_exact_autograd": True,
        "summary": summary,
    }
    out_json = Path(f"cdm_exploration/experiments/v2_adaptive_testing_v2_fixed{out_tag}.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    verified = verify_splits(train_items, test_items, label="adaptive_testing_v2_fixed")
    log_experiment(
        name=f"adaptive_testing_v2_fixed{out_tag}",
        config={"n_eval": n_eval, "K": K, "n_repeats": n_repeats,
                "cal_sizes": cal_sizes, "criteria": criteria,
                "top_k_prefilter": TOP_K_PREFILTER,
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
