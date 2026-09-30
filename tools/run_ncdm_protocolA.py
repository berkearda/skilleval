"""CDMEval NCDM K=100 under Protocol A (interaction-wise random triplet split).

Companion to run_k_ablation.py, but uses the random 80/10/10 triplet split
instead of the item-wise 80/20 split. Same v2 dataset (3811 LLMs x 9523
items, K=100), same TextConditionedNet model, same hyperparameters.

Reports test AUC, test Acc, test RMSE on the held-out 10% triplets.
Skips routing Acc@k because the routing metric does not translate cleanly
under interaction-wise splitting (every item has both train and test
cells, so the "argmax LLM per item" semantics differ from Protocol B).

Usage:
    python tools/run_ncdm_protocolA.py device=cuda            # default seed 42
    python tools/run_ncdm_protocolA.py device=cuda +seed=43

Output:
    cdm_exploration/experiments/v2_ncdm_protocolA_K100.json
        (or _s{seed}.json suffix when seed != 42)
    cdm_exploration/checkpoints/expanded/text_conditioned_protocolA_K100[suf].pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


K_FIXED = 100  # Protocol A run scoped to K=100 per the 2026-04-25 request


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_dataloader, make_text_dataloader
    from cdmeval.modeling.free_item import FreeItemNCDM
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint

    seed = int(cfg.seed) if hasattr(cfg, "seed") else 42
    seed_everything(seed)
    device = resolve_device(cfg.device)
    print(f"Device: {device}, K={K_FIXED}, seed={seed}, protocol=A", flush=True)

    # ── Load data ──
    data_dir = Path(cfg.paths.cdm_ready)
    print(f"\nLoading v2 data from {data_dir}", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / f"qmatrix_v2_K{K_FIXED}.npy")
    text_embs = np.load(
        data_dir / "item_text_embeddings_v2_full.npz",
    )["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    # +free_items=true: Table 2 baseline (a), the same network with free per-item difficulty and discrimination
    # instead of text-derived ones (an external review, 2026-09-22). Everything else is unchanged.
    free_items = bool(cfg.free_items) if hasattr(cfg, "free_items") else False
    # +subset.n_llms=N: smoke tests only (same random subset rule as train_expanded.py).
    if hasattr(cfg, "subset") and hasattr(cfg.subset, "n_llms"):
        idx = np.random.RandomState(42).choice(R.shape[0], size=int(cfg.subset.n_llms), replace=False)
        R = R[idx]
        llm_names = [llm_names[i] for i in idx]
        print(f"  SMOKE: {len(idx)}-LLM subset", flush=True)
    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    text_dim = text_embs.shape[1]
    assert n_skills == K_FIXED, (
        f"Q-matrix K={n_skills} does not match expected K={K_FIXED}"
    )
    print(
        f"  {n_llms} LLMs x {n_items} items, K={n_skills}, "
        f"text_dim={text_dim}",
        flush=True,
    )

    # ── Protocol A: 80/10/10 random split over ALL triplets ──
    print("\nBuilding all triplets...", flush=True)
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])
    n_total = len(triplets)
    print(f"  Total triplets: {n_total:,}", flush=True)

    all_idx = np.arange(n_total)
    trvl_idx, te_idx = train_test_split(
        all_idx, test_size=0.10, random_state=42,
    )
    tr_idx, va_idx = train_test_split(
        trvl_idx, test_size=10.0 / 90.0, random_state=42,
    )
    print(
        f"  Protocol A split: train={len(tr_idx):,}, val={len(va_idx):,}, "
        f"test={len(te_idx):,}",
        flush=True,
    )

    bs = cfg.model.batch_size
    # The free-item model takes item ids where the text model takes text embeddings; the training and
    # evaluation loops pass the second element of each batch through unchanged.
    make = (lambda t, sh: make_dataloader(t, q_matrix, bs, shuffle=sh)) if free_items else \
           (lambda t, sh: make_text_dataloader(t, text_embs, q_matrix, bs, shuffle=sh))
    train_loader = make(triplets[tr_idx], True)
    val_loader = make(triplets[va_idx], False)
    test_loader = make(triplets[te_idx], False)

    # ── Train ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    n_batches = len(triplets[tr_idx]) // bs
    print(
        f"\nTraining TextConditionedNet (K={K_FIXED}, {n_llms} LLMs, "
        f"Protocol A)",
        flush=True,
    )
    print(f"  {n_batches:,} batches/epoch, {epochs} epochs", flush=True)

    net = FreeItemNCDM(n_skills, n_llms, n_items) if free_items else TextConditionedNet(n_skills, n_llms, text_dim)
    print(f"  model: {type(net).__name__}", flush=True)
    net = train_text_model(
        net, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
    )

    # ── Evaluate ──
    print("\nEvaluating on held-out triplets...", flush=True)
    auc, acc, rmse = eval_text_model(net, test_loader, device)
    print(
        f"  Test AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}",
        flush=True,
    )

    # ── Persist ──
    seed_suffix = ("" if seed == 42 else f"_s{seed}") + ("_freeitems" if free_items else "") + \
                  ("_smoke" if hasattr(cfg, "subset") else "")
    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    ckpt_path = ckpt_dir / f"text_conditioned_protocolA_K{K_FIXED}{seed_suffix}.pt"
    save_checkpoint(
        net, ckpt_path,
        config={
            "K": K_FIXED, "seed": seed, "protocol": "A",
            "n_llms": n_llms, "n_items": n_items,
            "epochs": epochs, "lr": lr, "text_dim": text_dim,
            "free_items": free_items, "batch_size": int(bs),
        },
        train_items=np.array([], dtype=np.int64),  # not item-wise; sentinel
        test_items=np.array([], dtype=np.int64),
        val_auc=auc, epoch=epochs,
    )

    results = {
        "K": K_FIXED, "seed": seed, "protocol": "A",
        "split": "interaction_wise_80_10_10",
        "n_total_triplets": int(n_total),
        "n_train_triplets": int(len(tr_idx)),
        "n_val_triplets": int(len(va_idx)),
        "n_test_triplets": int(len(te_idx)),
        "test_auc": float(auc),
        "test_acc": float(acc),
        "test_rmse": float(rmse),
        "n_llms": n_llms, "n_items": n_items,
        "epochs": epochs, "lr": lr,
        "model": type(net).__name__, "free_items": free_items, "batch_size": int(bs),
    }
    out = Path(
        f"cdm_exploration/experiments/v2_ncdm_protocolA_K{K_FIXED}"
        f"{seed_suffix}_results.json",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

    # log_experiment requires train/test items for verify; we pass sentinels
    # because Protocol A is triplet-wise. Skip verify_splits, log directly.
    log_experiment(
        name=f"ncdm_protocolA_K{K_FIXED}{seed_suffix}",
        config={
            "model": type(net).__name__, "batch_size": int(bs),
            "K": K_FIXED, "seed": seed, "protocol": "A",
            "epochs": epochs, "lr": lr, "device": device,
        },
        results=results,
        split_info={
            "split_type": "interaction_wise_80_10_10",
            "n_total_triplets": int(n_total),
            "n_train_triplets": int(len(tr_idx)),
            "n_val_triplets": int(len(va_idx)),
            "n_test_triplets": int(len(te_idx)),
            "n_llms": n_llms, "n_items": n_items,
        },
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
