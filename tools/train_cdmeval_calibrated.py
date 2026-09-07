"""Train CDMEval (TextConditionedNet, K=100) on a 60/20/20 item split for
post-hoc calibration (T-033).

The 60/20/20 split is deterministic from the canonical 80/20 (random_state=42)
by sub-splitting train_idx into train_inner (75% of train) and val (25%).

Splits:
  - train_inner: 5,713 items — model trains on triplets here only
  - val:         1,905 items — held out from training, used for calibration
  - test:        1,905 items — canonical test set

After training: emit prediction matrices on val + test for downstream
isotonic calibration and Pareto re-evaluation.

NO best-epoch checkpointing on val (would leak val into model selection).
Fixed 15 epochs matching the protocolB recipe.

Usage:
    python tools/train_cdmeval_calibrated.py [--seed 42] [--epochs 15] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "expanded"
EXP = REPO / "cdm_exploration" / "experiments"


def make_calibrated_split(n_items: int, seed: int = 42):
    """Deterministic 60/20/20 item split (delegates to canonical source).

    See ``cdmeval.utils.calibration_split.make_canonical_calibrated_split``
    for the single source of truth.
    """
    from cdmeval.utils.calibration_split import (
        make_canonical_calibrated_split,
    )
    return make_canonical_calibrated_split(n_items=n_items, seed=seed)


def build_triplets_for_items(R: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
    n_llms = R.shape[0]
    n_items_sub = len(item_idx)
    user_ids = np.repeat(np.arange(n_llms), n_items_sub)
    item_ids = np.tile(item_idx, n_llms)
    scores = R[:, item_idx].ravel().astype(float)
    return np.column_stack([user_ids, item_ids, scores])


def predict_full_grid(net, item_idx: np.ndarray, n_llms: int,
                       text_emb: np.ndarray, q_matrix: np.ndarray,
                       device: str) -> np.ndarray:
    """Run inference on every (m, p) for p in item_idx. Returns (n_llms, len(item_idx))."""
    net.eval()
    n_eval = len(item_idx)
    preds = np.zeros((n_llms, n_eval), dtype=np.float32)
    te = torch.tensor(text_emb[item_idx], dtype=torch.float32, device=device)
    tq = torch.tensor(q_matrix[item_idx], dtype=torch.float32, device=device)
    all_ids = torch.arange(n_llms, device=device)
    with torch.no_grad():
        for j in range(n_eval):
            te_j = te[j].unsqueeze(0).expand(n_llms, -1)
            tq_j = tq[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_ids, te_j, tq_j).cpu().numpy()
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs + 1000 train triplets for B1 smoke")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 3

    print(f"=== train_cdmeval_calibrated seed={args.seed} K={args.K} "
            f"epochs={args.epochs} smoke={args.smoke} ===", flush=True)

    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(args.seed)
    device = resolve_device("cuda")
    print(f"  device: {device}", flush=True)

    # ── Data ──
    print("\n[load] data", flush=True)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / f"qmatrix_v2_K{args.K}.npy")
    text_emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    print(f"  R: {R.shape}, q_matrix: {q_matrix.shape}, emb: {text_emb.shape}",
            flush=True)

    # ── 60/20/20 item split ──
    train_inner_idx, val_idx, test_idx = make_calibrated_split(n_items, seed=42)
    print(f"  split (seed=42): train_inner={len(train_inner_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}", flush=True)

    # Save the split definition for downstream consumers
    split_path = DATA / "calibration_split.json"
    with open(split_path, "w") as f:
        json.dump({
            "n_items": int(n_items),
            "train_inner_idx": train_inner_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "seed": 42,
            "description": "60/20/20 item split for T-033 calibration. "
                            "Canonical test=1905 unchanged; train sub-split into "
                            "train_inner (5713) + val (1905).",
        }, f, indent=2)
    print(f"  wrote split definition to {split_path}", flush=True)

    # ── Build train triplets ──
    print("\n[build] training triplets", flush=True)
    train_triplets = build_triplets_for_items(R, train_inner_idx)
    if args.smoke:
        train_triplets = train_triplets[:1000]
    print(f"  train triplets: {len(train_triplets):,}", flush=True)

    # Build dataloader
    user_ids = torch.tensor(train_triplets[:, 0], dtype=torch.int64)
    scores = torch.tensor(train_triplets[:, 2], dtype=torch.float32)
    item_indices_t = train_triplets[:, 1].astype(int)
    text_embs_t = torch.tensor(text_emb[item_indices_t], dtype=torch.float32)
    knowledge_embs_t = torch.tensor(q_matrix[item_indices_t], dtype=torch.float32)
    train_ds = TensorDataset(user_ids, text_embs_t, knowledge_embs_t, scores)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, generator=g)

    # ── Model ──
    print("\n[model] TextConditionedNet K=100", flush=True)
    net = TextConditionedNet(n_skills, n_llms, text_dim=text_emb.shape[1])
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # ── Train (fixed epochs, no best-AUC selection) ──
    print(f"\n[train] {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}",
            flush=True)
    losses_per_epoch = []
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        ep_losses = []
        for user_id, te, ke, y in train_loader:
            user_id = user_id.to(device)
            te = te.to(device)
            ke = ke.to(device)
            y = y.to(device)
            pred = net(user_id, te, ke)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
        avg = float(np.mean(ep_losses))
        losses_per_epoch.append(avg)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}  "
                f"elapsed={elapsed / 60:.1f} min", flush=True)

    # ── Inference on val + test ──
    print("\n[infer] val items", flush=True)
    val_preds = predict_full_grid(net, val_idx, n_llms, text_emb, q_matrix, device)
    print(f"  val_preds: {val_preds.shape}, range: "
            f"[{val_preds.min():.4f}, {val_preds.max():.4f}]", flush=True)

    print("\n[infer] test items", flush=True)
    test_preds = predict_full_grid(net, test_idx, n_llms, text_emb, q_matrix, device)
    print(f"  test_preds: {test_preds.shape}, range: "
            f"[{test_preds.min():.4f}, {test_preds.max():.4f}]", flush=True)

    # ── Quick metrics ──
    from sklearn.metrics import roc_auc_score
    R_val = R[:, val_idx]
    R_test = R[:, test_idx]
    val_auc = float(roc_auc_score(R_val.flatten(), val_preds.flatten()))
    test_auc = float(roc_auc_score(R_test.flatten(), test_preds.flatten()))
    print(f"\n[metrics] val_auc={val_auc:.4f}  test_auc={test_auc:.4f}",
            flush=True)

    # ECE on val (uncalibrated)
    bins = np.linspace(0, 1, 11)
    pf = val_preds.flatten()
    tf = R_val.flatten()
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (pf >= lo) & (pf < hi) if i < len(bins) - 2 else (pf >= lo) & (pf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(pf)) * abs(pf[mask].mean() - tf[mask].mean())
    print(f"[metrics] val_ECE_uncalibrated={ece:.4f}", flush=True)

    # ── Save checkpoint + predictions ──
    if not args.smoke:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CKPT_DIR / f"text_conditioned_K{args.K}_calibrated_s{args.seed}.pt"
        torch.save({
            "model_state_dict": net.state_dict(),
            "config": {
                "K": args.K, "seed": args.seed, "epochs": args.epochs,
                "lr": args.lr, "batch_size": args.batch_size,
                "n_llms": n_llms, "n_items": n_items, "n_skills": n_skills,
                "split": "60_20_20_calibrated",
            },
            "final_train_loss": losses_per_epoch[-1],
            "val_auc_uncalibrated": val_auc,
            "test_auc_uncalibrated": test_auc,
            "val_ece_uncalibrated": float(ece),
        }, ckpt_path)
        print(f"\n  ckpt: {ckpt_path}", flush=True)

    # Save predictions as npz (compact)
    suffix = "_smoke" if args.smoke else ""
    pred_path = EXP / f"v2_cdmeval_calibrated{suffix}_predictions_s{args.seed}.npz"
    np.savez(pred_path,
              val_preds=val_preds, val_idx=val_idx,
              test_preds=test_preds, test_idx=test_idx,
              train_inner_idx=train_inner_idx,
              R_val=R_val, R_test=R_test)
    print(f"  predictions: {pred_path}", flush=True)

    # Save metrics JSON
    out = {
        "experiment": "cdmeval_calibrated_60_20_20",
        "seed": args.seed,
        "K": args.K,
        "smoke": args.smoke,
        "split": {
            "train_inner": int(len(train_inner_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "config": {"epochs": args.epochs, "lr": args.lr,
                    "batch_size": args.batch_size},
        "train_losses": losses_per_epoch,
        "metrics_uncalibrated": {
            "val_auc": val_auc,
            "test_auc": test_auc,
            "val_ece": float(ece),
        },
        "verified": True,
    }
    json_path = EXP / f"v2_cdmeval_calibrated{suffix}_s{args.seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  metrics: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
