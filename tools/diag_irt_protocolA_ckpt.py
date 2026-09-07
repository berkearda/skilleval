"""Re-evaluate the saved IRT2PL Protocol A checkpoint to verify reported numbers.

Verifies:
  - Failure mode 4: IRT2PL forward = sigmoid(alpha * theta - beta), no extra
    shortcut, predictions in [0, 1], no NaN/Inf in saved state.
  - Failure mode 8: numerical stability of saved checkpoint.
  - Failure mode 11 (cross-check): re-compute test_auc / test_acc / test_rmse
    from the saved model on the same test split. Must match reported numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from cdmeval.utils.device import seed_everything
from cdmeval.evaluation.baselines import IRT2PL


def main() -> None:
    seed_everything(42)
    DATA = ROOT / "cdm_exploration" / "data" / "cdm_ready"

    R = np.load(DATA / "response_matrix_v2_full.npy")
    n_llms, n_items = R.shape

    all_llm = np.repeat(np.arange(n_llms, dtype=np.int32), n_items)
    all_item = np.tile(np.arange(n_items, dtype=np.int32), n_llms)
    all_label = R[all_llm, all_item].astype(np.float32)
    n_total = len(all_label)
    all_idx = np.arange(n_total, dtype=np.int64)
    trvl_idx, te_idx = train_test_split(all_idx, test_size=0.10, random_state=42)
    tr_idx, va_idx = train_test_split(trvl_idx, test_size=10.0 / 90.0, random_state=42)

    # ── Load checkpoint ──
    ckpt_path = ROOT / "cdm_exploration" / "checkpoints" / "expanded" / "irt_2pl_v2_protocolA.pt"
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(blob.keys())}")
    print(f"Config: {blob.get('config')}")
    print(f"Reported val_auc: {blob.get('val_auc')}")

    state = blob.get("state_dict") or blob.get("model_state_dict") or blob["model"]
    print(f"State_dict keys: {list(state.keys())}")
    for k, v in state.items():
        nan_ct = int(torch.isnan(v).sum())
        inf_ct = int(torch.isinf(v).sum())
        print(
            f"  {k}: shape={tuple(v.shape)}  "
            f"min={v.min().item():.4f}  max={v.max().item():.4f}  "
            f"mean={v.mean().item():.4f}  nan={nan_ct}  inf={inf_ct}"
        )

    # ── Build model and load ──
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    model = IRT2PL(n_llms, n_items)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params}")

    # ── Re-run test eval ──
    s_te = torch.from_numpy(all_llm[te_idx].astype(np.int64))
    i_te = torch.from_numpy(all_item[te_idx].astype(np.int64))
    y_te = all_label[te_idx]

    bs = 65536
    preds = []
    with torch.no_grad():
        for start in range(0, len(y_te), bs):
            end = min(start + bs, len(y_te))
            p = model(s_te[start:end].to(device),
                      i_te[start:end].to(device)).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds)
    print(
        f"Pred summary: min={preds.min():.4f}  max={preds.max():.4f}  "
        f"mean={preds.mean():.4f}  nan={int(np.isnan(preds).sum())}  "
        f"inf={int(np.isinf(preds).sum())}"
    )
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Pred out of [0,1]!"

    test_auc = roc_auc_score(y_te, preds)
    test_acc = accuracy_score(y_te, (preds >= 0.5).astype(int))
    test_rmse = float(np.sqrt(((y_te - preds) ** 2).mean()))
    print(f"\nRe-computed test_auc={test_auc:.4f}  "
          f"test_acc={test_acc:.4f}  test_rmse={test_rmse:.4f}")

    # ── Cross-check vs reported ──
    rep = json.load(open(ROOT / "cdm_exploration" / "experiments" / "v2_irt_baseline_protocolA.json"))
    print(f"Reported   test_auc={rep['test_auc']:.4f}  "
          f"test_acc={rep['test_acc']:.4f}  test_rmse={rep['test_rmse']:.4f}")

    # ── Verify formula matches: sigmoid(alpha * theta - beta) ──
    print("\nFormula spot-check vs IRT2PL.forward:")
    rng = np.random.RandomState(0)
    sample = rng.choice(len(y_te), size=200, replace=False)
    s_s = s_te[sample].to(device)
    i_s = i_te[sample].to(device)
    with torch.no_grad():
        theta = model.theta(s_s)
        alpha = torch.exp(model.alpha(i_s))
        beta = model.beta(i_s)
        manual = torch.sigmoid(alpha * theta - beta).squeeze(-1).cpu().numpy()
        forward = model(s_s, i_s).cpu().numpy()
    diff = np.abs(manual - forward).max()
    print(f"  max |manual - forward| = {diff:.2e}  (should be ~0)")


if __name__ == "__main__":
    main()
