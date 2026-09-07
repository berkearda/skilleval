"""Re-run a few epochs of IRT 2PL Protocol A and check val_auc curve matches."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from cdmeval.utils.device import seed_everything, resolve_device
from cdmeval.evaluation.baselines import IRT2PL


def main() -> None:
    seed_everything(42)
    DATA = ROOT / "cdm_exploration" / "data" / "cdm_ready"
    device = resolve_device("mps")

    R = np.load(DATA / "response_matrix_v2_full.npy")
    n_llms, n_items = R.shape

    all_llm = np.repeat(np.arange(n_llms, dtype=np.int32), n_items)
    all_item = np.tile(np.arange(n_items, dtype=np.int32), n_llms)
    all_label = R[all_llm, all_item].astype(np.float32)
    n_total = len(all_label)
    all_idx = np.arange(n_total, dtype=np.int64)
    trvl_idx, te_idx = train_test_split(all_idx, test_size=0.10, random_state=42)
    tr_idx, va_idx = train_test_split(trvl_idx, test_size=10.0 / 90.0, random_state=42)

    s_all = torch.from_numpy(all_llm.astype(np.int64))
    i_all = torch.from_numpy(all_item.astype(np.int64))
    y_all = torch.from_numpy(all_label)
    s_tr, i_tr, y_tr = s_all[tr_idx], i_all[tr_idx], y_all[tr_idx]
    s_va, i_va, y_va = s_all[va_idx], i_all[va_idx], y_all[va_idx]

    epochs = 3
    lr = 0.005
    bs = 16384
    model = IRT2PL(n_llms, n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    n_train = len(s_tr)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        for start in range(0, n_train, bs):
            idx = perm[start : start + bs]
            s_b = s_tr[idx].to(device)
            i_b = i_tr[idx].to(device)
            y_b = y_tr[idx].to(device)
            pred = model(s_b, i_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        bs_eval = 65536
        preds = []
        with torch.no_grad():
            for start in range(0, len(y_va), bs_eval):
                end = min(start + bs_eval, len(y_va))
                p = model(s_va[start:end].to(device),
                          i_va[start:end].to(device)).cpu().numpy()
                preds.append(p)
        preds = np.concatenate(preds)
        va_auc = roc_auc_score(y_va.numpy(), preds)
        va_acc = accuracy_score(y_va.numpy(), (preds >= 0.5).astype(int))
        avg_loss = float(np.mean(losses))
        print(
            f"  Epoch {epoch + 1}: loss={avg_loss:.4f} val_auc={va_auc:.4f} "
            f"val_acc={va_acc:.4f} ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
