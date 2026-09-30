#!/usr/bin/env python3
"""Table 2 baseline (b): IRT 2PL with item parameters from the item text (an external review, 2026-09-22).

The existing 2PL row (tools/run_irt_baseline_v2_protocolA.py) learns a free discrimination and difficulty per item.
Here both come from a linear projection of the item's SBERT embedding, as SkillEval's item parameters do
(cdmeval/modeling/text_item_irt.py). Everything else follows the 2PL run: one ability per LLM, Adam lr 0.005, batch
16384, up to 15 epochs, stop after 3 epochs without validation gain, keep the best-validation-AUC weights.

Splits (identical to the rows it is compared with):
  A (ID): 80/10/10 over all (LLM, item) triplets, random_state 42, as the 2PL and SkillEval ID values.
  B (OOD): items 80/20 random_state 42, validation = 10% of training triplets random_state 42, as SkillEval's OOD
           value (tools/run_multi_seed.py, tools/train_expanded.py). Test = every LLM on the 1,905 test items.
The seed changes only initialisation and shuffling.

Also saved, to check the baseline is not degenerate: spread of the item parameters, range of the predictions.

    python tools/run_irt2pl_textitems.py --protocol A --seed 42 --device cuda
    python tools/run_irt2pl_textitems.py --protocol B --seed 43 --device cuda
    python tools/run_irt2pl_textitems.py --protocol A --device cpu --n_llms 100 --epochs 2    # smoke test
    python tools/run_irt2pl_textitems.py --protocol B --epochs 60 --patience 1000 --tag _conv60 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.modeling.text_item_irt import TextItemIRT2PL
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, save_checkpoint, verify_splits

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
CKPT = REPO / "cdm_exploration/checkpoints/expanded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n_llms", type=int, default=0, help="smoke test: random subset of LLMs")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _conv60 (convergence checks)")
    args = ap.parse_args()
    smoke = args.n_llms > 0
    device = args.device

    seed_everything(args.seed)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    text = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    if smoke:
        R = R[np.random.RandomState(42).choice(R.shape[0], size=args.n_llms, replace=False)]
    n_llms, n_items = R.shape
    print(f"Device {device}, protocol {args.protocol}, seed {args.seed}: {n_llms} LLMs x {n_items} items")

    all_llm = np.repeat(np.arange(n_llms, dtype=np.int64), n_items)
    all_item = np.tile(np.arange(n_items, dtype=np.int64), n_llms)
    all_label = R.ravel().astype(np.float32)
    n_total = len(all_label)
    if args.protocol == "A":
        # Same calls as tools/run_irt_baseline_v2_protocolA.py and tools/run_ncdm_protocolA.py.
        all_idx = np.arange(n_total, dtype=np.int64)
        trvl_idx, te_idx = train_test_split(all_idx, test_size=0.10, random_state=42)
        tr_idx, va_idx = train_test_split(trvl_idx, test_size=10.0 / 90.0, random_state=42)
        ok = bool(len(np.intersect1d(tr_idx, te_idx)) == 0 and len(np.intersect1d(va_idx, te_idx)) == 0)
        train_items = test_items = np.arange(n_items)
    else:
        # Same calls as tools/run_multi_seed.py and tools/train_expanded.py.
        train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
        ok = verify_splits(train_items, test_items, expected_seed=42, label=f"irt2pl_textitems_B_s{args.seed}")
        is_train = np.isin(all_item, train_items)
        tv = np.where(is_train)[0]
        te_idx = np.where(~is_train)[0]
        tr_pos, va_pos = train_test_split(np.arange(len(tv)), test_size=0.1, random_state=42)
        tr_idx, va_idx = tv[tr_pos], tv[va_pos]
        ok = ok and not np.isin(all_item[tr_idx], test_items).any() and not np.isin(all_item[va_idx], test_items).any()
    print(f"  split: train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}  leakage check {'PASS' if ok else 'FAIL'}")
    assert ok

    s_all, i_all, y_all = map(torch.from_numpy, (all_llm, all_item, all_label))
    s_tr, i_tr, y_tr = s_all[tr_idx], i_all[tr_idx], y_all[tr_idx]
    s_va, i_va, y_va = s_all[va_idx], i_all[va_idx], y_all[va_idx]
    s_te, i_te, y_te = s_all[te_idx], i_all[te_idx], y_all[te_idx]
    del s_all, i_all, y_all, all_llm, all_item, all_label

    model = TextItemIRT2PL(n_llms, text).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  TextItemIRT2PL: {n_params:,} trainable parameters; lr {args.lr}, batch {args.batch_size}, "
          f"epochs {args.epochs}, patience {args.patience}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()

    def evaluate(s_, i_, y_, bs=65536):
        model.eval()
        with torch.no_grad():
            p = np.concatenate([model(s_[a:a + bs].to(device), i_[a:a + bs].to(device)).cpu().numpy()
                                for a in range(0, len(y_), bs)])
        y = y_.numpy()
        return (roc_auc_score(y, p), accuracy_score(y, (p >= 0.5).astype(int)),
                float(np.sqrt(((y - p) ** 2).mean())), p)

    best_auc, best_state, bad, history = 0.0, None, 0, []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(y_tr))
        losses = []
        for a in range(0, len(y_tr), args.batch_size):
            idx = perm[a:a + args.batch_size]
            loss = loss_fn(model(s_tr[idx].to(device), i_tr[idx].to(device)), y_tr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va_auc, va_acc, _, _ = evaluate(s_va, i_va, y_va)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_auc": float(va_auc),
                        "val_acc": float(va_acc), "elapsed_s": time.time() - t0})
        print(f"  Epoch {epoch + 1:>2}/{args.epochs}: loss={np.mean(losses):.4f} val_auc={va_auc:.4f} "
              f"val_acc={va_acc:.4f} ({time.time() - t0:.1f}s)")
        if va_auc > best_auc:
            best_auc, bad = va_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {epoch + 1}")
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_auc, test_acc, test_rmse, p_te = evaluate(s_te, i_te, y_te)
    with torch.no_grad():
        alpha, beta = (x.cpu().numpy() for x in model.item_params())
    diag = {"log_discrimination_mean": float(alpha.mean()), "log_discrimination_std": float(alpha.std()),
            "difficulty_mean": float(beta.mean()), "difficulty_std": float(beta.std()),
            "theta_std": float(model.theta.weight.detach().cpu().numpy().std()),
            "pred_min": float(p_te.min()), "pred_max": float(p_te.max()), "pred_std": float(p_te.std())}
    print(f"  TEST: auc={test_auc:.4f} acc={test_acc:.4f} rmse={test_rmse:.4f}")
    print(f"  degeneracy check: {diag}")

    tag = f"protocol{args.protocol}_s{args.seed}{args.tag}" + ("_smoke" if smoke else "")
    res = {"experiment": "irt2pl_textitems", "model": "TextItemIRT2PL", "protocol": args.protocol,
           "split": "interaction_wise_80_10_10" if args.protocol == "A" else "item_wise_80_20_val_triplet_10",
           "seed": args.seed, "n_llms": n_llms, "n_items": n_items, "n_test_items": int(len(test_items)),
           "n_train_triplets": int(len(tr_idx)), "n_val_triplets": int(len(va_idx)), "n_test_triplets": int(len(te_idx)),
           "lr": args.lr, "batch_size": args.batch_size, "epochs_max": args.epochs, "patience": args.patience,
           "epochs_trained": len(history), "trainable_params": n_params, "best_val_auc": float(best_auc),
           "test_auc": float(test_auc), "test_acc": float(test_acc), "test_rmse": float(test_rmse),
           "degeneracy_check": diag, "history": history, "tag": args.tag, "verified": bool(ok)}
    (EXP / f"v2_irt2pl_textitems_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"  saved v2_irt2pl_textitems_{tag}.json")
    if not smoke:
        save_checkpoint(model, CKPT / f"irt2pl_textitems_{tag}.pt",
                        config={k: res[k] for k in ("protocol", "seed", "lr", "batch_size", "epochs_trained")},
                        train_items=train_items, test_items=test_items, val_auc=best_auc, epoch=len(history))
        log_experiment(name=f"irt2pl_textitems_{tag}", config={k: res[k] for k in ("protocol", "seed", "lr", "batch_size")},
                       results={k: res[k] for k in ("test_auc", "test_acc", "test_rmse", "best_val_auc", "degeneracy_check")},
                       split_info={k: res[k] for k in ("split", "n_train_triplets", "n_val_triplets", "n_test_triplets")},
                       verified=bool(ok))


if __name__ == "__main__":
    main()
