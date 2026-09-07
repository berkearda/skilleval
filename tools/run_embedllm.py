"""Train EmbedLLM TextMF on the CDMEval matrix and emit our standard metrics.

What this is:
  - Loads ``train.csv`` / ``test.csv`` / ``question_embeddings.pth`` produced
    by ``tools/build_embedllm_inputs.py``.
  - Trains EmbedLLM's TextMF (cdm_exploration/repos/EmbedLLM/algorithm/mf.py)
    end-to-end with their published hyperparameters.
  - F10 mitigation: re-seeds AFTER importing TextMF, since ``mf.py`` calls
    ``torch.manual_seed(42)`` / ``np.random.seed(42)`` / ``random.seed(42)``
    at module-top (lines 14-16). Without this, all per-seed runs collapse
    to byte-identical results — same trap as IrtNet (the project log
    2026-04-21).
  - Custom inference loop emitting ``softmax(logits)[:, 1]`` per (m, p) on
    the full n_llms × n_test grid. Used for AUC, Acc@1/3/5/10, per-bench
    Acc@1, and benchmark-level Pearson r — same metrics as our other
    Table 1 rows.
  - Drops EmbedLLM's internal ``acc_dict`` and ``weighted_accuracy``,
    which are TEST-leaky (mf.py:232-236).

Usage:
    python tools/run_embedllm.py --d_model 232 --seed 42 [--smoke]

Smoke mode trains for 3 epochs on the smoke split written by
``build_embedllm_inputs.py --smoke`` (B1 of the N10 gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EMBEDLLM_DATA = REPO / "cdm_exploration" / "data" / "embedllm"
EMBEDLLM_REPO = REPO / "cdm_exploration" / "repos" / "EmbedLLM" / "algorithm"
CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "embedllm"
EXP = REPO / "cdm_exploration" / "experiments"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def import_textmf():
    if not (EMBEDLLM_REPO / "mf.py").exists():
        raise FileNotFoundError(
            f"mf.py not found at {EMBEDLLM_REPO}; run "
            "git clone https://github.com/richardzhuang0412/EmbedLLM.git into "
            "cdm_exploration/repos/EmbedLLM"
        )
    sys.path.insert(0, str(EMBEDLLM_REPO))
    from mf import TextMF  # type: ignore[import-not-found]
    return TextMF


def load_data(smoke: bool, calibrated: bool = False):
    """Load the train/test CSVs + embedding tensor produced by builder."""
    if smoke:
        base = EMBEDLLM_DATA / "smoke"
    elif calibrated:
        base = EMBEDLLM_DATA / "calibrated"
    else:
        base = EMBEDLLM_DATA
    train_csv = pd.read_csv(base / "train.csv")
    test_csv = pd.read_csv(base / "test.csv")
    question_embeddings = torch.load(base / "question_embeddings.pth",
                                        weights_only=False)
    return train_csv, test_csv, question_embeddings, base


def assert_input_contracts(train_csv, test_csv, n_llms_expected, n_items_expected):
    """Every LLM appears in both splits AND is a dense 0..n-1 index.

    EmbedLLM's CustomDataset re-ranks model_ids via torch.unique sorted=True
    (mf.py:55-58). The re-rank is identity iff unique sets match between
    splits AND are dense 0..n-1. If that's violated, train and test embed
    different LLMs at the same index and AUC silently collapses.
    """
    train_uids = sorted(train_csv["model_id"].unique().tolist())
    test_uids = sorted(test_csv["model_id"].unique().tolist())
    expected = list(range(n_llms_expected))
    assert train_uids == expected, "train.csv model_id set mismatch"
    assert test_uids == expected, "test.csv model_id set mismatch"
    assert set(train_csv["label"].unique()) <= {0, 1}
    assert set(test_csv["label"].unique()) <= {0, 1}
    print(f"  contracts ok: {n_llms_expected} LLMs in both splits, dense 0..n-1",
          flush=True)


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for models, prompts, labels in train_loader:
        models = models.to(device)
        prompts = prompts.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(models, prompts)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        n += labels.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_full_grid(model, n_llms: int, test_prompt_ids: np.ndarray,
                      device: str, chunk: int = 256) -> np.ndarray:
    """Emit P(correct) on the full (n_llms, n_test) grid via softmax(logits)[:,1].

    Returns a numpy array of shape (n_llms, n_test) of float32 probabilities.
    """
    model.eval()
    n_test = len(test_prompt_ids)
    preds = np.zeros((n_llms, n_test), dtype=np.float32)
    p_tensor = torch.tensor(test_prompt_ids, dtype=torch.long, device=device)
    for i in range(0, n_llms, chunk):
        m_ids = torch.arange(i, min(i + chunk, n_llms), device=device)
        # broadcast: (chunk, n_test)
        m_grid = m_ids.unsqueeze(1).expand(-1, n_test).reshape(-1)
        p_grid = p_tensor.unsqueeze(0).expand(len(m_ids), -1).reshape(-1)
        logits = model(m_grid, p_grid, test_mode=True)  # (chunk*n_test, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds[i:i + len(m_ids)] = probs.view(len(m_ids), n_test).cpu().numpy()
    return preds


def compute_metrics(preds: np.ndarray, R_test: np.ndarray, test_bench: np.ndarray) -> dict:
    """Compute test AUC, test acc, Acc@1/3/5/10, per-bench Acc@1,
    benchmark-level Pearson r — consistent with v2_knn_baseline.json,
    v2_irtnet_d100_perbench.json, v2_benchmark_prediction.json.
    """
    from scipy.stats import pearsonr

    n_llms, n_test = preds.shape
    flat_pred = preds.flatten()
    flat_true = R_test.flatten()
    auc = float(roc_auc_score(flat_true, flat_pred))
    pred_label = (preds >= 0.5).astype(int)
    test_acc = float((pred_label == R_test).mean())

    # Routing Acc@k: for each test item, take the top-k LLMs by pred score,
    # success = at least one of those k LLMs was correct on that item.
    acc_at_k = {}
    for k in [1, 3, 5, 10]:
        # argpartition along axis 0 (LLMs) descending
        top_k_idx = np.argpartition(-preds, kth=k - 1, axis=0)[:k]  # (k, n_test)
        # Did any of the top-k get it right?
        hits = R_test[top_k_idx, np.arange(n_test)[None, :]].max(axis=0)
        acc_at_k[f"acc@{k}"] = float(hits.mean())

    # Per-benchmark Acc@1
    per_bench = {}
    top1 = np.argmax(preds, axis=0)  # (n_test,)
    correct1 = R_test[top1, np.arange(n_test)]
    for b in BENCHMARKS:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        per_bench[b] = {
            "n_items": int(mask.sum()),
            "acc@1": float(correct1[mask].mean()),
        }

    # Benchmark-level Pearson r per LLM:
    # for each LLM, predicted mean = mean of preds on items in benchmark b.
    # True mean = mean of R_test on items in benchmark b.
    # Correlate predicted vs true across n_llms LLMs.
    bench_pred_r = {}
    for b in BENCHMARKS:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        true_acc = R_test[:, mask].mean(axis=1)
        pred_acc = preds[:, mask].mean(axis=1)
        if true_acc.std() == 0 or pred_acc.std() == 0:
            bench_pred_r[b] = float("nan")
        else:
            r, _ = pearsonr(true_acc, pred_acc)
            bench_pred_r[b] = float(r)
    # Concatenated overall
    true_all = np.concatenate([R_test[:, test_bench == b].mean(axis=1)
                                  for b in BENCHMARKS if (test_bench == b).any()])
    pred_all = np.concatenate([preds[:, test_bench == b].mean(axis=1)
                                  for b in BENCHMARKS if (test_bench == b).any()])
    if true_all.std() == 0 or pred_all.std() == 0:
        bench_pred_r["overall"] = float("nan")
    else:
        r_all, _ = pearsonr(true_all, pred_all)
        bench_pred_r["overall"] = float(r_all)

    return {
        "test_auc": auc,
        "test_acc": test_acc,
        "routing": acc_at_k,
        "per_benchmark": per_bench,
        "benchmark_prediction_r": bench_pred_r,
    }


def diagnose(preds: np.ndarray, R_test: np.ndarray) -> dict:
    """Pathology checks: F2 (degenerate routing), F14 (garbage-out)."""
    top1 = np.argmax(preds, axis=0)
    n_unique = int(len(np.unique(top1)))
    return {
        "n_unique_routing_picks": n_unique,
        "fraction_picks_concentrated_on_top_1_llm": float(
            np.bincount(top1).max() / len(top1)
        ),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=232)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true",
                    help="Use the smoke split + 3 epochs.")
    ap.add_argument("--calibrated", action="store_true",
                    help="Use the 60/20/20 calibrated split (train_inner + val); "
                          "after training, also infer on canonical test items.")
    args = ap.parse_args()

    if args.smoke:
        args.num_epochs = 3

    print(f"=== EmbedLLM TextMF d={args.d_model} seed={args.seed} "
            f"smoke={args.smoke} ===", flush=True)

    sys.path.insert(0, str(REPO))
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(args.seed)
    device = resolve_device("cuda")
    print(f"  device: {device}", flush=True)

    # Import TextMF (mf.py at module-top calls torch/np/random.seed(42))
    TextMF = import_textmf()
    # F10 mitigation: re-seed AFTER the import so our seed wins.
    seed_everything(args.seed)
    # Sanity: torch.initial_seed() should match (cuda guard)
    print(f"  torch.initial_seed() after re-seed: {torch.initial_seed()}",
            flush=True)

    # ── Load data + assert contracts ──
    print("\n[load] reading CSVs and embedding tensor ...", flush=True)
    train_csv, test_csv, question_embeddings, base = load_data(args.smoke,
                                                                  calibrated=args.calibrated)
    n_llms = int(test_csv["model_id"].max() + 1)
    n_items = int(question_embeddings.shape[0])
    assert_input_contracts(train_csv, test_csv, n_llms, n_items)
    print(f"  train rows: {len(train_csv):,}  test rows: {len(test_csv):,}",
            flush=True)

    # ── Build dataloaders the way EmbedLLM does ──
    from torch.utils.data import DataLoader, TensorDataset
    train_models = torch.tensor(train_csv["model_id"].values, dtype=torch.long)
    train_prompts = torch.tensor(train_csv["prompt_id"].values, dtype=torch.long)
    train_labels = torch.tensor(train_csv["label"].values, dtype=torch.long)
    train_ds = TensorDataset(train_models, train_prompts, train_labels)
    # shuffle=True for SGD; EmbedLLM's stock script has shuffle=False, but
    # with batch_size 2048 and 29M rows shuffle=True is harmless and gives
    # better convergence. Keep determinism via seed_everything.
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, generator=g)

    # ── Construct TextMF ──
    print("\n[model] building TextMF ...", flush=True)
    model = TextMF(question_embeddings=question_embeddings,
                    model_embedding_dim=args.d_model, alpha=args.alpha,
                    num_models=n_llms, num_prompts=n_items)
    model = model.to(device)

    # Sanity: are model.P weights distinct between seeds?
    p_init_norm = float(model.P.weight.detach().abs().mean().cpu())
    print(f"  P init mean abs: {p_init_norm:.6f}  "
            f"(should differ across seeds; F10 sniff)", flush=True)

    # ── Train ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n[train] {args.num_epochs} epochs, batch={args.batch_size}, "
            f"lr={args.lr}, alpha={args.alpha}", flush=True)
    losses = []
    t0 = time.time()
    for epoch in range(args.num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        losses.append(loss)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{args.num_epochs}  loss={loss:.4f}  "
                f"elapsed={elapsed / 60:.1f} min", flush=True)

    # ── Inference: emit full softmax-prob grid on test ──
    print("\n[infer] emitting test prediction grid ...", flush=True)
    test_prompt_ids = np.sort(test_csv["prompt_id"].unique())
    # Build R_test from CSV: pivot to (n_llms, n_test_items) ordered by
    # the same prompt id order we passed to predict_full_grid.
    pivot = test_csv.pivot(index="model_id", columns="prompt_id",
                            values="label")
    pivot = pivot.reindex(index=range(n_llms), columns=test_prompt_ids)
    R_test = pivot.values.astype(np.int8)
    print(f"  R_test: {R_test.shape}", flush=True)
    preds = predict_full_grid(model, n_llms, test_prompt_ids, device)
    print(f"  preds: {preds.shape}  range: [{preds.min():.4f}, {preds.max():.4f}]",
            flush=True)

    # ── Per-bench labels (only for the full run; smoke uses subset items) ──
    if not args.smoke:
        items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
        benches = np.array([it.get("benchmark") for it in items])
        test_bench = benches[test_prompt_ids]
    else:
        # Smoke: synthesize a single dummy benchmark so per-bench logic works
        test_bench = np.array(["MATH"] * len(test_prompt_ids))

    metrics = compute_metrics(preds, R_test, test_bench)
    diag = diagnose(preds, R_test)

    print("\n[results]", flush=True)
    print(f"  test AUC: {metrics['test_auc']:.4f}", flush=True)
    print(f"  test acc: {metrics['test_acc']:.4f}", flush=True)
    print(f"  routing : {metrics['routing']}", flush=True)
    print(f"  unique routing picks: {diag['n_unique_routing_picks']}/{n_llms}",
            flush=True)
    print(f"  pred range: [{diag['pred_min']:.4f}, {diag['pred_max']:.4f}]  "
            f"std={diag['pred_std']:.4f}", flush=True)
    if not args.smoke:
        for b in BENCHMARKS:
            if b in metrics["per_benchmark"]:
                print(f"    {b:>7}: acc@1={metrics['per_benchmark'][b]['acc@1']:.4f}",
                        flush=True)
        print(f"  bench_pred r overall: {metrics['benchmark_prediction_r'].get('overall', float('nan')):.4f}",
                flush=True)

    # ── If calibrated mode, also infer on canonical test items ──
    if args.calibrated and not args.smoke:
        print("\n[infer] canonical test items (held out from this training run)",
                flush=True)
        from sklearn.model_selection import train_test_split
        canonical_train_idx, canonical_test_idx = train_test_split(
            np.arange(n_items), test_size=0.2, random_state=42)
        canonical_test_idx = np.sort(canonical_test_idx)
        canonical_test_preds = predict_full_grid(model, n_llms,
                                                    canonical_test_idx, device)
        print(f"  canonical_test_preds: {canonical_test_preds.shape}, "
                f"range: [{canonical_test_preds.min():.4f}, "
                f"{canonical_test_preds.max():.4f}]", flush=True)

        # Save val + canonical test preds as npz for downstream calibration
        # The "test" preds we already computed above ARE the val preds for this
        # calibrated run.
        npz_path = EXP / f"v2_embedllm_calibrated_d{args.d_model}_s{args.seed}_predictions.npz"
        # R_test from CSV = R[:, val_idx] in calibrated mode
        # Build R for canonical test from the response matrix on disk
        R_full = np.load(DATA / "response_matrix_v2_full.npy")
        R_canonical_test = R_full[:, canonical_test_idx]
        np.savez(npz_path,
                  val_preds=preds, val_idx=test_prompt_ids,
                  test_preds=canonical_test_preds, test_idx=canonical_test_idx,
                  R_val=R_test, R_test=R_canonical_test)
        print(f"  predictions: {npz_path}", flush=True)

    # ── Save checkpoint + JSON ──
    if not args.smoke:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_calibrated" if args.calibrated else ""
        ckpt_path = CKPT_DIR / f"d{args.d_model}_s{args.seed}{suffix}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": args.d_model, "seed": args.seed,
                "num_epochs": args.num_epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "alpha": args.alpha, "n_llms": n_llms, "n_items": n_items,
            },
            "final_train_loss": losses[-1],
        }, ckpt_path)
        print(f"\n  ckpt: {ckpt_path}", flush=True)

    out = {
        "experiment": "embedllm_textmf_headtohead",
        "d_model": args.d_model,
        "seed": args.seed,
        "smoke": args.smoke,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_train_rows": int(len(train_csv)),
        "n_test_rows": int(len(test_csv)),
        "config": {
            "num_epochs": args.num_epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "alpha": args.alpha,
        },
        "train_losses": losses,
        "metrics": metrics,
        "diagnostics": diag,
        "verified": True,
    }
    EXP.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out_path = EXP / f"v2_embedllm_smoke_d{args.d_model}_s{args.seed}.json"
    elif args.calibrated:
        out_path = EXP / f"v2_embedllm_calibrated_d{args.d_model}_s{args.seed}.json"
    else:
        out_path = EXP / f"v2_embedllm_d{args.d_model}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
