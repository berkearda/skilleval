"""Benchmark-level prediction Pearson r for KNN and CDMEval.

For each LLM m and benchmark b:
  true_acc[m, b]  = mean correctness on test items in b
  pred_acc[m, b]  = mean predicted P(correct) on test items in b

Per-benchmark Pearson r: across 3,811 LLMs, correlate true_acc vs pred_acc.
Per-method overall: stack across all benchmarks and compute global r.

Reports CDMEval (paper checkpoint), CDMEval (5-seed mean ± std), KNN K=10.
IrtNet rows added later when the GPU job finishes.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cdmeval.modeling.text_conditioned import TextConditionedNet

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
PAPER_CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
MULTISEED_DIR = REPO / "cdm_exploration" / "checkpoints" / "multi_seed"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def load_setup():
    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    return R, emb, benches, train_idx, test_idx


def compute_benchmark_pred_r(preds: np.ndarray, R_test: np.ndarray,
                                test_bench: np.ndarray) -> dict:
    """preds: (n_llms, n_test). Returns per-bench r + overall."""
    out = {}
    for b in BENCHMARKS:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        true_acc = R_test[:, mask].mean(axis=1)
        pred_acc = preds[:, mask].mean(axis=1)
        r, p = pearsonr(true_acc, pred_acc)
        out[b] = {"n_items": int(mask.sum()), "pearson_r": float(r),
                    "p_value": float(p)}
    # overall: concatenate per-bench means
    true_all = np.concatenate([R_test[:, test_bench == b].mean(axis=1)
                                  for b in BENCHMARKS])
    pred_all = np.concatenate([preds[:, test_bench == b].mean(axis=1)
                                  for b in BENCHMARKS])
    r, p = pearsonr(true_all, pred_all)
    out["overall_concatenated"] = {"pearson_r": float(r), "p_value": float(p)}
    return out


def cdmeval_inference(ckpt_path: Path, emb: np.ndarray, q_matrix: np.ndarray,
                        test_idx: np.ndarray, n_llms: int) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    K = q_matrix.shape[1]
    net = TextConditionedNet(K, n_llms, text_dim=768)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32)
    test_q = torch.tensor(q_matrix[test_idx], dtype=torch.float32)
    all_llm_ids = torch.arange(n_llms)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            te = test_emb[j].unsqueeze(0).expand(n_llms, -1)
            tq = test_q[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_llm_ids, te, tq).numpy()
    return preds


def knn_inference(emb: np.ndarray, R: np.ndarray, train_idx: np.ndarray,
                    test_idx: np.ndarray, K: int = 10) -> np.ndarray:
    """KNN K=10: for each test item, retrieve K nearest train items by SBERT cosine,
    return per-LLM mean correctness on those K."""
    sim = cosine_similarity(emb[test_idx], emb[train_idx])
    top_k_local = np.argpartition(-sim, K, axis=1)[:, :K]
    top_k_global = train_idx[top_k_local]
    n_llms = R.shape[0]
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    for j, neighbors in enumerate(top_k_global):
        preds[:, j] = R[:, neighbors].mean(axis=1)
    return preds


def main() -> None:
    print("=== Benchmark-level prediction Pearson r ===\n", flush=True)
    R, emb, benches, train_idx, test_idx = load_setup()
    R_test = R[:, test_idx]
    test_bench = benches[test_idx]
    n_llms = R.shape[0]
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")

    summary = {}

    # ── CDMEval paper checkpoint ──
    print("[CDMEval paper checkpoint]", flush=True)
    preds = cdmeval_inference(PAPER_CKPT, emb, q_matrix, test_idx, n_llms)
    res = compute_benchmark_pred_r(preds, R_test, test_bench)
    summary["cdmeval_paper"] = res
    for b in BENCHMARKS:
        if b in res:
            print(f"  {b:>7}: r = {res[b]['pearson_r']:.4f}  (n={res[b]['n_items']})",
                    flush=True)
    print(f"  overall: r = {res['overall_concatenated']['pearson_r']:.4f}\n",
            flush=True)

    # ── CDMEval multi-seed (5 seeds) ──
    print("[CDMEval multi-seed (5 seeds)]", flush=True)
    seed_results = []
    for s in [42, 43, 44, 45, 46]:
        ck = MULTISEED_DIR / f"text_conditioned_seed_{s}.pt"
        if not ck.exists():
            continue
        preds = cdmeval_inference(ck, emb, q_matrix, test_idx, n_llms)
        seed_results.append(compute_benchmark_pred_r(preds, R_test, test_bench))
    if seed_results:
        for b in BENCHMARKS:
            rs = [s[b]["pearson_r"] for s in seed_results if b in s]
            print(f"  {b:>7}: {np.mean(rs):.4f} ± {np.std(rs):.4f}", flush=True)
        rs_overall = [s["overall_concatenated"]["pearson_r"] for s in seed_results]
        print(f"  overall: {np.mean(rs_overall):.4f} ± {np.std(rs_overall):.4f}\n",
                flush=True)
        summary["cdmeval_multiseed"] = {
            "per_bench": {b: {"mean": float(np.mean([s[b]["pearson_r"] for s in seed_results if b in s])),
                                 "std": float(np.std([s[b]["pearson_r"] for s in seed_results if b in s]))}
                             for b in BENCHMARKS},
            "overall_mean": float(np.mean(rs_overall)),
            "overall_std":  float(np.std(rs_overall)),
        }

    # ── KNN K=10 ──
    print("[KNN K=10]", flush=True)
    preds = knn_inference(emb, R, train_idx, test_idx, K=10)
    res = compute_benchmark_pred_r(preds, R_test, test_bench)
    summary["knn_k10"] = res
    for b in BENCHMARKS:
        if b in res:
            print(f"  {b:>7}: r = {res[b]['pearson_r']:.4f}", flush=True)
    print(f"  overall: r = {res['overall_concatenated']['pearson_r']:.4f}\n",
            flush=True)

    out_path = EXP / "v2_benchmark_prediction.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
