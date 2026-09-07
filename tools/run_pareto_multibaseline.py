"""Apples-to-apples Pareto curves for CDMEval, IrtNet d=100/d=232, KNN K=10.

Each method emits per-(m, p) probability of correctness. Then every method
is wrapped in the SAME cheapest-above-threshold router policy:

  for each test item:
    LLMs with pred > tau  ->  pick the CHEAPEST by FLOPs
    none above tau        ->  fall back to argmax(pred)

Sweep tau in [0.10, 0.95] step 0.05; emit (mean_cost, accuracy) per tau.
This is the same protocol as `tools/pareto_routing_v2.py:threshold_sweep`,
just applied to four methods on the same axes.

T-032 / the project log 2026-04-30. No single-LLM anchors per user instruction.

Usage:
    python tools/run_pareto_multibaseline.py [--smoke]

Smoke mode: 100 test items, 5 tau values (0.3, 0.5, 0.7, 0.9, 0.95) for B1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
CDM_CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
IRTNET_CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "irtnet"
EMBEDLLM_CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "embedllm"
EMBEDLLM_REPO = REPO / "cdm_exploration" / "repos" / "EmbedLLM" / "algorithm"


def threshold_sweep(preds: np.ndarray, R_test: np.ndarray, flops: np.ndarray,
                     thresholds: list[float]) -> list[dict]:
    """Cheapest-above-threshold routing protocol.

    preds: (n_llms, n_test) probability of correct.
    R_test: (n_llms, n_test) ground truth 0/1.
    flops: (n_llms,) cost in GFLOPs per LLM.
    thresholds: list of tau values.

    Returns list of {threshold, accuracy, mean_cost, n_unique_picks,
    fallback_rate} per tau.
    """
    n_llms, n_test = preds.shape
    out = []
    for t in thresholds:
        correct = 0
        total_cost = 0.0
        picks = []
        n_fallback = 0
        for j in range(n_test):
            p = preds[:, j]
            above = np.where(p > t)[0]
            if len(above) > 0:
                chosen = above[np.argmin(flops[above])]
            else:
                chosen = int(np.argmax(p))
                n_fallback += 1
            picks.append(chosen)
            total_cost += flops[chosen]
            if R_test[chosen, j] > 0:
                correct += 1
        out.append({
            "threshold": float(t),
            "accuracy": correct / n_test,
            "mean_cost": total_cost / n_test,
            "n_unique_picks": int(len(np.unique(picks))),
            "fallback_rate": n_fallback / n_test,
        })
    return out


def cdmeval_predictions(test_idx: np.ndarray, n_llms: int, n_skills: int,
                          emb: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    ckpt = torch.load(CDM_CKPT, map_location="cpu", weights_only=False)
    net = TextConditionedNet(n_skills, n_llms, text_dim=768)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32)
    test_q = torch.tensor(q_matrix[test_idx], dtype=torch.float32)
    all_ids = torch.arange(n_llms)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            te = test_emb[j].unsqueeze(0).expand(n_llms, -1)
            tq = test_q[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_ids, te, tq).numpy()
    return preds


def irtnet_predictions(d_model: int, seed: int, test_idx: np.ndarray,
                        n_llms: int, n_items: int, emb: np.ndarray,
                        device: str) -> np.ndarray:
    from tools.run_irtnet_headtohead import import_irtnet_model_class
    MoEClassifier = import_irtnet_model_class()
    ck = IRTNET_CKPT_DIR / f"d{d_model}_s{seed}.pt"
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_hp = {
        "model_embed_dim": cfg.get("model_embed_dim", d_model),
        "num_experts": cfg.get("num_experts", 39),
        "top_k_experts": cfg.get("top_k_experts", 39),
        "expert_hidden_dim": cfg.get("expert_hidden_dim", 512),
        "shared_expert_hidden_dim": cfg.get("shared_expert_hidden_dim", 512),
        "expert_output_dim": cfg.get("expert_output_dim", 256),
        "dropout_rate": cfg.get("dropout_rate", 0.5),
        "embedding_noise": cfg.get("embedding_noise", 0.05),
    }
    prompt_embeddings = torch.tensor(emb, dtype=torch.float32)
    model = MoEClassifier(num_models=n_llms, num_prompts=n_items,
                            prompt_embeddings=prompt_embeddings, **model_hp)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    all_ids = torch.arange(n_llms, device=device)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j, item_idx in enumerate(test_idx):
            p_id = torch.full((n_llms,), int(item_idx), dtype=torch.long,
                                device=device)
            preds[:, j] = model(all_ids, p_id).cpu().numpy()
    # IrtNet's forward returns LOGITS (range ~[-6, 5] for our checkpoints).
    # For argmax routing it doesn't matter (sigmoid is monotone), but for
    # cheapest-above-threshold we need probabilities on a [0, 1] scale to
    # match CDMEval / KNN. Caught at B1 smoke 2026-04-30 (F14 sniff).
    preds = 1.0 / (1.0 + np.exp(-preds))
    return preds


def embedllm_predictions(d_model: int, seed: int, test_idx: np.ndarray,
                           n_llms: int, n_items: int, emb: np.ndarray,
                           device: str) -> np.ndarray:
    """Load EmbedLLM TextMF checkpoint, emit softmax(logits)[:, 1] per (m, p)."""
    if str(EMBEDLLM_REPO) not in sys.path:
        sys.path.insert(0, str(EMBEDLLM_REPO))
    from mf import TextMF  # type: ignore[import-not-found]
    ck = EMBEDLLM_CKPT_DIR / f"d{d_model}_s{seed}.pt"
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    question_embeddings = torch.tensor(emb, dtype=torch.float32)
    model = TextMF(question_embeddings=question_embeddings,
                    model_embedding_dim=cfg.get("d_model", d_model),
                    alpha=cfg.get("alpha", 0.05),
                    num_models=n_llms, num_prompts=n_items)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    p_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)
    chunk = 256
    with torch.no_grad():
        for i in range(0, n_llms, chunk):
            m_ids = torch.arange(i, min(i + chunk, n_llms), device=device)
            n_test = len(test_idx)
            m_grid = m_ids.unsqueeze(1).expand(-1, n_test).reshape(-1)
            p_grid = p_tensor.unsqueeze(0).expand(len(m_ids), -1).reshape(-1)
            logits = model(m_grid, p_grid, test_mode=True)  # (chunk*n_test, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds[i:i + len(m_ids)] = probs.view(len(m_ids), n_test).cpu().numpy()
    return preds


def knn_predictions(emb: np.ndarray, R: np.ndarray, train_idx: np.ndarray,
                      test_idx: np.ndarray, K: int = 10) -> np.ndarray:
    sim = cosine_similarity(emb[test_idx], emb[train_idx])  # (n_test, n_train)
    top_k_local = np.argpartition(-sim, K, axis=1)[:, :K]
    top_k_global = train_idx[top_k_local]
    n_llms = R.shape[0]
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    for j, neighbors in enumerate(top_k_global):
        preds[:, j] = R[:, neighbors].mean(axis=1)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="100 test items × 5 thresholds for B1.")
    args = ap.parse_args()

    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(42)
    device = resolve_device("cpu")
    print(f"=== pareto multibaseline (smoke={args.smoke}, device={device}) ===",
            flush=True)

    print("\n[load] data", flush=True)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(DATA / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    print(f"  R={R.shape}, K={n_skills}", flush=True)

    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    if args.smoke:
        rng = np.random.default_rng(42)
        test_idx = np.sort(rng.choice(test_idx, 100, replace=False))
        thresholds = [0.30, 0.50, 0.70, 0.90, 0.95]
    else:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05).tolist()]
    print(f"  test={len(test_idx)}, thresholds={len(thresholds)}", flush=True)

    print("\n[flops]", flush=True)
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])
    print(f"  range: {flops.min():.0f} – {flops.max():.0f} GFLOPs", flush=True)

    R_test = R[:, test_idx]

    # ── Generate predictions per method ──
    print("\n[infer] CDMEval (paper checkpoint)", flush=True)
    cdm_preds = cdmeval_predictions(test_idx, n_llms, n_skills, emb, q_matrix)
    print(f"  range: [{cdm_preds.min():.3f}, {cdm_preds.max():.3f}]", flush=True)

    print("\n[infer] IrtNet d=100 seed 42", flush=True)
    irtnet_d100 = irtnet_predictions(100, 42, test_idx, n_llms, n_items, emb, device)
    print(f"  range: [{irtnet_d100.min():.3f}, {irtnet_d100.max():.3f}]", flush=True)

    print("\n[infer] IrtNet d=232 seed 42", flush=True)
    irtnet_d232 = irtnet_predictions(232, 42, test_idx, n_llms, n_items, emb, device)
    print(f"  range: [{irtnet_d232.min():.3f}, {irtnet_d232.max():.3f}]", flush=True)

    print("\n[infer] EmbedLLM d=232 seed 42", flush=True)
    embedllm_d232 = embedllm_predictions(232, 42, test_idx, n_llms, n_items, emb, device)
    print(f"  range: [{embedllm_d232.min():.3f}, {embedllm_d232.max():.3f}]", flush=True)

    print("\n[infer] KNN K=10", flush=True)
    knn_preds = knn_predictions(emb, R, train_idx, test_idx, K=10)
    print(f"  range: [{knn_preds.min():.3f}, {knn_preds.max():.3f}]", flush=True)

    # ── Threshold sweeps ──
    print("\n[sweep] running threshold_sweep on each method", flush=True)
    cdm_sweep = threshold_sweep(cdm_preds, R_test, flops, thresholds)
    print(f"  CDMEval done", flush=True)
    irtnet_d100_sweep = threshold_sweep(irtnet_d100, R_test, flops, thresholds)
    print(f"  IrtNet d=100 done", flush=True)
    irtnet_d232_sweep = threshold_sweep(irtnet_d232, R_test, flops, thresholds)
    print(f"  IrtNet d=232 done", flush=True)
    embedllm_d232_sweep = threshold_sweep(embedllm_d232, R_test, flops, thresholds)
    print(f"  EmbedLLM d=232 done", flush=True)
    knn_sweep = threshold_sweep(knn_preds, R_test, flops, thresholds)
    print(f"  KNN done", flush=True)

    # ── Strongest reference for normalization ──
    train_acc = R[:, train_idx].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_acc = float(R[strongest_idx, test_idx].mean())
    strongest_cost = float(flops[strongest_idx])
    print(f"\n[ref] strongest = {llm_names[strongest_idx]}", flush=True)
    print(f"      acc={strongest_acc:.4f}, cost={strongest_cost:.0f} GFLOPs",
            flush=True)

    def normalize(sweep):
        return [{**p,
                  "cost_pct": p["mean_cost"] / strongest_cost * 100,
                  "acc_pct": p["accuracy"] / strongest_acc * 100}
                 for p in sweep]

    # ── Save ──
    out = {
        "experiment": "pareto_multibaseline",
        "smoke": args.smoke,
        "n_llms": n_llms,
        "n_test_items": int(len(test_idx)),
        "thresholds": thresholds,
        "strongest": {
            "name": llm_names[strongest_idx],
            "test_acc": strongest_acc,
            "cost_gflops": strongest_cost,
        },
        "methods": {
            "cdmeval": {"sweep": cdm_sweep, "normalized": normalize(cdm_sweep)},
            "irtnet_d100": {"sweep": irtnet_d100_sweep, "normalized": normalize(irtnet_d100_sweep)},
            "irtnet_d232": {"sweep": irtnet_d232_sweep, "normalized": normalize(irtnet_d232_sweep)},
            "embedllm_d232": {"sweep": embedllm_d232_sweep, "normalized": normalize(embedllm_d232_sweep)},
            "knn_k10": {"sweep": knn_sweep, "normalized": normalize(knn_sweep)},
        },
        "verified": True,
    }
    suffix = "_smoke" if args.smoke else ""
    out_path = EXP / f"v2_pareto_multibaseline{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # ── Summary table ──
    print("\n=== Summary (cost_pct, acc_pct per tau) ===", flush=True)
    header = f"  {'tau':>5} {'CDM':>15} {'IrtNet d=100':>15} {'IrtNet d=232':>15} {'EmbedLLM d=232':>17} {'KNN':>15}"
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for i, t in enumerate(thresholds):
        c = normalize(cdm_sweep)[i]
        i100 = normalize(irtnet_d100_sweep)[i]
        i232 = normalize(irtnet_d232_sweep)[i]
        e232 = normalize(embedllm_d232_sweep)[i]
        k = normalize(knn_sweep)[i]
        print(f"  {t:>5.2f} "
                f"({c['cost_pct']:>5.1f}, {c['acc_pct']:>5.1f}) "
                f"({i100['cost_pct']:>5.1f}, {i100['acc_pct']:>5.1f}) "
                f"({i232['cost_pct']:>5.1f}, {i232['acc_pct']:>5.1f}) "
                f"({e232['cost_pct']:>5.1f}, {e232['acc_pct']:>5.1f}) "
                f"({k['cost_pct']:>5.1f}, {k['acc_pct']:>5.1f})",
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
