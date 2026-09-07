"""CD-CAT gradient proxy validation: compare Q-row proxy to exact autograd.

The operational CD-CAT v3 uses g_j ≈ q_j * e_j as a fast proxy for the true
gradient dp_j/dtheta_raw, which actually flows through sigmoid → PosLinear1 →
sigmoid → PosLinear2 → sigmoid → PosLinear3 → sigmoid. This script quantifies
the approximation error by comparing exact autograd gradients against the proxy
on 100 items and a sample of LLMs.

Metrics reported:
  - Per-item cosine similarity(proxy, exact) averaged over LLMs
  - Per-item magnitude ratio ||proxy|| / ||exact||
  - Spearman rank correlation of A-optimality scores computed with proxy vs
    exact over the 100 candidate items

Usage:
    python tools/run_cdcat_gradient_validation.py device=cpu
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def forward_single(net, theta_raw_m, item_id, q_matrix, text_embs, device):
    """Forward pass for one LLM, one item. Returns scalar prediction.

    theta_raw_m: (K,) tensor (requires_grad).
    """
    idx = int(item_id)
    te = torch.tensor(text_embs[idx:idx + 1], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx:idx + 1], dtype=torch.float32, device=device)

    with torch.no_grad():
        k_d = torch.sigmoid(net.k_difficulty_proj(te))  # (1, K)
        e_d = torch.sigmoid(net.e_difficulty_proj(te))  # (1, 1)

    stat = torch.sigmoid(theta_raw_m).unsqueeze(0)  # (1, K)
    x = e_d * (stat - k_d) * qr  # (1, K)
    h1 = torch.sigmoid(net.prednet_full1(x))
    h2 = torch.sigmoid(net.prednet_full2(h1))
    pred = torch.sigmoid(net.prednet_full3(h2)).squeeze()  # scalar
    return pred


def exact_gradient(net, theta_raw_m, item_id, q_matrix, text_embs, device):
    """Autograd gradient of p_j w.r.t. theta_raw_m (K-dim)."""
    theta_param = nn.Parameter(theta_raw_m.detach().clone())
    p = forward_single(net, theta_param, item_id, q_matrix, text_embs, device)
    g = torch.autograd.grad(p, theta_param)[0]
    return g.detach().cpu().numpy()


def proxy_gradient(net, item_id, q_matrix, text_embs, device):
    """Q-row proxy: g_j ≈ q_j * e_j (ignores PosLinear layers and sigmoid')."""
    idx = int(item_id)
    te = torch.tensor(text_embs[idx:idx + 1], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx:idx + 1], dtype=torch.float32, device=device)
    with torch.no_grad():
        e_d = torch.sigmoid(net.e_difficulty_proj(te))  # (1, 1)
        g = (qr * e_d).squeeze(0).cpu().numpy()  # (K,)
    return g


def proxy_gradient_corrected(net, theta_raw_m, item_id, q_matrix, text_embs, device):
    """Improved proxy: g_j ≈ q_j * e_j * sigmoid'(theta_raw).

    Still ignores PosLinear layers but restores the sigmoid derivative factor
    that dampens gradients on saturated skills. Free to compute since theta is
    already available.
    """
    idx = int(item_id)
    te = torch.tensor(text_embs[idx:idx + 1], dtype=torch.float32, device=device)
    qr = torch.tensor(q_matrix[idx:idx + 1], dtype=torch.float32, device=device)
    with torch.no_grad():
        e_d = torch.sigmoid(net.e_difficulty_proj(te))  # (1, 1)
        sig = torch.sigmoid(theta_raw_m)
        sig_deriv = sig * (1 - sig)  # (K,)
        g = (qr * e_d).squeeze(0) * sig_deriv
        g = g.cpu().numpy()  # (K,)
    return g


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # Load data
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    # Load trained NCDM
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, "cpu",
    )
    net = net.to(device)
    net.eval()

    # Sample items and LLMs
    rng = np.random.RandomState(42)
    n_items_sample = 100
    n_llms_sample = 50
    skills_per_item = q_matrix.sum(axis=1)
    eligible = np.where(skills_per_item >= 1)[0]
    sample_items = rng.choice(eligible, n_items_sample, replace=False)
    sample_llms = rng.choice(n_llms, n_llms_sample, replace=False)

    print(f"\nSampling {n_items_sample} items, {n_llms_sample} LLMs", flush=True)
    print(f"  Mean skills/item (sampled): "
          f"{q_matrix[sample_items].sum(axis=1).mean():.2f}", flush=True)

    # Extract theta_raw (pre-sigmoid mastery) for sampled LLMs
    with torch.no_grad():
        theta_raw_all = net.student_emb.weight.detach().clone()  # (n_llms, K)

    # ── Gradient comparison ──
    print(f"\nComputing exact vs proxy gradients...", flush=True)
    cos_basic = []
    cos_corrected = []
    mag_ratios = []
    mag_ratios_corrected = []

    # Cache basic proxy gradients (LLM-independent)
    proxy_grads = {int(j): proxy_gradient(net, j, q_matrix, text_embs, device)
                   for j in sample_items}

    for m_idx, m in enumerate(sample_llms):
        theta_m = theta_raw_all[m].to(device)
        for j in sample_items:
            g_exact = exact_gradient(net, theta_m, j, q_matrix, text_embs, device)
            g_proxy = proxy_grads[int(j)]
            g_corrected = proxy_gradient_corrected(net, theta_m, j, q_matrix,
                                                    text_embs, device)

            cos_basic.append(cosine(g_proxy, g_exact))
            cos_corrected.append(cosine(g_corrected, g_exact))
            ne = np.linalg.norm(g_exact)
            if ne > 1e-12:
                mag_ratios.append(np.linalg.norm(g_proxy) / ne)
                mag_ratios_corrected.append(np.linalg.norm(g_corrected) / ne)

        if (m_idx + 1) % 10 == 0:
            print(f"  LLM {m_idx + 1}/{n_llms_sample}", flush=True)

    cos_basic = np.array(cos_basic)
    cos_corrected = np.array(cos_corrected)
    mag_ratios = np.array(mag_ratios)
    mag_ratios_corrected = np.array(mag_ratios_corrected)

    print(f"\n{'='*60}", flush=True)
    print("GRADIENT VECTOR COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Cosine(basic proxy, exact):     "
          f"mean={cos_basic.mean():.4f}  median={np.median(cos_basic):.4f}  "
          f"min={cos_basic.min():.4f}  max={cos_basic.max():.4f}", flush=True)
    print(f"  Cosine(corrected proxy, exact): "
          f"mean={cos_corrected.mean():.4f}  median={np.median(cos_corrected):.4f}  "
          f"min={cos_corrected.min():.4f}  max={cos_corrected.max():.4f}", flush=True)
    print(f"  ||basic|| / ||exact||:    "
          f"median={np.median(mag_ratios):.4f}  "
          f"p95={np.percentile(mag_ratios, 95):.4f}", flush=True)
    print(f"  ||corrected|| / ||exact||: "
          f"median={np.median(mag_ratios_corrected):.4f}  "
          f"p95={np.percentile(mag_ratios_corrected, 95):.4f}", flush=True)

    # ── A-optimality score comparison ──
    # For each LLM m, build a "Sigma" (just identity for this test — we want
    # to know whether proxy preserves the RANKING of items by A-score, which
    # is what CD-CAT actually uses).
    print(f"\n{'='*60}", flush=True)
    print("A-OPTIMALITY SCORE RANK CORRELATION (Sigma = I)", flush=True)
    print(f"{'='*60}", flush=True)
    rank_basic = []
    rank_corrected = []
    top10_basic = []
    top10_corrected = []
    for m in sample_llms:
        theta_m = theta_raw_all[m].to(device)
        s_exact, s_basic, s_corr = [], [], []
        for j in sample_items:
            g_exact = exact_gradient(net, theta_m, j, q_matrix, text_embs, device)
            g_proxy = proxy_grads[int(j)]
            g_corr = proxy_gradient_corrected(net, theta_m, j, q_matrix,
                                               text_embs, device)
            p = forward_single(net, theta_m, j, q_matrix, text_embs, device).item()
            c = max(p * (1 - p), 1e-8)
            s_exact.append((g_exact ** 2).sum() / c)
            s_basic.append((g_proxy ** 2).sum() / c)
            s_corr.append((g_corr ** 2).sum() / c)
        s_exact = np.array(s_exact)
        s_basic = np.array(s_basic)
        s_corr = np.array(s_corr)
        rank_basic.append(spearmanr(s_basic, s_exact)[0])
        rank_corrected.append(spearmanr(s_corr, s_exact)[0])

        top_e = set(np.argsort(-s_exact)[:10])
        top10_basic.append(len(set(np.argsort(-s_basic)[:10]) & top_e) / 10.0)
        top10_corrected.append(len(set(np.argsort(-s_corr)[:10]) & top_e) / 10.0)

    rank_basic = np.array(rank_basic)
    rank_corrected = np.array(rank_corrected)
    top10_basic = np.array(top10_basic)
    top10_corrected = np.array(top10_corrected)
    print(f"  Spearman rho (basic proxy):     "
          f"mean={rank_basic.mean():.4f}  median={np.median(rank_basic):.4f}  "
          f"min={rank_basic.min():.4f}  max={rank_basic.max():.4f}", flush=True)
    print(f"  Spearman rho (corrected proxy): "
          f"mean={rank_corrected.mean():.4f}  median={np.median(rank_corrected):.4f}  "
          f"min={rank_corrected.min():.4f}  max={rank_corrected.max():.4f}", flush=True)
    print(f"  Top-10 overlap (basic):     mean={top10_basic.mean():.4f}  "
          f"median={np.median(top10_basic):.4f}", flush=True)
    print(f"  Top-10 overlap (corrected): mean={top10_corrected.mean():.4f}  "
          f"median={np.median(top10_corrected):.4f}", flush=True)

    # ── Save ──
    save_data = {
        "experiment": "cdcat_gradient_validation",
        "n_items_sample": n_items_sample,
        "n_llms_sample": n_llms_sample,
        "K": int(K),
        "gradient_vector_basic": {
            "cosine_mean": float(cos_basic.mean()),
            "cosine_median": float(np.median(cos_basic)),
            "cosine_min": float(cos_basic.min()),
            "cosine_max": float(cos_basic.max()),
            "mag_ratio_median": float(np.median(mag_ratios)),
            "mag_ratio_p95": float(np.percentile(mag_ratios, 95)),
        },
        "gradient_vector_corrected": {
            "cosine_mean": float(cos_corrected.mean()),
            "cosine_median": float(np.median(cos_corrected)),
            "cosine_min": float(cos_corrected.min()),
            "cosine_max": float(cos_corrected.max()),
            "mag_ratio_median": float(np.median(mag_ratios_corrected)),
            "mag_ratio_p95": float(np.percentile(mag_ratios_corrected, 95)),
        },
        "a_optimality_ranking_basic": {
            "spearman_mean": float(rank_basic.mean()),
            "spearman_median": float(np.median(rank_basic)),
            "spearman_min": float(rank_basic.min()),
            "spearman_max": float(rank_basic.max()),
            "top10_overlap_mean": float(top10_basic.mean()),
            "top10_overlap_median": float(np.median(top10_basic)),
        },
        "a_optimality_ranking_corrected": {
            "spearman_mean": float(rank_corrected.mean()),
            "spearman_median": float(np.median(rank_corrected)),
            "spearman_min": float(rank_corrected.min()),
            "spearman_max": float(rank_corrected.max()),
            "top10_overlap_mean": float(top10_corrected.mean()),
            "top10_overlap_median": float(np.median(top10_corrected)),
        },
    }

    out_json = Path("cdm_exploration/experiments/v2_cdcat_gradient_validation.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="cdcat_gradient_validation",
        config={"n_items": n_items_sample, "n_llms": n_llms_sample, "K": K,
                "device": device},
        results=save_data,
        split_info={"n_items_sample": n_items_sample,
                    "n_llms_sample": n_llms_sample},
        verified=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
