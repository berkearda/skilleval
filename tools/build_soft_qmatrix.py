"""Build continuous (soft) Q-matrices via per-phrase soft assignment to
cluster centroids at multiple softmax temperatures.

Hypothesis under test: does relaxing the discreteness of Q close part of
the EmbedLLM AUC gap (3.08 points)?

Steps:
    1. Compute cluster centroids in phrase-embedding space (mean of the
       SBERT phrase embeddings per cluster).
    2. Re-embed raw item phrases with all-mpnet-base-v2 (cached if the
       output file already exists).
    3. For each tau in [0.05, 0.1, 0.3, 1.0, 3.0]:
       per-phrase softmax(cos_sim/tau) over centroids -> per-item average.
    4. Verify each soft Q (shape, NaN, row-sum, peak distribution,
       Frobenius distance to binary baseline).

All outputs land in cdm_exploration/data/cdm_ready/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


DATA_DIR = Path("cdm_exploration/data/cdm_ready")
TAUS = [0.05, 0.1, 0.3, 1.0, 3.0]
SBERT_MODEL = "all-mpnet-base-v2"


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def compute_centroids(skill_emb_path: Path, K: int, out_path: Path) -> np.ndarray:
    print(f"\n[1] Computing centroids from {skill_emb_path.name} (K={K})...",
          flush=True)
    data = np.load(skill_emb_path, allow_pickle=True)
    embs = data["embeddings"].astype(np.float32, copy=False)
    labels = data["labels"]
    if embs.ndim != 2 or embs.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"FAIL: skill_embeddings shape mismatch "
            f"emb={embs.shape} labels={labels.shape}"
        )

    centroids = np.zeros((K, embs.shape[1]), dtype=np.float32)
    sizes = np.zeros(K, dtype=np.int64)
    flagged = []
    for k in range(K):
        mask = labels == k
        n = int(mask.sum())
        sizes[k] = n
        if n == 0:
            raise SystemExit(
                f"FAIL: cluster {k} has 0 phrases -- centroid undefined. "
                f"Refusing to silently substitute zeros (no hardcoded fallbacks)."
            )
        if n < 2:
            flagged.append(k)
        centroids[k] = embs[mask].mean(axis=0)

    print(f"    cluster size: min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f} median={np.median(sizes):.1f}", flush=True)
    if flagged:
        print(f"    WARN (F1): {len(flagged)} clusters have <2 phrases: "
              f"{flagged[:20]}{'...' if len(flagged) > 20 else ''}",
              flush=True)
    else:
        print("    All clusters have >=2 phrases.", flush=True)

    if np.isnan(centroids).any():
        raise SystemExit("FAIL: NaN in centroids")
    np.save(out_path, centroids)
    print(f"    saved -> {out_path}  shape={centroids.shape} "
          f"dtype={centroids.dtype}", flush=True)
    return centroids


def embed_raw_phrases(
    skills_json: Path,
    out_path: Path,
    item_text_emb_path: Path,
    n_items: int,
) -> dict:
    if out_path.exists():
        print(f"\n[2] Phrase embeddings cached -> {out_path}", flush=True)
        cached = np.load(out_path, allow_pickle=True)
        return {
            "embeddings": cached["embeddings"],
            "item_idx": cached["item_idx"],
            "phrase_text": cached["phrase_text"],
        }

    print(f"\n[2] Loading raw skills from {skills_json.name}...", flush=True)
    with open(skills_json) as f:
        items = json.load(f)
    phrases, item_ids = [], []
    items_with = set()
    for it in items:
        i = int(it["item_idx"])
        for s in it.get("skills", []):
            if isinstance(s, str) and s.strip():
                phrases.append(s.strip())
                item_ids.append(i)
                items_with.add(i)
    if not phrases:
        raise SystemExit("FAIL: no raw phrases collected")
    print(f"    collected {len(phrases):,} raw phrases across "
          f"{len(items_with):,} items "
          f"({n_items - len(items_with)} items have empty skills)",
          flush=True)

    print(f"    loading SBERT '{SBERT_MODEL}' ...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    t0 = time.time()
    embs = model.encode(
        phrases,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=False,  # we normalize separately for cosine
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"    encoded in {time.time() - t0:.1f}s -> shape {embs.shape}",
          flush=True)
    if np.isnan(embs).any():
        raise SystemExit("FAIL: NaN in raw phrase embeddings")

    # Backstop: items with empty skills (parse errors upstream) have no
    # phrase to soft-assign. Use the cached item-text embedding as a
    # single proxy phrase. This mirrors the binary Q's fix_zero_skill_items
    # logic (nearest-centroid-from-text) but in soft form. Without this,
    # the soft Q would have zero rows -- a hardcoded fallback we forbid.
    missing = sorted(set(range(n_items)) - items_with)
    if missing:
        if not item_text_emb_path.exists():
            raise SystemExit(
                f"FAIL: {len(missing)} items have empty skills and "
                f"item_text_embeddings_v2_full.npz is missing. Cannot "
                f"backfill without re-running upstream skill extraction."
            )
        text_embs = np.load(item_text_emb_path)["embeddings"].astype(
            np.float32, copy=False
        )
        if text_embs.shape[0] != n_items:
            raise SystemExit(
                f"FAIL: text_embs.shape[0]={text_embs.shape[0]} "
                f"!= n_items={n_items}"
            )
        backfill = text_embs[missing]
        embs = np.concatenate([embs, backfill], axis=0)
        item_ids.extend(missing)
        phrases.extend([f"<item_text_proxy:{i}>" for i in missing])
        print(
            f"    backfilled {len(missing)} empty-skill items with their "
            f"item-text embedding as a single proxy phrase "
            f"(F1 mitigation, mirrors binary fix_zero_skill_items)",
            flush=True,
        )
    item_ids = np.asarray(item_ids, dtype=np.int64)
    phrase_text = np.asarray(phrases, dtype=object)

    np.savez(
        out_path,
        embeddings=embs,
        item_idx=item_ids,
        phrase_text=phrase_text,
    )
    print(f"    saved -> {out_path}  total phrases (incl. backfill)="
          f"{len(item_ids):,}", flush=True)
    return {"embeddings": embs, "item_idx": item_ids, "phrase_text": phrase_text}


def build_soft_q(
    phrase_embs: np.ndarray,
    item_ids: np.ndarray,
    centroids: np.ndarray,
    n_items: int,
    tau: float,
) -> np.ndarray:
    K = centroids.shape[0]
    pe_n = _l2_normalize(phrase_embs)
    ce_n = _l2_normalize(centroids)
    sims = pe_n @ ce_n.T  # (N_phrases, K)
    logits = sims / float(tau)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    probs = e / e.sum(axis=1, keepdims=True)  # (N_phrases, K)

    Q = np.zeros((n_items, K), dtype=np.float32)
    counts = np.zeros(n_items, dtype=np.int64)
    np.add.at(Q, item_ids, probs.astype(np.float32, copy=False))
    np.add.at(counts, item_ids, 1)

    if (counts == 0).any():
        zero_items = int((counts == 0).sum())
        raise SystemExit(
            f"FAIL: {zero_items} items have no raw phrases -- refusing "
            f"to produce zero rows in soft Q (no hardcoded fallbacks)."
        )
    Q /= counts[:, None].astype(np.float32)
    return Q


def verify_soft_q(Q: np.ndarray, tau: float, Q_bin: np.ndarray) -> dict:
    n_items, K = Q.shape
    out = {"tau": tau, "shape": list(Q.shape), "dtype": str(Q.dtype)}
    if Q.dtype != np.float32:
        raise SystemExit(f"FAIL: tau={tau} dtype={Q.dtype}, expected float32")
    if np.isnan(Q).any():
        raise SystemExit(f"FAIL (F14): tau={tau} has NaN entries")
    sums = Q.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        bad = int((np.abs(sums - 1.0) > 1e-5).sum())
        raise SystemExit(
            f"FAIL: tau={tau} {bad} rows do not sum to 1 (max dev "
            f"{np.max(np.abs(sums - 1.0)):.3e})"
        )
    row_max = Q.max(axis=1)
    out["row_max_mean"] = float(row_max.mean())
    out["row_max_median"] = float(np.median(row_max))
    out["row_max_min"] = float(row_max.min())
    out["row_max_max"] = float(row_max.max())
    out["frac_row_max_gt_0.5"] = float((row_max > 0.5).mean())
    out["frac_row_max_lt_0.2"] = float((row_max < 0.2).mean())
    out["entropy_bits_mean"] = float(
        -(Q * np.log2(np.clip(Q, 1e-12, 1.0))).sum(axis=1).mean()
    )

    fro = float(np.linalg.norm(Q - Q_bin.astype(np.float32)))
    out["frobenius_to_binary"] = fro
    return out


def sanity_top_clusters(
    Q: np.ndarray, Q_bin: np.ndarray, items_data: list, sample_idx: list
) -> None:
    print("    Top-3 clusters per soft Q at tau=0.3 (sample of 5 items):",
          flush=True)
    for i in sample_idx:
        top3 = np.argsort(-Q[i])[:3]
        bin_active = np.where(Q_bin[i] > 0)[0].tolist()
        bench = items_data[i].get("benchmark", "?")
        print(
            f"      item {i:5d} [{bench}] binary={bin_active} "
            f"soft_top3={top3.tolist()} "
            f"(probs={[round(float(Q[i, k]), 3) for k in top3]})",
            flush=True,
        )


def main() -> None:
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Soft Q-matrix construction (5 temperatures)", flush=True)
    print("=" * 70, flush=True)

    # B2 input contracts
    needed = {
        "qmatrix": DATA_DIR / "qmatrix_v2_K100.npy",
        "skill_embeddings": DATA_DIR / "skill_embeddings_v2_K100.npz",
        "skills_json": DATA_DIR / "skills_extracted_v2_full.json",
    }
    for name, p in needed.items():
        if not p.exists():
            raise SystemExit(f"FAIL: missing input {name} at {p}")

    Q_bin = np.load(needed["qmatrix"])
    n_items, K = Q_bin.shape
    print(f"\nBinary Q-matrix: shape={Q_bin.shape} dtype={Q_bin.dtype} "
          f"sum={Q_bin.sum()}", flush=True)
    if K != 100:
        raise SystemExit(f"FAIL: expected K=100, got K={K}")

    centroids_path = DATA_DIR / "cluster_centroids_v2_K100.npy"
    centroids = compute_centroids(needed["skill_embeddings"], K, centroids_path)

    raw_path = DATA_DIR / "raw_phrase_embeddings_v2.npz"
    item_text_emb_path = DATA_DIR / "item_text_embeddings_v2_full.npz"
    phrase_data = embed_raw_phrases(
        needed["skills_json"], raw_path, item_text_emb_path, n_items
    )
    phrase_embs = phrase_data["embeddings"]
    item_ids = phrase_data["item_idx"]
    if int(item_ids.max()) >= n_items:
        raise SystemExit(
            f"FAIL: phrase item_idx max={item_ids.max()} >= n_items={n_items}"
        )

    items_data = json.load(open(needed["skills_json"]))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_items, size=5, replace=False).tolist()

    print(f"\n[3] Building soft Q at tau in {TAUS}...", flush=True)
    diagnostics = []
    for tau in TAUS:
        t0 = time.time()
        Q = build_soft_q(phrase_embs, item_ids, centroids, n_items, tau)
        out = DATA_DIR / f"qmatrix_v2_K100_soft_tau{tau}.npy"
        np.save(out, Q)
        info = verify_soft_q(Q, tau, Q_bin)
        info["path"] = str(out)
        info["wall_seconds"] = float(time.time() - t0)
        diagnostics.append(info)
        print(
            f"  tau={tau:>5}: row_max mean={info['row_max_mean']:.3f} "
            f"frac>.5={info['frac_row_max_gt_0.5']:.2f} "
            f"frac<.2={info['frac_row_max_lt_0.2']:.2f} "
            f"H_bits={info['entropy_bits_mean']:.2f} "
            f"||Q-Q_bin||_F={info['frobenius_to_binary']:.1f} "
            f"({info['wall_seconds']:.1f}s) -> {out.name}",
            flush=True,
        )
        if tau == 0.3:
            sanity_top_clusters(Q, Q_bin, items_data, sample_idx)

    # Monotonic Frobenius growth check
    fros = [d["frobenius_to_binary"] for d in diagnostics]
    if not all(fros[i] <= fros[i + 1] + 1e-3 for i in range(len(fros) - 1)):
        print(f"  WARN: Frobenius distance not monotonic in tau: {fros}",
              flush=True)
    else:
        print(f"  OK: Frobenius distance monotonic in tau: "
              f"{[round(f, 1) for f in fros]}", flush=True)

    # Peakiness check at low tau (B4 sanity)
    low = diagnostics[0]
    if low["frac_row_max_gt_0.5"] < 0.5:
        print(
            f"  WARN: at tau={low['tau']} only "
            f"{low['frac_row_max_gt_0.5']:.2%} of rows have max>.5"
            f" (expected >.5 for most items)",
            flush=True,
        )
    high = diagnostics[-1]
    if high["frac_row_max_lt_0.2"] < 0.5:
        print(
            f"  WARN: at tau={high['tau']} only "
            f"{high['frac_row_max_lt_0.2']:.2%} of rows have max<.2"
            f" (expected <.2 for most items)",
            flush=True,
        )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_items": int(n_items),
        "K": int(K),
        "taus": TAUS,
        "centroids_path": str(centroids_path),
        "raw_phrase_embeddings_path": str(raw_path),
        "binary_qmatrix_path": str(needed["qmatrix"]),
        "binary_qmatrix_norm_F": float(np.linalg.norm(Q_bin.astype(np.float32))),
        "diagnostics": diagnostics,
        "wall_seconds_total": float(time.time() - t_start),
    }
    summary_path = DATA_DIR / "qmatrix_v2_K100_soft_build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[4] Summary written -> {summary_path}", flush=True)
    print(f"Total wall: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
