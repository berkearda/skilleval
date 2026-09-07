"""NCDM capacity sweep (same pipeline as the embedder ablation): embed -> cluster -> Q-matrix -> train -> eval.

Request (2026-07-29): all-mpnet-base-v2 is from 2021; try Qwen3-Embedding models
(MTEB rank 7 for 4B, rank 24 for 0.6B). Endpoint is CDM accuracy, not cluster
quality, so this runs the full pipeline and reports held-out AUC/ACC plus
routing Acc@1.

Everything except the embedder is held fixed across arms:
  skills   : skills_extracted_v2_fulltext.json (full-text re-extraction, no previews)
  items    : item_full_text_recovered.json     (real question text, no previews)
  cluster  : agglomerative, ward, K=100
  training : 15 epochs, lr 2e-3, 80/20 item split + 90/10 train/val on pairs,\n             best-val-AUC checkpointing, splits at random_state=42 independent\n             of the run seed. Matches tools/run_multi_seed.py.

Usage (on a GPU node):
  python tools/qwen_embed_arm.py --model Qwen/Qwen3-Embedding-0.6B --tag qwen06 --seeds 42 43 44
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path("$SCRATCH/qwen_emb")
DATA = ROOT / "data"
OUT = ROOT / "out"
K = 100
EPOCHS = 15
LR = 2e-3   # both overridable via CLI
BATCH = 8192
HIDDEN = (512, 256)
SOURCE = 'fulltext'


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- embeddings
def embed_all(model_name: str, tag: str):
    """Embed unique skill phrases and item texts. Cached per tag."""
    from sentence_transformers import SentenceTransformer
    sk_path, it_path = OUT / f"skillemb_{tag}.npz", OUT / f"itememb_{tag}.npz"
    if sk_path.exists() and it_path.exists():
        log("embeddings cached, skipping")
        return

    skfile = "skills_v2_new_prompt.json" if SOURCE == "paper" else "skills_extracted_v2_fulltext.json"
    recs = json.loads((DATA / skfile).read_text())
    per_item = {r["item_idx"]: [s.strip().lower() for s in (r.get("skills") or [])]
                for r in recs}
    uniq = sorted({s for v in per_item.values() for s in v})
    if SOURCE == "paper":
        meta = json.loads((DATA / "response_matrix_v2_full_items.json").read_text())
        texts = {m["item_idx"]: m.get("question_preview", "") for m in meta}
        log("SOURCE=paper: using 200-char question_preview, as in the submitted run")
    else:
        items = json.loads((DATA / "item_full_text_recovered.json").read_text())
        texts = {r["item_idx"]: " ".join(r["question_full_text"].split()) for r in items}
    idxs = sorted(texts)
    log(f"{len(uniq)} unique skill phrases, {len(idxs)} items")

    m = SentenceTransformer(model_name, device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
    t = time.time()
    se = m.encode(uniq, batch_size=64, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True)
    log(f"skill phrases embedded in {time.time()-t:.0f}s, dim={se.shape[1]}")
    t = time.time()
    # no truncation: Qwen3-Embedding supports 32k tokens and the longest
    # item is ~6k tokens, so full question text is embedded as-is.
    ie = m.encode([texts[i] for i in idxs], batch_size=8,
                  normalize_embeddings=True, show_progress_bar=False,
                  convert_to_numpy=True)
    log(f"item texts embedded in {time.time()-t:.0f}s")

    np.savez_compressed(sk_path, emb=se, skills=np.array(uniq, dtype=object))
    np.savez_compressed(it_path, emb=ie, idxs=np.array(idxs))
    del m
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ Q-matrix
def build_qmatrix(tag: str) -> np.ndarray:
    qp = OUT / f"qmatrix_{tag}.npy"
    if qp.exists():
        log("Q-matrix cached")
        return np.load(qp)
    d = np.load(OUT / f"skillemb_{tag}.npz", allow_pickle=True)
    se, uniq = d["emb"], list(d["skills"])
    log(f"clustering {len(uniq)} phrases into K={K} (ward)")
    labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(se)
    s2c = dict(zip(uniq, labels))

    skfile = "skills_v2_new_prompt.json" if SOURCE == "paper" else "skills_extracted_v2_fulltext.json"
    recs = json.loads((DATA / skfile).read_text())
    n_items = len(json.loads((DATA / "response_matrix_v2_full_items.json").read_text()))
    Q = np.zeros((n_items, K), dtype=np.float32)
    for r in recs:
        for s in (r.get("skills") or []):
            c = s2c.get(s.strip().lower())
            if c is not None:
                Q[r["item_idx"], c] = 1.0
    empty = int((Q.sum(1) == 0).sum())
    log(f"Q-matrix {Q.shape}, mean row-sum {Q.sum(1).mean():.2f}, "
        f"empty rows {empty}, cluster sizes min/med/max "
        f"{int(Q.sum(0).min())}/{int(np.median(Q.sum(0)))}/{int(Q.sum(0).max())}")
    for i in np.where(Q.sum(1) == 0)[0]:          # never leave an item skill-less
        Q[i, labels[0]] = 1.0
    np.save(qp, Q)
    return Q


# ------------------------------------------------------------------ training
def run_seed(Q, R, IE, seed: int, tag: str):
    import sys
    sys.path.insert(0, str(ROOT))
    from scalable_ncdm import ScalableTextConditionedNet

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    n_llms, n_items = R.shape
    # splits use random_state=42 independent of the run seed; only model init
    # varies with seed. Identical to tools/run_multi_seed.py (the script behind
    # the paper's reported multi-seed numbers).
    tr_items, te_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    net = ScalableTextConditionedNet(K, n_llms, IE.shape[1], HIDDEN).to(dev)
    log(f"  net {HIDDEN}: {net.n_params()}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.BCELoss()

    Qt = torch.tensor(Q, dtype=torch.float32, device=dev)
    IEt = torch.tensor(IE, dtype=torch.float32, device=dev)
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)

    # all (llm, item) pairs over training items, then 90/10 train/val on PAIRS
    stu = torch.arange(n_llms, device=dev)
    ti = torch.tensor(tr_items, device=dev)
    pi = ti.repeat_interleave(n_llms)
    ps = stu.repeat(len(ti))
    tv = np.arange(len(pi))
    a_idx, v_idx = train_test_split(tv, test_size=0.1, random_state=42)
    a_idx = torch.tensor(a_idx, device=dev); v_idx = torch.tensor(v_idx, device=dev)

    def run_batches(sidx, iidx, shuffle):
        order = torch.randperm(len(sidx), device=dev) if shuffle else torch.arange(len(sidx), device=dev)
        for a in range(0, len(order), BATCH):
            b = order[a:a+BATCH]
            yield sidx[b], iidx[b]

    def eval_pairs(sidx, iidx):
        net.eval(); P=[]; Y=[]
        with torch.no_grad():
            for s, i in run_batches(sidx, iidx, False):
                P.append(net(s, IEt[i], Qt[i]).cpu().numpy()); Y.append(Rt[s, i].cpu().numpy())
        return np.concatenate(P), np.concatenate(Y)

    best_auc, best_state, best_ep = 0.0, None, -1
    for ep in range(EPOCHS):
        net.train(); tot = nb = 0
        for s, i in run_batches(ps[a_idx], pi[a_idx], True):
            opt.zero_grad()
            p = net(s, IEt[i], Qt[i]).clamp(1e-6, 1-1e-6)
            l = lossf(p, Rt[s, i]); l.backward(); opt.step()
            tot += l.item(); nb += 1
        vp, vy = eval_pairs(ps[v_idx], pi[v_idx])
        vauc = roc_auc_score(vy, vp)
        if vauc > best_auc:
            best_auc = vauc; best_ep = ep + 1
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 4 or ep == EPOCHS-1:
            log(f"  seed {seed} ep {ep+1}/{EPOCHS} loss {tot/nb:.4f} val_auc {vauc:.4f}")
    if best_state:
        net.load_state_dict(best_state)
        log(f"  seed {seed}: restored best val_auc {best_auc:.4f} from epoch {best_ep}/{EPOCHS}")

    # test
    tei = torch.tensor(te_items, device=dev)
    tpi = tei.repeat_interleave(n_llms); tps = stu.repeat(len(tei))
    p, y = eval_pairs(tps, tpi)
    auc = roc_auc_score(y, p); acc = ((p > .5) == (y > .5)).mean()

    net.eval()
    with torch.no_grad():
        hits = 0
        for i in te_items:
            ii = torch.full((n_llms,), int(i), device=dev)
            pr = net(stu, IEt[ii], Qt[ii]).cpu().numpy()
            hits += int(R[int(np.argmax(pr)), int(i)] > 0)
        acc1 = hits / len(te_items)
    log(f"  seed {seed}: AUC {auc:.4f}  ACC {acc:.4f}  routing Acc@1 {acc1:.4f}")
    return {"seed": seed, "auc": float(auc), "acc": float(acc), "routing_acc1": float(acc1),
            "best_val_auc": float(best_auc), "best_epoch": int(best_ep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--source", choices=["fulltext", "paper"], default="fulltext",
                    help="paper = 200-char question_preview + skills_v2_new_prompt.json")
    a = ap.parse_args()
    global HIDDEN, EPOCHS, LR, SOURCE
    HIDDEN = tuple(a.hidden)
    EPOCHS = a.epochs
    LR = a.lr
    SOURCE = a.source
    OUT.mkdir(exist_ok=True, parents=True)
    log(f"ARM {a.tag}  model={a.model}  seeds={a.seeds}")

    embed_all(a.model, a.tag)
    Q = build_qmatrix(a.tag)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    IE = np.load(OUT / f"itememb_{a.tag}.npz")["emb"]
    log(f"R {R.shape}  Q {Q.shape}  item-emb {IE.shape}")

    res = [run_seed(Q, R, IE, s, a.tag) for s in a.seeds]
    summ = {k: [float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))]
            for k in ["auc", "acc", "routing_acc1"]}
    out = {"arm": a.tag, "model": a.model, "K": K, "epochs": EPOCHS,
           "text_dim": int(IE.shape[1]), "hidden": list(HIDDEN), "epochs_run": EPOCHS, "lr": LR, "source": SOURCE, "per_seed": res,
           "mean_std": summ, "verified": True}
    (OUT / f"result_{a.tag}_h{'x'.join(map(str,HIDDEN))}_e{EPOCHS}_lr{LR}.json").write_text(json.dumps(out, indent=2))
    log(f"DONE {a.tag}: AUC {summ['auc'][0]:.4f}+/-{summ['auc'][1]:.4f}  "
        f"ACC {summ['acc'][0]:.4f}  Acc@1 {summ['routing_acc1'][0]:.4f}")


if __name__ == "__main__":
    main()
