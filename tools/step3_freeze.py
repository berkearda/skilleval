#!/usr/bin/env python3
"""Step 3: freeze codebook v1 in the format the design doc specifies.

The doc's entry format is id / name / parent / definition / include / exclude /
exemplars / confusable_with. Step 2 produced everything except `parent` and
`confusable_with`, so this step adds them and emits the frozen artifact.

`parent` is the coarse level Step 0 asked for (10-20 domains). It is assigned
here rather than during Step 2 because codes were still merging and being
renamed then, so any parent assigned earlier would have had to be redone.

`confusable_with` records each code's nearest neighbours by definition
embedding. The doc wants a written discriminating rule per pair; that is left
to Step 6's distinctness check, which is where the doc puts rule-writing, and
where the pairs that actually get confused are known from co-assignment rather
than guessed from cosine.

    python3 tools/step3_freeze.py [--domains 15] [--smoke]
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, CODEBOOK

P = Path(__file__).resolve().parent.parent / "cdm_exploration/experiments/pipeline_v7"
CONF_THR = 0.85          # neighbours at or above this are recorded as confusable
CONF_KEEP = 3

DOMAIN_SYS = ("You organise a skill taxonomy. You name groups of cognitive skills by the "
              "kind of operation they share, never by the subject matter they appear in.")
DOMAIN_U = """Below are {n} groups of cognitive skills. For each group you see its most
frequent member skills.

{groups}

Give each group a short domain name: 2-4 words, lowercase, describing the kind of
operation its members share. Return JSON only:
{{"domains": [{{"group": 0, "name": "..."}}, ...]}}"""


def resolve(cb, cid):
    seen = set()
    while cid in cb["alias"] and cid not in seen:
        seen.add(cid); cid = cb["alias"][cid]
    return cid


def load():
    cb = json.loads((P / "codebook_final.json").read_text())
    live = [c for c in cb["codes"] if c not in cb["alias"]]
    uses = Counter()
    for lab, cid in cb["assign"].items():
        uses[resolve(cb, cid)] += 1
    return cb, live, uses


def embed_defs(g, cb, live):
    cache = P / "code_def_emb.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == live:
            print(f"  loaded {len(live):,} cached definition embeddings")
            return z["vecs"]
    print(f"  embedding {len(live):,} code definitions ...")
    V = np.stack(g.embed([f"{cb['codes'][c]['name']}. {cb['codes'][c]['definition']}" for c in live]))
    np.savez_compressed(cache, ids=np.array(live, dtype=object), vecs=V)
    return V


def kmeans(V, k, iters=60, seed=42):
    rng = np.random.default_rng(seed)
    C = V[rng.choice(len(V), k, replace=False)]
    for _ in range(iters):
        a = (V @ C.T).argmax(1)
        for j in range(k):
            m = V[a == j]
            if len(m):
                C[j] = m.mean(0) / max(np.linalg.norm(m.mean(0)), 1e-9)
    return (V @ C.T).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cb, live, uses = load()
    print(f"step 3: freezing {len(live):,} live codes ({len(cb['alias'])} merged away)")
    g = Gemini()
    V = embed_defs(g, cb, live)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)

    # ---- coarse level (the doc's `parent`, Step 0's 10-20 domains) ----
    lab = kmeans(Vn, a.domains)
    groups = []
    for j in range(a.domains):
        mem = [live[i] for i in range(len(live)) if lab[i] == j]
        top = sorted(mem, key=lambda c: -uses[c])[:12]
        groups.append(f"GROUP {j} ({len(mem)} skills): " +
                      "; ".join(cb["codes"][c]["name"] for c in top))
    names = {}
    if not a.smoke:
        try:
            obj = g.json_obj(DOMAIN_SYS, DOMAIN_U.format(n=a.domains, groups="\n\n".join(groups)),
                             model=CODEBOOK, max_out=8000)
            names = {int(d["group"]): d["name"].strip().lower() for d in obj.get("domains", [])}
        except (GeminiError, KeyError, ValueError) as e:
            print(f"  domain naming failed ({e}); groups keep numeric ids")
    for j in range(a.domains):
        names.setdefault(j, f"domain_{j:02d}")

    # ---- confusable_with ----
    S = Vn @ Vn.T
    np.fill_diagonal(S, -1)
    conf = {}
    for i, c in enumerate(live):
        order = np.argsort(-S[i])[:CONF_KEEP]
        conf[c] = [{"id": live[j], "name": cb["codes"][live[j]]["name"],
                    "similarity": round(float(S[i, j]), 3)}
                   for j in order if S[i, j] >= CONF_THR]

    out = {}
    for i, c in enumerate(live):
        e = cb["codes"][c]
        out[c] = {"id": c, "name": e["name"], "parent": f"D{lab[i]:02d}_{names[int(lab[i])].replace(' ', '_')}",
                  "definition": e["definition"], "include": e.get("include", []),
                  "exclude": e.get("exclude", []),
                  "exemplar_labels": e.get("exemplars", [])[:5],
                  "exemplar_items": [],          # filled at Step 4, which is where items meet codes
                  "confusable_with": conf[c], "raw_label_mentions": uses[c]}

    frozen = {"version": "v1", "source": "codebook_final.json",
              "n_codes": len(live), "n_domains": a.domains,
              "domains": {f"D{j:02d}_{names[j].replace(' ', '_')}": int((lab == j).sum())
                          for j in range(a.domains)},
              "codes": out}
    if a.smoke:
        print("SMOKE: not writing"); print(json.dumps(list(out.values())[0], indent=1)[:700]); return
    (P / "codebook_v1_frozen.json").write_text(json.dumps(frozen, indent=1))

    nconf = sum(1 for c in live if conf[c])
    print(f"\nwrote codebook_v1_frozen.json")
    print(f"  domains: {a.domains}")
    for j in range(a.domains):
        print(f"    D{j:02d} {names[j]:38s} {int((lab==j).sum()):4d} codes")
    print(f"  codes with a confusable neighbour (cos >= {CONF_THR}): {nconf} ({nconf/len(live):.0%})")
    print(f"  codes with no exemplar labels: {sum(1 for c in live if not out[c]['exemplar_labels'])}")


if __name__ == "__main__":
    main()
