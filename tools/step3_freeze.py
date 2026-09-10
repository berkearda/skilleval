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
import argparse, os, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, CODEBOOK

P = Path(__file__).resolve().parent.parent / "cdm_exploration/experiments/pipeline_v7"

# Every artifact this step reads or writes is namespaced by STEP_TAG, so a second
# run cannot overwrite the first. Steps 2, 6 and 9 already work this way; steps
# 3-5 did not, and running them untagged would have destroyed
# codebook_v1_frozen.json and item_labels.jsonl, which the 230-skill taxonomy and
# Berke's gold-set score both trace to.
TAG = os.environ.get("STEP_TAG", "")


def tagged(name):
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}{dot}{ext}"

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
    cb = json.loads((P / tagged("codebook_final.json")).read_text())
    live = [c for c in cb["codes"] if c not in cb["alias"]]
    uses = Counter()
    for lab, cid in cb["assign"].items():
        uses[resolve(cb, cid)] += 1
    return cb, live, uses


# code_text and the cache guard live in tools/codeemb so that every step
# reading code_def_emb.npz gets the staleness check, not just this one.
from tools.codeemb import code_text, load as _load_code_vecs  # noqa: E402


def embed_defs(g, cb, live):
    return _load_code_vecs(P / "code_def_emb.npz", cb, live, g=g)


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
    # exemplar_items are item ids, which only exist once Step 4 has run. On a
    # second pass they are filled from it; on the first they stay empty.
    ex_items = {}
    lf = P / tagged("item_labels.jsonl")
    if lf.exists():
        from collections import defaultdict as _dd
        acc = _dd(list)
        for l in lf.open():
            r = json.loads(l)
            if "error" in r:
                continue
            for asg in r.get("assigned", []):
                if len(acc[asg["code"]]) < 3:
                    acc[asg["code"]].append(r["item_idx"])
        ex_items = dict(acc)
        print(f"  filling exemplar_items from {lf.name}: {len(ex_items):,} codes have items")
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
    rules = {}
    # Namespaced: code ids restart at c_0001 in every run, so ids COLLIDE across
    # runs while meaning different skills. Reading an untagged validation file
    # here attached the previous run's discriminating rules to this run's codes:
    # c_0019 was "calculating temporal offsets" then and "solving quadratic
    # equations" now, and the rule would have gone into Step 4's prompt as
    # guidance about the wrong skill.
    df = P / tagged("validation_distinctness.json")
    if df.exists():
        for r in json.loads(df.read_text()).get("results", []):
            if r.get("rule") and not r.get("merge") and len(r.get("pair", [])) == 2:
                rules[tuple(sorted(r["pair"]))] = r["rule"]
        print(f"  loaded {len(rules):,} discriminating rules from validation_distinctness.json")
    S = Vn @ Vn.T
    np.fill_diagonal(S, -1)
    conf = {}
    for i, c in enumerate(live):
        order = np.argsort(-S[i])[:CONF_KEEP]
        conf[c] = [{"id": live[j], "name": cb["codes"][live[j]]["name"],
                    "similarity": round(float(S[i, j]), 3),
                    "rule": rules.get(tuple(sorted((c, live[j]))))}
                   for j in order if S[i, j] >= CONF_THR]

    out = {}
    for i, c in enumerate(live):
        e = cb["codes"][c]
        out[c] = {"id": c, "name": e["name"], "parent": f"D{lab[i]:02d}_{names[int(lab[i])].replace(' ', '_')}",
                  "definition": e["definition"], "include": e.get("include", []),
                  "exclude": e.get("exclude", []),
                  "exemplar_labels": e.get("exemplars", [])[:5],
                  "exemplar_items": ex_items.get(c, []),
                  "confusable_with": conf[c], "raw_label_mentions": uses[c]}

    frozen = {"version": "v1", "source": tagged("codebook_final.json"),
              "n_codes": len(live), "n_domains": a.domains,
              "domains": {f"D{j:02d}_{names[j].replace(' ', '_')}": int((lab == j).sum())
                          for j in range(a.domains)},
              "codes": out}
    if a.smoke:
        print("SMOKE: not writing"); print(json.dumps(list(out.values())[0], indent=1)[:700]); return
    (P / tagged("codebook_v1_frozen.json")).write_text(json.dumps(frozen, indent=1))

    nconf = sum(1 for c in live if conf[c])
    print(f"\nwrote {tagged('codebook_v1_frozen.json')}")
    print(f"  domains: {a.domains}")
    for j in range(a.domains):
        print(f"    D{j:02d} {names[j]:38s} {int((lab==j).sum()):4d} codes")
    print(f"  codes with a confusable neighbour (cos >= {CONF_THR}): {nconf} ({nconf/len(live):.0%})")
    print(f"  codes with no exemplar labels: {sum(1 for c in live if not out[c]['exemplar_labels'])}")


if __name__ == "__main__":
    main()
