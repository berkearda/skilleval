"""Step 4: relabel every item against the FROZEN merged bank (189 skills), using
retrieval. Each item's reasoning steps are matched to the top-K candidate skills,
each shown as NAME + FULL DEFINITION, and the LLM assigns a bank skill per step
(or NEW: via an escape hatch). Order-independent (bank is frozen), so parallel.

Also reports label CHURN versus the pre-Step-4 labels: what changed, what stayed.

    .venv312/bin/python tools/pilot_step4.py --smoke 6
    .venv312/bin/python tools/pilot_step4.py \
        --out cdm_exploration/experiments/pilot_step4_labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
MERGED = REPO / "cdm_exploration/experiments/pilot_merged_drop.json"
STEPF = D / "skills_stepwise_full.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"

SYSTEM = """You label a test item's reasoning steps using a FIXED bank of cognitive skills. For each step, pick the candidate skill whose definition best matches the operation that step requires, and reuse its EXACT name. If genuinely none of the candidates fits, output "NEW:snake_case_name" with a one-line definition (this should be rare). Judge by meaning, not wording."""

SYSTEM_NOESC = """You label a test item's reasoning steps using a FIXED, COMPLETE bank of cognitive skills. For each step, pick the candidate skill whose definition best matches the operation that step requires, and reuse its EXACT name. You MUST choose from the candidates shown; do NOT invent new skills. If a step is borderline, pick the closest candidate. Judge by meaning, not wording."""

USER = """ITEM [{bench}/{sub}]

Reasoning steps:
{steps}

Candidate skills (name: definition):
{cands}

Return JSON exactly: {{"labels": [{{"skill": "<exact candidate name OR NEW:new_name>", "definition": "<one line, ONLY for NEW>"}}]}} with one entry per step, in order."""


def load_key():
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def norm(s):
    s = re.sub(r"^new\s*:\s*", "", s.strip(), flags=re.I)
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "", s.lower().strip().replace(" ", "_"))).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no-escape", action="store_true",
                    help="force the model to pick from candidates (no NEW skills)")
    ap.add_argument("--merged", default=str(MERGED))
    ap.add_argument("--src", default=str(STEPF))
    ap.add_argument("--out")
    args = ap.parse_args()
    SYS = SYSTEM_NOESC if args.no_escape else SYSTEM

    merged = json.load(open(args.merged))
    names = [b["name"] for b in merged["bank"]]
    defn = {b["name"]: b["definition"] for b in merged["bank"]}
    old_labels = {pi["item_idx"]: list(pi["skills"]) for pi in merged["per_item"]}
    meta = {pi["item_idx"]: (pi["benchmark"], pi["subtask"]) for pi in merged["per_item"]}
    steps_of = {r["item_idx"]: [s.get("step", "") for s in r.get("reasoning_steps", [])]
                for r in json.load(open(args.src))}

    order = [i for i in old_labels if steps_of.get(i)]
    if args.smoke:
        order = order[: args.smoke]

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.embedder)
    BV = emb.encode([f"{n}: {defn[n]}" for n in names], normalize_embeddings=True)

    # precompute candidate lists per item (batched embedding of all steps)
    flat, bounds = [], {}
    for i in order:
        st = steps_of[i]
        bounds[i] = (len(flat), len(flat) + len(st))
        flat.extend(st)
    SV = emb.encode(flat, normalize_embeddings=True) if flat else np.zeros((0, BV.shape[1]))

    tasks = []
    for i in order:
        lo, hi = bounds[i]
        sims = SV[lo:hi] @ BV.T              # (steps, 189)
        score = sims.max(0)
        top = np.argsort(-score)[: args.topk]
        cands = [(names[j], defn[names[j]]) for j in top]
        tasks.append((i, steps_of[i], cands))

    client_key = load_key()
    from openai import OpenAI
    client = OpenAI(api_key=client_key)

    def relabel(task):
        i, steps, cands = task
        b, sub = meta[i]
        u = USER.format(bench=b, sub=sub,
                        steps="\n".join(f"{k+1}. {s}" for k, s in enumerate(steps)),
                        cands="\n".join(f"- {n}: {d}" for n, d in cands))
        res = {}
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": u}])
                res = json.loads(r.choices[0].message.content)
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    res = {}
        bank_names = set(names)
        out, news = [], []
        for lab in res.get("labels", []) or []:
            if not isinstance(lab, dict):
                continue
            raw = str(lab.get("skill", ""))
            nm = norm(raw)
            if not nm:
                continue
            if nm not in bank_names:
                if args.no_escape:
                    continue   # forced to bank; drop any out-of-bank label
                news.append((nm, str(lab.get("definition", "")).strip()))
            if nm not in out:
                out.append(nm)
        return i, out, news

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(relabel, tasks))

    new_labels = {i: sk for i, sk, _ in results}
    escape_new = defaultdict(str)
    for _, _, news in results:
        for nm, d in news:
            escape_new[nm] = d or escape_new[nm]

    # ---- churn vs old labels ----
    jac, identical, retained, changed_out, added, n_old, n_new = [], 0, 0, 0, 0, 0, 0
    for i in order:
        o, n = set(old_labels[i]), set(new_labels[i])
        u = o | n
        jac.append(len(o & n) / max(len(u), 1))
        identical += (o == n)
        retained += len(o & n)
        changed_out += len(o - n)
        added += len(n - o)
        n_old += len(o)
        n_new += len(n)
    used_old = len({s for i in order for s in old_labels[i]})
    used_new = len({s for i in order for s in new_labels[i]})

    if args.smoke:
        for i in order:
            print(f"\n{meta[i]} item {i}")
            print(f"  old: {old_labels[i]}")
            print(f"  new: {new_labels[i]}")

    print("\n=== Step 4 relabel: churn vs pre-Step-4 labels ===")
    print(f"items={len(order)}  candidates/item={args.topk}  escape-hatch new skills={len(escape_new)}")
    print(f"mean Jaccard(old,new) per item = {np.mean(jac):.2f}")
    print(f"items with identical skill set = {identical}/{len(order)} ({100*identical/len(order):.0f}%)")
    print(f"old labels={n_old}  new labels={n_new}  retained(in both)={retained} "
          f"({100*retained/max(n_old,1):.0f}% of old)")
    print(f"dropped from old={changed_out}  added in new={added}")
    print(f"distinct skills used: old={used_old}  new={used_new}  (bank=189)")

    if args.out and not args.smoke:
        bank = [{"name": n, "definition": defn[n], "rank": k} for k, n in enumerate(names)]
        bank += [{"name": n, "definition": d, "rank": len(names) + k}
                 for k, (n, d) in enumerate(escape_new.items())]
        json.dump({"subset": merged["subset"], "model": args.model, "topk": args.topk,
                   "n_items": len(order), "bank": bank,
                   "per_item": [{"item_idx": i, "benchmark": meta[i][0],
                                 "subtask": meta[i][1], "skills": new_labels[i]}
                                for i in order],
                   "churn": {"mean_jaccard": float(np.mean(jac)),
                             "identical_frac": identical / len(order),
                             "retained_frac": retained / max(n_old, 1),
                             "dropped": changed_out, "added": added,
                             "used_old": used_old, "used_new": used_new,
                             "escape_new": len(escape_new)}},
                  open(args.out, "w"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
