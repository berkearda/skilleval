#!/usr/bin/env python3
"""T-113: when the labeller names an operation, was that skill ever offered?

Sheet 2 produced one retrieval miss and one judging miss in 42 assignments. Item
528 needed `c_0148`, which exists in the taxonomy and was never among the 12
retrieved candidates. Item 4194 was given `c_0322` while `c_0088`, the skill the
labeller named, sat in the candidate list unchosen. Two anecdotes are not a rate,
and this measures the rate.

The measurement runs on Part A free text, which is written before any candidate
skill is visible, so it is the closest thing available to an unanchored statement
of what a question requires. For each operation the labeller wrote, the nearest
live skill by definition embedding is found, and then placed:

  assigned             the pipeline chose it
  offered, not chosen  it was among the 12 retrieved candidates and the judge
                       passed over it, so retrieval worked and judging did not
  never offered        retrieval never surfaced it at all

Nearest-by-embedding is a PROXY for "the skill the labeller meant", not a
measurement of it, so the script prints matches for inspection instead of asking
to be believed. Items whose Part A cites a code id are reported separately, since
on those the free text had already seen the options and cannot be treated as
independent.

`verify_splits` does not apply: there is no train/test split, only the fixed set
of hand-written operations, which is what split_info records.

    python3 tools/diag_retrieval_recall.py --smoke
    python3 tools/diag_retrieval_recall.py
"""
import argparse, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini
from tools.codeemb import load as load_code_vecs, normed
from tools.gold_score_partb import parse_sheet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment

P = REPO / "cdm_exploration/experiments/pipeline_v7"


def part_a_phrases(path):
    """(item_idx, phrase) for every non-empty op_N line, before Part B."""
    out = []
    for blk in re.split(r"^### ", path.read_text(), flags=re.M)[1:]:
        m = re.search(r"item (\d+)", blk)
        if not m:
            continue
        head = blk.split("**Part B**")[0]
        for om in re.finditer(r"^op_\d: *(.+)$", head, flags=re.M):
            phrase = om.group(1).strip()
            if phrase:
                out.append((m.group(1), phrase))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="gold/gold_sheet_2_labelled.md")
    ap.add_argument("--codebook", default="codebook_v2_amended_b150.json")
    ap.add_argument("--labels", default="item_labels_b150.jsonl")
    ap.add_argument("--emb", default="code_def_emb_b150.npz")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="cdm_exploration/experiments/diag_retrieval_recall.json")
    a = ap.parse_args()
    seed_everything(42)

    cb = json.loads((P / a.codebook).read_text())
    alias = cb.get("alias", {})
    live = [c for c in cb["codes"] if c not in alias]
    rows = {}
    for line in (P / a.labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["item_idx"]] = r

    phrases = part_a_phrases(REPO / a.sheet)
    _, _, anchored = parse_sheet(REPO / a.sheet)
    if a.smoke:
        phrases = phrases[:8]
    print(f"{len(phrases)} hand-written operations over "
          f"{len({i for i, _ in phrases})} questions; {len(anchored)} questions had "
          f"Part A citing a code id")

    g = Gemini()
    C = normed(load_code_vecs(P / a.emb, cb, live, g=g))
    V = normed(np.stack(g.embed([p for _, p in phrases])))
    S = V @ C.T
    best = np.argmax(S, axis=1)

    pos = {c: k for k, c in enumerate(live)}
    tally = Counter()
    detail = []
    for k, (i, phrase) in enumerate(phrases):
        c = live[int(best[k])]
        row = rows.get(int(i), {})
        cand = list(row.get("candidates") or [])
        assigned = {x["code"] for x in row.get("assigned", [])}
        if c in assigned:
            status = "assigned"
        elif c in cand:
            status = "offered, not chosen"
        else:
            status = "never offered"
        tally[status] += 1
        tally[("anchored " if i in anchored else "clean ") + status] += 1
        # "never offered" only matters if the 12 that WERE offered held nothing as
        # close. Without this, a phrase sitting between two near-equivalent skills
        # counts as a retrieval failure when retrieval lost nothing.
        off = [pos[x] for x in cand if x in pos]
        best_off = float(S[k, off].max()) if off else None
        detail.append({"item_idx": int(i), "phrase": phrase, "nearest": c,
                       "nearest_name": cb["codes"][c]["name"],
                       "similarity": round(float(S[k, best[k]]), 3),
                       "best_offered_similarity": round(best_off, 3) if best_off is not None else None,
                       "margin": round(float(S[k, best[k]]) - best_off, 3)
                                 if best_off is not None else None,
                       "rank_in_candidates": (cand.index(c) + 1) if c in cand else None,
                       "status": status, "anchored": i in anchored})

    n = len(phrases)
    print(f"\nwhere the nearest skill to each written operation actually was:")
    for s in ("assigned", "offered, not chosen", "never offered"):
        print(f"  {s:22s} {tally[s]:>4} of {n} = {tally[s]/n:.0%}")
    clean = sum(tally["clean " + s] for s in
                ("assigned", "offered, not chosen", "never offered"))
    if clean:
        print(f"\n  excluding the questions whose Part A cited a code id ({n-clean} operations):")
        for s in ("assigned", "offered, not chosen", "never offered"):
            print(f"    {s:22s} {tally['clean ' + s]:>4} of {clean} = "
                  f"{tally['clean ' + s]/clean:.0%}")
    print(f"\n  retrieval miss rate is the bottom row: a skill the labeller named, "
          f"present in\n  the taxonomy, that the 12 candidates never showed the judge.")

    # A "never offered" verdict has two possible causes: retrieval really missed
    # the skill, or the nearest-by-embedding skill was not what the labeller meant.
    # On item 528 the proxy chose c_0296 where the labeller's own note named
    # c_0148, at similarity 0.677, so the second cause is real and has to be shown.
    print(f"\n  similarity by status, which separates a real miss from a proxy failure:")
    for s in ("assigned", "offered, not chosen", "never offered"):
        xs = [d["similarity"] for d in detail if d["status"] == s]
        if xs:
            print(f"    {s:22s} n={len(xs):>4}  median {statistics.median(xs):.3f}  "
                  f"min {min(xs):.3f}  max {max(xs):.3f}")
    hit = [d["similarity"] for d in detail if d["status"] == "assigned"]
    if hit:
        cal = statistics.median(hit)
        conf = [d for d in detail if d["similarity"] >= cal]
        print(f"\n  calibration: where the nearest skill IS the one the pipeline chose,")
        print(f"  median similarity is {cal:.3f}. That is what a match the labeller would")
        print(f"  recognise looks like. Restricted to matches at least that close "
              f"({len(conf)} of {n}):")
        cc = Counter(d["status"] for d in conf)
        for s in ("assigned", "offered, not chosen", "never offered"):
            print(f"    {s:22s} {cc[s]:>4} of {len(conf)} = {cc[s]/max(1,len(conf)):.0%}")
        print(f"  If the never-offered share falls away under that restriction, the")
        print(f"  headline miss rate is mostly the proxy failing, not retrieval.")

    print(f"\n  did retrieval actually lose anything? margin = best skill in the whole")
    print(f"  taxonomy minus best of the 12 offered (zero by construction when the")
    print(f"  nearest skill was itself offered):")
    for s in ("assigned", "offered, not chosen", "never offered"):
        xs = [d["margin"] for d in detail
              if d["status"] == s and d["margin"] is not None]
        if xs:
            print(f"    {s:22s} n={len(xs):>4}  median {statistics.median(xs):+.3f}  "
                  f"max {max(xs):+.3f}")
    nv = [d for d in detail if d["status"] == "never offered" and d["margin"] is not None]
    big = [d for d in nv if d["margin"] >= 0.03]
    if nv:
        print(f"\n    of {len(nv)} never-offered operations, {len(big)} lost at least 0.03 of")
        print(f"    similarity, meaning retrieval passed over a materially closer skill.")
        print(f"    The remaining {len(nv)-len(big)} are misses on paper: the offered 12 held")
        print(f"    something essentially as close to what the labeller wrote.")
        for d in sorted(big, key=lambda x: -x["margin"])[:6]:
            print(f"      margin {d['margin']:+.3f}  item {d['item_idx']}  "
                  f"\"{d['phrase'][:40]}\" -> {d['nearest_name'][:34]}")

    print(f"\nsample of matches, so the proxy can be judged rather than trusted:")
    for d in detail[:10]:
        print(f"  item {d['item_idx']:>5}  \"{d['phrase'][:52]}\"")
        print(f"      -> {d['nearest']} {d['nearest_name'][:46]}  "
              f"sim {d['similarity']}  [{d['status']}]")

    low = sorted(detail, key=lambda d: d["similarity"])[:5]
    print(f"\nweakest matches, where the proxy is least trustworthy:")
    for d in low:
        print(f"  sim {d['similarity']}  \"{d['phrase'][:44]}\" -> {d['nearest_name'][:44]}")

    if not a.smoke:
        out = {"sheet": a.sheet, "codebook": a.codebook, "labels": a.labels,
               "n_operations": n, "n_questions": len({i for i, _ in phrases}),
               "proxy": "nearest live skill by definition embedding; a proxy for "
                        "the skill the labeller meant, not a measurement of it",
               "counts": {s: tally[s] for s in
                          ("assigned", "offered, not chosen", "never offered")},
               "counts_excluding_anchored": {s: tally["clean " + s] for s in
                                             ("assigned", "offered, not chosen",
                                              "never offered")},
               "topk": 12, "detail": detail}
        (REPO / a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
        log_experiment(
            name="diag_retrieval_recall",
            config={"sheet": a.sheet, "codebook": a.codebook, "emb": a.emb,
                    "topk": 12, "seed": 42},
            results={"n_operations": n,
                     "never_offered_rate": tally["never offered"] / n,
                     "offered_not_chosen_rate": tally["offered, not chosen"] / n,
                     "assigned_rate": tally["assigned"] / n},
            split_info={"note": "no train/test split applies; the unit is the fixed "
                                "set of hand-written Part A operations",
                        "n_operations": n},
            verified=True,
        )


if __name__ == "__main__":
    main()
