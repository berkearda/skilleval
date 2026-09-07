"""De-duplicate the v5.1 bank before the final assignment.

The v5 dedup ran before the v5.1 splits created new paraphrase-duplicates
("deduce order from positional clues" / "infer order from logical statements" /
"determine order ranking from statements" = one skill, three names). Token
overlap misses paraphrases, so the primary nomination signal here is
CO-ASSIGNMENT: the multi-label pass put duplicate skills on the same question.

Nominate (co-assigned >= 4 times) OR (name-token Jaccard >= 0.4) OR
(item overlap >= 0.3); judge same/different; union-find; each merged union
must pass the 12-trial group test or the merge is dropped. Keep the
best-certified name. Writes round7_dedup snapshot.

    python3 tools/dedup_v51.py
"""
from __future__ import annotations

import itertools
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

tp.P = tp.E / "pipeline_v51"
V = tp.E / "pipeline_v51"
STOP = {"of", "the", "with", "from", "for", "in", "to", "a", "an", "and", "or", "by"}
rng = random.Random(42)


def toks(s):
    return {w for w in s.lower().replace("(2)", " ").split() if w not in STOP}


MSYS = (f"You judge whether two skill groups in a test-skill taxonomy are the same skill. "
        f"{tp.SKILL_DEF} Judge by the operation, never by topic or wording.")
MERGE_U = """Two skill groups:

Skill A: {na}
Sample questions:
{qa}

Skill B: {nb}
Sample questions:
{qb}

Do A and B require the SAME cognitive operation (should they be one skill), or are they genuinely different operations? Return JSON {{"same": true|false}}."""


def main():
    snap = json.load(open(V / "round6.json"))
    cert = json.load(open(V / "certified.json"))
    crow = {r["skill"]: r for r in cert["rows"]}
    st = tp.State(snap)
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(tp.D / "item_full_text_recovered.json"))}

    bank = [s for s in st.substantive() if not st.skills[s]["flag"]]
    bankset = set(bank)

    # co-assignment signal from the multi-label pass
    co = Counter()
    for line in open(V / "assignments_multi.jsonl"):
        sk = [s for s in json.loads(line)["skills"] if s in bankset]
        for a, b in itertools.combinations(sorted(set(sk)), 2):
            co[(a, b)] += 1

    nominated = set()
    for (a, b), c in co.items():
        if c >= 4:
            nominated.add((a, b))
    items = {s: set(st.skills[s]["items"]) for s in bank}
    for a, b in itertools.combinations(bank, 2):
        ta, tb = toks(a), toks(b)
        nj = len(ta & tb) / len(ta | tb) if ta | tb else 0
        io = len(items[a] & items[b]) / min(len(items[a]), len(items[b]))
        if nj >= 0.4 or io >= 0.3:
            nominated.add(tuple(sorted((a, b))))
    pairs = sorted(nominated)
    print(f"bank {len(bank)} skills; nominated {len(pairs)} pairs "
          f"({sum(1 for p in pairs if co.get(p,0)>=4)} from co-assignment)")

    jd = tp.Judges(tp.make_client(), qtext)

    def sample_q(s, n=5):
        return "\n".join(f"- {qtext.get(i,'')[:250]}"
                         for i in rng.sample(sorted(st.skills[s]["items"]),
                                             min(n, len(st.skills[s]["items"]))))

    def decide(p):
        a, b = p
        u = MERGE_U.format(na=a, qa=sample_q(a), nb=b, qb=sample_q(b))
        return (a, b, bool(jd._json(tp.SMALL, MSYS, u, "same")))

    with ThreadPoolExecutor(max_workers=8) as ex:
        verd = list(ex.map(decide, pairs))
    same = [(a, b) for a, b, s in verd if s]
    print(f"judge says same: {len(same)}/{len(verd)}")

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in same:
        parent[find(a)] = find(b)
    comps = defaultdict(list)
    for s in {x for ab in same for x in ab}:
        comps[find(s)].append(s)
    comps = [sorted(v) for v in comps.values() if len(v) > 1]
    print(f"{len(comps)} merge components")

    merged = 0
    all_items = sorted({i for d in st.skills.values() for i in d["items"]})
    for comp in comps:
        union = set()
        for s in comp:
            union |= st.skills[s]["items"]
        bar = tp.BIG_THR if len(union) > tp.BIG_CLUSTER else tp.ROUTE_THR
        pool = [i for i in all_items if i not in union]
        trials = []
        for _ in range(12):
            own = rng.sample(sorted(union), 4); intr = rng.choice(pool)
            five = own + [intr]; rng.shuffle(five)
            trials.append((five, five.index(intr) + 1))
        with ThreadPoolExecutor(max_workers=8) as ex:
            hits = sum(ex.map(lambda t: jd.group_one(tp.SMALL, t[0], t[1]), trials))
        keep = max(comp, key=lambda s: ((crow.get(s) or {}).get("validity") or 0,
                                        len(st.skills[s]["items"])))
        ok = hits / 12 > bar
        print(f"  {'MERGE' if ok else 'drop '} [{hits}/12 bar {bar:.0%}] "
              f"{len(union)}q -> '{keep}'  ({', '.join(c[:22] for c in comp)})")
        if not ok:
            continue
        defn = st.skills[keep]["definition"]
        for s in comp:
            del st.skills[s]
        st.skills[keep] = {"definition": defn, "items": union, "attempts": 0, "flag": None}
        tp.journal("dedup_merge", component=comp, keep=keep, union=len(union),
                   group=hits / 12)
        merged += 1

    cov = st.item_cover()
    assert all(cov.get(i, 0) >= 1 for i in range(9523)), "coverage invariant"
    f = st.save("round7_dedup")
    print(f"merged {merged} families; snapshot {f.name}; "
          f"bank now {len([s for s in st.substantive() if not st.skills[s]['flag']])} "
          f"non-flagged substantive; spent ${tp.spend['usd']:.2f}")


if __name__ == "__main__":
    main()
