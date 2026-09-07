"""T-085: de-duplication pass for v5 (the merge stage the fresh run lacked).

Nominate near-twin skills (name-token overlap or item overlap), LLM judge
decides same/different operation, union-find components merge, every merged
union must pass the strict group test (12 trials, size-aware bar) or the
merge is undone. Newly created merged skills are certified fresh (they never
faced the certifier, so the one-shot policy is intact).

    python3 tools/merge_v5.py dry    # show nominated pairs only
    python3 tools/merge_v5.py run    # judge, merge, verify, certify
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

tp.P = tp.E / "pipeline_fresh"
FRESH = tp.E / "pipeline_fresh"
STOP = {"of", "the", "with", "from", "for", "in", "to", "a", "an", "and", "or"}

MSYS = (f"You judge whether two skill groups in a test-skill taxonomy are the same skill. "
        f"{tp.SKILL_DEF} Judge by the operation, never by topic or wording.")
MERGE_U = """Two skill groups from a taxonomy:

Skill A: {na}
Sample questions:
{qa}

Skill B: {nb}
Sample questions:
{qb}

Do A and B require the SAME cognitive operation (would merging them into one skill be correct), or are they genuinely different operations? Return JSON {{"same": true|false}}."""


def norm_tokens(name):
    return {w for w in name.lower().replace("(2)", " ").replace("(3)", " ").split()
            if w not in STOP}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    snaps = sorted(FRESH.glob("round*.json"))
    print(f"state: {snaps[-1].name}")
    snap = json.load(open(snaps[-1]))
    st = tp.State(snap)
    cert = json.load(open(FRESH / "certified.json"))
    crow = {r["skill"]: r for r in cert["rows"]}

    cand = sorted(s for s in st.substantive() if not st.skills[s]["flag"])
    pairs = []
    for i, a in enumerate(cand):
        ta, ia = norm_tokens(a), st.skills[a]["items"]
        for b in cand[i + 1:]:
            tb, ib = norm_tokens(b), st.skills[b]["items"]
            nj = len(ta & tb) / len(ta | tb) if ta | tb else 0
            io = len(ia & ib) / min(len(ia), len(ib))
            if nj >= 0.5 or io >= 0.3:
                pairs.append((a, b, round(nj, 2), round(io, 2)))
    print(f"nominated pairs: {len(pairs)}")
    for a, b, nj, io in pairs:
        print(f"  name-sim {nj:.2f}  item-overlap {io:.2f}  {a[:44]:46s} | {b[:44]}")
    if mode == "dry":
        return

    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(tp.D / "item_full_text_recovered.json"))}
    jd = tp.Judges(tp.make_client(), qtext)
    rng = random.Random(42)

    def sample_q(s, n=5):
        return "\n".join(f"- {qtext.get(i, '')[:250]}"
                         for i in rng.sample(sorted(st.skills[s]["items"]),
                                             min(n, len(st.skills[s]["items"]))))

    def decide(p):
        a, b, _, _ = p
        u = MERGE_U.format(na=a, qa=sample_q(a), nb=b, qb=sample_q(b))
        r = jd._json(tp.SMALL, MSYS, u, "same")
        return (a, b, bool(r))

    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = list(ex.map(decide, pairs))

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)
    same = [(a, b) for a, b, s in verdicts if s]
    print(f"judge says same: {len(same)}/{len(verdicts)}")
    for a, b in same:
        union(a, b)
    comps = defaultdict(list)
    for a, b in same:
        comps[find(a)]
    for s in {x for ab in same for x in ab}:
        comps[find(s)].append(s)
    comps = [sorted(v) for v in comps.values() if len(v) > 1]

    merged_names = []
    for comp in comps:
        # keep the best-certified name (validity, then size)
        keep = max(comp, key=lambda s: ((crow.get(s) or {}).get("validity") or 0,
                                        len(st.skills[s]["items"])))
        items = set()
        for s in comp:
            items |= st.skills[s]["items"]
        bar = tp.BIG_THR if len(items) > tp.BIG_CLUSTER else tp.ROUTE_THR
        # strict verification: 12 intruder trials on the union
        all_items = sorted({i for d in st.skills.values() for i in d["items"]})
        pool = [i for i in all_items if i not in items]
        trials = []
        for _ in range(12):
            own = rng.sample(sorted(items), 4)
            intr = rng.choice(pool)
            five = own + [intr]
            rng.shuffle(five)
            trials.append((five, five.index(intr) + 1))
        with ThreadPoolExecutor(max_workers=8) as ex:
            hits = sum(ex.map(lambda t: jd.group_one(tp.SMALL, t[0], t[1]), trials))
        ok = hits / 12 > bar
        print(f"  component {comp} -> '{keep}' union {len(items)}q "
              f"group {hits}/12 bar {bar:.0%} -> {'MERGE' if ok else 'UNDO'}")
        tp.journal("merge_component", component=comp, keep=keep, union=len(items),
                   group=hits / 12, bar=bar, accepted=ok)
        if not ok:
            continue
        defn = st.skills[keep]["definition"]
        for s in comp:
            del st.skills[s]
        st.skills[keep] = {"definition": defn, "items": items, "attempts": 0, "flag": None}
        merged_names.append(keep)

    if not merged_names:
        print("no merges accepted; v5 distinctness verified as-is")
    else:
        # certify the merged skills fresh (new objects, never certified before)
        st.round = 200
        tp.measure(st, jd, merged_names, model=tp.BIG,
                   val_n=tp.CERT_VAL_N, grp_n=tp.CERT_GRP_N, topup=False)
        rows = [r for r in cert["rows"] if r["skill"] in st.skills]
        for s in merged_names:
            g = st.metrics["group"].get(s)
            v = st.metrics["validity"].get(s)
            bar = tp.BIG_THR if len(st.skills[s]["items"]) > tp.BIG_CLUSTER else tp.ROUTE_THR
            ok = (g is None or g > bar) and (v is None or v > tp.ROUTE_THR)
            rows = [r for r in rows if r["skill"] != s]
            rows.append({"skill": s, "n_items": len(st.skills[s]["items"]),
                         "validity": v, "group": g, "certified": ok})
            print(f"  merged '{s}' certified: val {v:.0%} grp {g:.0%} -> {ok}")
        gm = [r["group"] for r in rows if r["group"] is not None]
        vm = [r["validity"] for r in rows if r["validity"] is not None]
        cert.update(rows=rows, mean_validity=sum(vm) / len(vm), mean_group=sum(gm) / len(gm),
                    n_certified=sum(r["certified"] for r in rows),
                    n_failed=sum(not r["certified"] for r in rows),
                    dedup_pass="2026-07-22 merge_v5.py")
        json.dump(cert, open(FRESH / "certified.json", "w"), indent=1)

    f = st.save("round9_dedup2" if (FRESH / "round6_dedup.json").exists() else "round6_dedup")
    cov = st.item_cover()
    assert all(cov.get(i, 0) >= 1 for i in range(9523)), "coverage invariant violated"
    print(f"snapshot {f.name}  coverage 9523/9523  spent ${tp.spend['usd']:.2f}")


if __name__ == "__main__":
    main()
