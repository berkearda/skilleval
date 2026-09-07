"""T-084: repair the irreducible object-locations cluster (1330q).

Diagnosis 2026-07-22: the model classifies correctly but emits one line per
question (fixed by the name-aggregating parser) and skips the MuSR narrative
block (fixed here by a discovery pass on the missing items + batch assign).

Steps: attempt-A pile with aggregating parser (raw output SAVED this time),
discovery on the missing block, batch-assign leftovers, degenerate check,
replace the flagged skill, certify new substantive piles fresh, patch
certified.json, snapshot round8_objloc.

    python3 tools/fix_objloc.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

tp.P = tp.E / "pipeline_fresh"
SKILL = "Inferring Object Locations From Contextual Clues"
rng = random.Random(42)


def main():
    st = tp.State(json.load(open(tp.P / "round7_names.json")))
    cert = json.load(open(tp.P / "certified.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(tp.D / "item_full_text_recovered.json"))}
    jd = tp.Judges(tp.make_client(), qtext)
    order = sorted(st.skills[SKILL]["items"])
    n = len(order)
    print(f"repairing '{SKILL}' ({n} questions)")

    # --- attempt A with aggregating parse, raw saved (cached if already run)
    rawf = tp.P / "objloc_raw_attemptA.txt"
    if rawf.exists() and len(rawf.read_text()) > 10000:
        txt = rawf.read_text()
        print("using cached attempt-A output")
    else:
        qs = "\n".join(f"{k+1}. {qtext.get(i,'')[:900]}" for k, i in enumerate(order))
        txt = jd._text(tp.BIG, tp.PSYS, tp.COMPACT_U.format(n=n, rules=tp.RULES, questions=qs))
        rawf.write_text(txt or "")
    by_name, count = {}, defaultdict(int)
    for line in (txt or "").splitlines():
        m = tp.LINE.match(line)
        if not m:
            continue
        name = m.group(1).strip().lower()
        nums = [int(x) for x in m.group(3).split()
                if x.isdigit() and 1 <= int(x) <= n and count[int(x)] < 2]
        for x in nums:
            count[x] += 1
        if not nums:
            continue
        if name in by_name:
            by_name[name]["nums"] += nums
        else:
            by_name[name] = {"name": name, "desc": m.group(2).strip()[:200], "nums": nums}
    piles = {k: v for k, v in by_name.items() if len(v["nums"]) >= 1}
    missing = [k for k in range(1, n + 1) if count[k] == 0]
    print(f"attempt A: {len(piles)} aggregated piles, missing {len(missing)}")

    # --- pile the skipped block directly, compact format (JSON discovery proved fragile)
    if missing:
        from collections import Counter as _C
        old_map = json.load(open(tp.E / "oldtax_repaired_FINAL.json"))
        bmap = {pi["item_idx"]: pi["benchmark"] for pi in old_map["per_item"]}
        print("skipped block composition:",
              dict(_C(bmap.get(order[k-1]) for k in missing)))
        qs2 = "\n".join(f"{j+1}. {qtext.get(order[k-1],'')[:900]}"
                        for j, k in enumerate(missing))
        txt2 = jd._text(tp.BIG, tp.PSYS,
                        tp.COMPACT_U.format(n=len(missing), rules=tp.RULES, questions=qs2))
        (tp.P / "objloc_raw_block.txt").write_text(txt2 or "")
        sub_count = defaultdict(int)
        for line in (txt2 or "").splitlines():
            m = tp.LINE.match(line)
            if not m:
                continue
            name = m.group(1).strip().lower()
            nums = [missing[int(x)-1] for x in m.group(3).split()
                    if x.isdigit() and 1 <= int(x) <= len(missing)
                    and sub_count[int(x)] < 2]
            for x in m.group(3).split():
                if x.isdigit():
                    sub_count[int(x)] += 1
            nums = [x for x in nums if count[x] == 0 or True]
            if not nums:
                continue
            for x in nums:
                count[x] += 1
            if name in piles:
                piles[name]["nums"] += nums
            else:
                piles[name] = {"name": name, "desc": m.group(2).strip()[:200],
                               "nums": nums}
        missing = [k for k in range(1, n + 1) if count[k] == 0]
        print(f"after piling the skipped block: {len(piles)} piles, missing {len(missing)}")

    # --- batch-assign the rest against the full pile table
    if missing:
        table = sorted(piles.values(), key=lambda p: -len(p["nums"]))
        tbl = "\n".join(f"{k+1}) {p['name']} | {p['desc']}" for k, p in enumerate(table))

        def batch(chunk, force):
            fb = (" You MUST choose the closest pile; 0 is not allowed." if force
                  else " If none fits, use 0.")
            qb = "\n".join(f"{k}. {qtext.get(order[k-1],'')[:600]}" for k in chunk)
            out = {}
            t = jd._text(tp.SMALL, tp.PSYS,
                         tp.ASSIGN_U.format(table=tbl, questions=qb, fallback=fb),
                         max_out=2000)
            for line in (t or "").splitlines():
                m = re.match(r"^\s*(\d+)\s*->\s*(\d+)", line)
                if m:
                    out[int(m.group(1))] = int(m.group(2))
            return out

        unplaced = list(missing)
        for force in (False, True):
            still = []
            chunks = [unplaced[i:i+40] for i in range(0, len(unplaced), 40)]
            with ThreadPoolExecutor(max_workers=8) as ex:
                for res_map, chunk in zip(ex.map(lambda c: batch(c, force), chunks), chunks):
                    for k in chunk:
                        p = res_map.get(k, 0)
                        if 1 <= p <= len(table):
                            table[p-1]["nums"].append(k)
                            count[k] += 1
                        else:
                            still.append(k)
            unplaced = still
            if not unplaced:
                break
        print(f"after assignment: unplaced {len(unplaced)}")
        for k in unplaced:
            piles[f"unplaced question {k}"] = {"name": f"unplaced question {k}",
                                               "desc": "", "nums": [k]}

    plist = [{"name": p["name"], "desc": p["desc"], "nums": sorted(set(p["nums"]))}
             for p in piles.values()]
    if tp.degenerate(plist, n):
        sys.exit("STILL DEGENERATE - aborting, nothing changed")
    print(f"final piling: {len(plist)} piles "
          f"(top: {sorted((len(p['nums']) for p in plist), reverse=True)[:6]})")

    # --- replace the flagged skill
    del st.skills[SKILL]
    new = []
    for p in plist:
        items = {order[k-1] for k in p["nums"]}
        name = p["name"]
        base, j = name, 2
        while name in st.skills:
            name = f"{base} ({j})"; j += 1
        st.skills[name] = {"definition": p["desc"], "items": items,
                           "attempts": 1, "flag": None}
        new.append(name)
    tp.journal("repile", skill=SKILL, piles=[(s, len(st.skills[s]["items"])) for s in new],
               round=8, note="T-084 aggregating parser + skipped-block recovery")

    # --- certify new substantive piles fresh
    subst = [s for s in new if len(st.skills[s]["items"]) >= 4]
    print(f"certifying {len(subst)} new substantive piles")
    st.round = 300
    tp.measure(st, jd, subst, model=tp.BIG,
               val_n=tp.CERT_VAL_N, grp_n=tp.CERT_GRP_N, topup=False)
    rows = [r for r in cert["rows"]]
    for s in subst:
        g = st.metrics["group"].get(s)
        v = st.metrics["validity"].get(s)
        bar = tp.BIG_THR if len(st.skills[s]["items"]) > tp.BIG_CLUSTER else tp.ROUTE_THR
        ok = (g is None or g > bar) and (v is None or v > tp.ROUTE_THR)
        rows.append({"skill": s, "n_items": len(st.skills[s]["items"]),
                     "validity": v, "group": g, "certified": ok})
        print(f"  {len(st.skills[s]['items']):5d}q  val {v:4.0%}  grp {g:4.0%}  "
              f"{'CERTIFIED' if ok else 'failed'}  {s[:50]}")
    gm = [r["group"] for r in rows if r["group"] is not None]
    vm = [r["validity"] for r in rows if r["validity"] is not None]
    cert.update(rows=rows, mean_validity=sum(vm)/len(vm), mean_group=sum(gm)/len(gm),
                n_certified=sum(r["certified"] for r in rows),
                n_failed=sum(not r["certified"] for r in rows),
                objloc_fix="2026-07-22 fix_objloc.py")
    json.dump(cert, open(tp.P / "certified.json", "w"), indent=1)

    cov = st.item_cover()
    assert all(cov.get(i, 0) >= 1 for i in range(9523)), "coverage invariant violated"
    f = st.save("round8_objloc")
    print(f"snapshot {f.name}  certified {cert['n_certified']}/{len(rows)}  "
          f"validity {cert['mean_validity']:.0%}  group {cert['mean_group']:.0%}  "
          f"spent ${tp.spend['usd']:.2f}")


if __name__ == "__main__":
    main()
