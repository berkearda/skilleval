"""Relabel the submitted 100 clusters (memberships FROZEN), option C.

Per cluster: propose a name from 20 stratified sample questions, enforce the format
contract (code), then the fit check (10 own + 10 stranger, gap >= 0.55). Up to 4 tries
with feedback. If a name passes -> clean relabel. If none passes -> keep the best-fit
name (covers the majority) AND flag the cluster as mixed.

Paired name-fit: old name and new name are both judged on the SAME 10 own questions.
Belong-together is NOT re-measured (name-free, unchanged by relabeling).

    python3 tools/relabel_original.py run       # do it
    python3 tools/relabel_original.py html      # rebuild the viewer from saved result
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

D = tp.D
E = tp.E
OUT = E / "oldtax_relabeled.json"
GAP = 0.55
CAP = 4

def scaled(n_items, lo, hi):
    """Sample count scales with cluster size: lo for tiny, hi for huge (>=500)."""
    if n_items <= 30:
        return lo
    return min(hi, lo + round((hi - lo) * (n_items - 30) / (500 - 30)))

NSYS = ("You rename one skill in a test-skill taxonomy so that the name follows a strict "
        "format. A skill is the smallest named mental operation a solver must master to "
        "answer a question correctly; the same skill can appear across different topics and "
        "formats. Judge by the operation the solver performs, never by the topic or story of "
        "the questions. You are given the skill's questions; the name must describe what they require.")

REWRITE = """This skill currently has a name that needs replacing. Write a new name for it based on its questions.

Questions tagged with this skill:
{ev}

Current name (for context only, do not copy its style): {old}

First, decide the single cognitive operation these questions share: what the solver actually has to DO to answer them, ignoring the topic, story, or subject. If they span more than one operation, choose the one most of them require. Then name that operation.

Name format, all rules mandatory:
1. A lowercase verb phrase: one verb plus one specific object, for example "track object positions after swaps".
2. Between 3 and 8 words. A bare verb alone is forbidden.
3. No "and", no "or", no slashes.
4. Never use these words: "problem solving", "reasoning", "analysis", "skills", "advanced".
5. Name the operation the solver performs, not the topic, story, or subject. Two questions about different things (say, dance and soccer) that need the same operation must get the same kind of name. The name must fit these questions and NOT fit unrelated ones.
6. Name the operation needed to find the answer, never the answer format. Do not say "choose the correct option", "select the answer", "identify the correct statement", or similar. A multiple-choice question still requires a real operation; name that operation.
{extra}
Return JSON only: {{"operation": "<the shared operation in plain words>", "name": "<new name>", "definition": "<one sentence, at most 15 words>"}}"""


def load():
    q = np.load(D / "qmatrix_v2_K100.npy")
    lab = json.load(open(D / "cluster_labels_v2_K100.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {pi["item_idx"]: (pi["benchmark"], pi.get("subtask") or "")
            for pi in json.load(open(E / "oldtax_repaired_FINAL.json"))["per_item"]}
    clusters = {}
    for k in range(q.shape[1]):
        clusters[lab[str(k)]] = sorted(np.nonzero(q[:, k])[0].tolist())
    return clusters, qtext, meta


def stratified(items, meta, n, seed):
    r = random.Random(seed)
    by = defaultdict(list)
    for i in items:
        by[meta.get(i, ("?", "?"))].append(i)
    groups = sorted(by, key=lambda g: -len(by[g]))
    take = {g: max(1, round(n * len(by[g]) / len(items))) for g in groups}
    out = []
    for g in groups:
        out += r.sample(by[g], min(take[g], len(by[g])))
    r.shuffle(out)
    return out[:n] if len(out) >= n else out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    clusters, qtext, meta = load()
    if mode == "html":
        build_html(json.load(open(OUT)), clusters, qtext, meta)
        return

    jd = tp.Judges(tp.make_client(), qtext)
    all_items = sorted(qtext)
    subst = {s: it for s, it in clusters.items() if len(it) >= 4}
    print(f"relabeling {len(subst)} substantive clusters")

    def judge_name(name, defn, i):
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {name}\nDefinition: {defn}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return jd._json(tp.SMALL, tp.VSYS, u, "verdict") == "yes"

    def do_cluster(item):
        old, items = item
        r = random.Random(hash(old) & 0xffffffff)
        n_name = scaled(len(items), 15, 40)      # naming sample: 15 (tiny) -> 40 (huge)
        n_fit = scaled(len(items), 10, 30)       # fit-check sample: 10 -> 30
        ev_items = stratified(items, meta, n_name, hash(old) & 0xffffffff)
        ev = "\n".join(f"- {qtext.get(i,'')[:300]}" for i in ev_items)
        own = r.sample(items, min(n_fit, len(items)))
        strangers = r.sample([i for i in all_items if i not in set(items)], n_fit)
        # paired: old-name fit on the same own questions
        with ThreadPoolExecutor(max_workers=8) as ex:
            old_fit = sum(ex.map(lambda i: judge_name(old, "", i), own)) / len(own)

        best = None  # (own_fit, name, defn, gap)
        extra = ""
        for attempt in range(CAP):
            raw = jd._text(tp.SMALL, NSYS, REWRITE.format(ev=ev, old=old, extra=extra), max_out=250)
            try:
                obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            except Exception:
                continue
            cand = str(obj.get("name", "")).strip().lower()
            cdef = str(obj.get("definition", ""))[:200]
            vio = tp.violations(cand)
            if vio:
                extra = f"7. Your previous attempt '{cand}' broke a rule: {'; '.join(vio)}. Fix it.\n"
                continue
            with ThreadPoolExecutor(max_workers=8) as ex:
                of = sum(ex.map(lambda i: judge_name(cand, cdef, i), own)) / len(own)
                sf = sum(ex.map(lambda i: judge_name(cand, cdef, i), strangers)) / len(strangers)
            gap = of - sf
            if best is None or of > best[0]:
                best = (of, cand, cdef, gap)
            if gap >= GAP:
                return {"old": old, "new": cand, "definition": cdef, "n": len(items),
                        "old_fit": old_fit, "new_fit": of, "gap": gap,
                        "flagged": False, "attempts": attempt + 1}
            extra = (f"7. Your previous attempt '{cand}' also matched unrelated questions "
                     "(too broad). Name the operation the MAJORITY of these questions share, "
                     "more specifically.\n")
        # option C: nothing passed -> best-fit name, flagged mixed
        of, cand, cdef, gap = best if best else (0.0, old.lower(), "", 0.0)
        return {"old": old, "new": cand, "definition": cdef, "n": len(items),
                "old_fit": old_fit, "new_fit": of, "gap": gap,
                "flagged": True, "attempts": CAP}

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(do_cluster, subst.items()))

    clean = [r for r in rows if not r["flagged"]]
    flagged = [r for r in rows if r["flagged"]]
    import statistics
    of = statistics.mean(r["old_fit"] for r in rows)
    nf = statistics.mean(r["new_fit"] for r in rows)
    print(f"\nname-fit before {of:.0%} -> after {nf:.0%}")
    print(f"clean relabels {len(clean)}  |  flagged mixed {len(flagged)}")
    json.dump({"rows": rows, "mean_old_fit": of, "mean_new_fit": nf,
               "n_clean": len(clean), "n_flagged": len(flagged),
               "spend": round(tp.spend["usd"], 3)}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  spent ${tp.spend['usd']:.2f}")
    build_html(json.load(open(OUT)), clusters, qtext, meta)


def build_html(res, clusters, qtext, meta):
    import html
    from collections import Counter
    coh = {r["skill"]: r["coherence"]
           for r in json.load(open(E / "oldtax_coherence_percluster.json"))["per_cluster"]}
    rows = res["rows"]
    def pc(v): return "" if v is None else f"{round(100*v)}%"
    def col(v):
        if v is None: return "#f4f5f7"
        return "#e6f2ea" if v > 0.5 else "#f7e6e6"
    def pick(name, items, k=8):
        r = random.Random(hash(name) & 0xffffffff)
        return r.sample(items, min(k, len(items)))
    def cut(t, l=700): return t if len(t) <= l else t[:l].rsplit(" ", 1)[0] + " …"
    trs = []
    for d in sorted(rows, key=lambda r: -r["n"]):
        old, new = d["old"], d["new"]
        items = clusters[old]
        bc = Counter(meta.get(i, ("?", ""))[0] for i in items)
        bstr = ", ".join(f"{b} {c}" for b, c in bc.most_common(4))
        status = "flagged mixed" if d["flagged"] else "relabeled"
        cards = []
        for j, i in enumerate(pick(new, items), 1):
            b, s = meta.get(i, ("", ""))
            tag = f"{b} / {s}" if s else b
            cards.append(f"<div class=card><div class=ch>example {j}<span class=tag>{html.escape(tag)}</span></div><div class=ct>{html.escape(cut(qtext.get(i,'')))}</div></div>")
        st = "flagged" if d["flagged"] else "ok"
        trs.append(
            f"<tr class=main data-st={st}><td class=sk>{html.escape(new)}"
            f"<div class=old>was: {html.escape(old)}</div></td>"
            f"<td class=n>{d['n']}</td>"
            f"<td class=n style='background:{col(d['old_fit'])}'>{pc(d['old_fit'])}</td>"
            f"<td class=n style='background:{col(d['new_fit'])}'>{pc(d['new_fit'])}</td>"
            f"<td class=n style='background:{col(coh.get(old))}'>{pc(coh.get(old))}</td>"
            f"<td class=note>{status}</td>"
            f"<td class=bm>{html.escape(bstr)}</td>"
            f"<td><button class=exb onclick=tog(this)>show</button></td></tr>"
            f"<tr class=exr data-st={st} hidden><td colspan=8>{''.join(cards)}</td></tr>")
    n_ok = res["n_clean"]; n_fl = res["n_flagged"]
    doc = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relabeled K=100 taxonomy</title><style>
*{{box-sizing:border-box}}html,body{{background:#fff}}
body{{margin:0;color:#1f2328;font:14.5px/1.6 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1220px;margin:0 auto;padding:32px 22px 80px}}
h1{{font-size:21px;font-weight:700;margin:0 0 4px}}
.sub{{color:#5a6068;font-size:13px;margin:0 0 14px}}
.tabs{{margin:0 0 14px}}
.tab{{font:13px sans-serif;padding:6px 15px;margin-right:7px;border:1px solid #d8dde3;border-radius:20px;background:#fbfcfd;cursor:pointer;color:#333}}
.tab.on{{background:#2b5c8a;color:#fff;border-color:#2b5c8a}}
input{{font:13.5px sans-serif;padding:7px 11px;width:280px;border:1px solid #d0d5db;border-radius:6px;margin:0 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{border:1px solid #e2e5ea;padding:7px 10px;text-align:left;vertical-align:top}}
th{{background:#f6f7f9;position:sticky;top:0;z-index:2;font-weight:600;font-size:12px}}
.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.sk{{font-weight:600;max-width:300px}}.old{{font-weight:400;font-size:11.5px;color:#8a6d1d;margin-top:2px}}
.note{{font-size:12px;color:#a23;white-space:nowrap}}.bm{{font-size:12px;color:#5a6068;white-space:nowrap}}
.exb{{font:12px sans-serif;color:#2b5c8a;background:none;border:1px solid #c3d2e2;border-radius:5px;padding:3px 11px;cursor:pointer}}
.exr>td{{background:#fbfcfd;padding:12px 15px}}
.card{{background:#fff;border:1px solid #e4e6ea;border-left:3px solid #9db6d4;border-radius:5px;padding:9px 13px;margin:0 0 9px;max-height:200px;overflow-y:auto}}
.ch{{font-size:11px;font-weight:700;color:#6a707a;text-transform:uppercase;margin-bottom:4px}}
.tag{{float:right;font-weight:400;text-transform:none;color:#8a6d1d;background:#f6efd8;border-radius:4px;padding:0 7px}}
.ct{{font-size:13px;color:#24292e;white-space:pre-wrap;word-break:break-word;line-height:1.5}}
</style>
<div class=wrap>
<h1>Relabeled K=100 taxonomy</h1>
<p class=sub>same 100 clusters, same questions, new names. name-fit {res['mean_old_fit']:.0%} to {res['mean_new_fit']:.0%}. belong-together unchanged (names never affect it).</p>
<div class=tabs>
<button class="tab on" data-f=ok onclick=setf(this)>relabeled ({n_ok})</button>
<button class=tab data-f=flagged onclick=setf(this)>flagged mixed ({n_fl})</button>
</div>
<input id=f placeholder="filter by name" oninput=flt()>
<table id=t><thead><tr><th>skill (new name)</th><th>questions</th><th>old name-fit</th><th>new name-fit</th><th>belong together</th><th>status</th><th>benchmarks</th><th>examples</th></tr></thead><tbody>
{''.join(trs)}
</tbody></table></div>
<script>
let cur='ok';
function tog(b){{const e=b.closest('tr').nextElementSibling;e.hidden=!e.hidden;b.textContent=e.hidden?'show':'hide';}}
function setf(b){{cur=b.dataset.f;for(const c of document.querySelectorAll('.tab'))c.classList.toggle('on',c===b);flt();}}
function flt(){{const q=document.getElementById('f').value.toLowerCase();
for(const r of document.querySelectorAll('#t tbody tr.main')){{const on=(r.dataset.st===cur)&&r.cells[0].textContent.toLowerCase().includes(q);
r.style.display=on?'':'none';const e=r.nextElementSibling;e.style.display=on?'':'none';if(!on){{e.hidden=true;r.querySelector('.exb').textContent='show';}}}}}}
flt();
</script>"""
    open("oldtax_relabeled_review.html", "w").write(doc)
    print(f"wrote oldtax_relabeled_review.html")


if __name__ == "__main__":
    main()
