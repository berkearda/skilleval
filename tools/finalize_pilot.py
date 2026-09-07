"""Finalize the pilot taxonomy: K=80 -> consolidate near-duplicates ->
min-size merge the tail -> measure discrimination -> emit cards + a browser.

Outputs:
  final_taxonomy_cards.txt   (readable, with per-skill reliability)
  final_taxonomy.html        (self-contained browser)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering

DATA = Path("cdm_exploration/data/cdm_ready")
T_CONS = 0.55      # centroid-merge threshold (consolidate near-duplicates)
MIN_SIZE = 15      # merge clusters below this into nearest larger


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA / "skills_stepwise_pilot_B.json"))
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--tcons", type=float, default=T_CONS)
    ap.add_argument("--minsize", type=int, default=MIN_SIZE)
    ap.add_argument("--prefix", default="final_taxonomy")
    A = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    recs = json.load(open(A.input))
    R = np.load(DATA / "response_matrix_v2_full.npy")
    full = {r["item_idx"]: r["question_full_text"]
            for r in json.load(open(DATA / "item_full_text_recovered.json"))}
    subt = {r["item_idx"]: r["subtask"] for r in recs}
    bench = {r["item_idx"]: r["benchmark"] for r in recs}
    per_item = {r["item_idx"]: [s.lower() for s in (r.get("atomic_skills") or [])] for r in recs}
    item_disp = {r["item_idx"]: {"a": r.get("atomic_skills", []),
                 "st": [[s.get("step", "")[:170], s.get("skill", "")] for s in r.get("reasoning_steps", [])],
                 "o": r.get("overall_skill", "")} for r in recs}
    uniq = sorted({s for ss in per_item.values() for s in ss})
    emb = SentenceTransformer("all-mpnet-base-v2").encode(uniq, normalize_embeddings=True,
                                                          show_progress_bar=False, batch_size=256)
    freq = Counter(s for ss in per_item.values() for s in ss)
    lab = AgglomerativeClustering(n_clusters=A.k, metric="cosine", linkage="average").fit_predict(emb)

    # consolidate the 80 cluster centroids
    cent = np.zeros((A.k, emb.shape[1]))
    for c in range(A.k):
        cent[c] = emb[[j for j in range(len(uniq)) if lab[j] == c]].mean(0)
    cent /= np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9, None)
    meta = AgglomerativeClustering(n_clusters=None, distance_threshold=A.tcons,
                                   metric="cosine", linkage="average").fit_predict(cent)
    skill2c = {uniq[j]: int(meta[lab[j]]) for j in range(len(uniq))}

    def items_of(assign):
        ci = {}
        for i, ss in per_item.items():
            for c in {assign[s] for s in ss if s in assign}:
                ci.setdefault(c, []).append(i)
        return ci

    # min-size merge: smallest cluster -> nearest other by skill-centroid cosine
    def centroid(c, assign):
        sk = [j for j in range(len(uniq)) if assign.get(uniq[j]) == c]
        v = emb[sk].mean(0); return v / max(np.linalg.norm(v), 1e-9)
    assign = dict(skill2c)
    while True:
        ci = items_of(assign)
        small = [c for c, it in ci.items() if len(it) < A.minsize]
        if not small:
            break
        c = min(small, key=lambda c: len(ci[c]))
        cents = {o: centroid(o, assign) for o in ci if o != c}
        cc = centroid(c, assign)
        tgt = max(cents, key=lambda o: float(cc @ cents[o]))
        for s in list(assign):
            if assign[s] == c:
                assign[s] = tgt

    ci = items_of(assign)
    # discrimination
    overall = R.mean(1)
    def resid(y, x): a, b = np.polyfit(x, y, 1); return y - (a * x + b)
    def corr(a, b): return 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
    rng = np.random.RandomState(0)
    def rel(items):
        idx = np.array(items, int)
        if len(idx) < 20:
            return None
        return float(np.mean([corr(resid(R[:, idx[p[:len(idx)//2]]].mean(1), overall),
                                    resid(R[:, idx[p[len(idx)//2:]]].mean(1), overall))
                              for p in (rng.permutation(len(idx)) for _ in range(10))]))

    clusters = []
    for c, items in ci.items():
        skl = Counter(s for i in items for s in per_item[i] if assign.get(s) == c)
        name = skl.most_common(1)[0][0] if skl else "?"
        clusters.append({"name": name.replace("_", " "), "size": len(items),
                         "nsub": len({subt[i] for i in items}),
                         "rel": rel(items),
                         "subs": Counter(subt[i] for i in items).most_common(),
                         "skills": skl.most_common(30), "items": items})
    clusters.sort(key=lambda d: -d["size"])

    # cards txt
    L = [f"FINAL PILOT TAXONOMY  ({len(clusters)} skills)  consolidate T={T_CONS}, min-size={MIN_SIZE}",
         "reliability = split-half diagnostic signal beyond general ability (higher=better)", ""]
    for c in clusters:
        rr = f"{c['rel']:.3f}" if c["rel"] is not None else "  -  "
        L.append(f"### {c['name']}  | {c['size']} items | reliab {rr} | {c['nsub']} subtasks: "
                 + ", ".join(f"{s}:{v}" for s, v in c["subs"][:5]))
        L.append("   merged skills: " + ", ".join(f"{s}({v})" for s, v in c["skills"][:10]))
    Path(A.prefix+"_cards.txt").write_text("\n".join(L))

    # html
    items_meta = {i: {"b": bench[i], "s": subt[i], "q": " ".join(full.get(i, "").split())[:1400]}
                  for i in per_item}
    data = {"clusters": [{**{k: c[k] for k in ("name", "size", "nsub", "rel", "subs", "skills")},
                          "items": c["items"]} for c in clusters],
            "items": items_meta, "disp": item_disp}
    Path(A.prefix+".html").write_text(HTML.replace("/*DATA*/", json.dumps(data)))
    meas = [c["rel"] for c in clusters if c["rel"] is not None]
    print(f"final clusters: {len(clusters)} | measurable: {len(meas)} | "
          f"reliability median {np.median(meas):.2f}, min {min(meas):.2f}")
    print(f"clusters <20 items (unmeasured): {sum(1 for c in clusters if c['rel'] is None)}")
    print("wrote "+A.prefix+"_cards.txt, "+A.prefix+".html")
    return 0


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Final Pilot Taxonomy</title><style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;background:#f8fafc}
header{background:#fff;border-bottom:1px solid #e2e8f0;padding:11px 18px;position:sticky;top:0;display:flex;gap:14px;align-items:center}
b.t{font-size:16px}#q{flex:1;max-width:420px;padding:8px 12px;border:1px solid #e2e8f0;border-radius:8px}
.wrap{display:flex;height:calc(100vh - 52px)}#side{width:380px;border-right:1px solid #e2e8f0;overflow:auto;background:#fff}
.row{padding:10px 14px;border-bottom:1px solid #eef2f7;cursor:pointer;display:flex;gap:10px;align-items:center}
.row:hover,.row.on{background:#eef2ff}.row .nm{font-weight:600;font-size:13.5px}.row .m{color:#64748b;font-size:12px}
.badge{font-size:11px;padding:2px 7px;border-radius:999px;color:#fff;flex:none}
#main{flex:1;overflow:auto;padding:20px 26px;max-width:980px}h2{margin:0 0 3px}
.pill{display:inline-block;background:#f1f5f9;border-radius:999px;padding:2px 9px;font-size:11px;margin:2px 4px 2px 0;color:#334155}
.sec{font-size:11px;text-transform:uppercase;color:#64748b;letter-spacing:.05em;margin:16px 0 6px;font-weight:700}
.skl span{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:6px;padding:2px 9px;margin:3px 4px 0 0;font-size:12px}
.card{border:1px solid #e2e8f0;border-radius:10px;padding:11px 13px;margin:9px 0;background:#fff}
.card .h{font-size:11px;color:#64748b;text-transform:uppercase}.chip{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;padding:1px 8px;font-size:12px;margin:2px 4px 0 0}
.steps{display:none;margin-top:7px;border-top:1px dashed #e2e8f0;padding-top:6px}.steps.open{display:block}.step{font-size:13px;margin:2px 0;color:#334155}.step b{color:#4f46e5}.tog{cursor:pointer;color:#4f46e5;font-size:12px;font-weight:600}
</style></head><body><header><b class="t">Final Pilot Taxonomy</b><span id="st" style="color:#64748b;font-size:12px"></span><input id="q" placeholder="search..."></header>
<div class="wrap"><div id="side"></div><div id="main"></div></div><script>
const D=/*DATA*/;let f="";
const relc=r=>r==null?'#94a3b8':r>=0.5?'#16a34a':r>=0.25?'#d97706':'#dc2626';
const relt=r=>r==null?'n/a':r.toFixed(2);
const esc=s=>(s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const side=document.getElementById('side'),main=document.getElementById('main');
document.getElementById('st').textContent=D.clusters.length+' skills · color = diagnostic reliability';
function rs(){side.innerHTML="";D.clusters.forEach((c,i)=>{
 if(f && !(c.name.includes(f)||c.skills.some(s=>s[0].includes(f))))return;
 const d=document.createElement('div');d.className='row';d.dataset.i=i;
 d.innerHTML=`<span class="badge" style="background:${relc(c.rel)}">${relt(c.rel)}</span>
   <div><div class="nm">${esc(c.name)}</div><div class="m">${c.size} q · ${c.nsub} subtasks</div></div>`;
 d.onclick=()=>show(i);side.appendChild(d);});}
function show(i){document.querySelectorAll('.row').forEach(e=>e.classList.toggle('on',e.dataset.i==i));
 const c=D.clusters[i];let its=c.items.map(x=>({i:x,...D.items[x]}));
 if(f)its=its.filter(it=>it.q.toLowerCase().includes(f)||D.disp[it.i].a.join(' ').toLowerCase().includes(f));
 let h=`<h2>${esc(c.name)}</h2><div style="color:#64748b;margin-bottom:6px">${c.size} questions · ${c.nsub} subtasks · diagnostic reliability <b style="color:${relc(c.rel)}">${relt(c.rel)}</b></div>`;
 h+=`<div class="sec">subtasks</div>`+c.subs.map(s=>`<span class="pill">${s[0]}: ${s[1]}</span>`).join('');
 h+=`<div class="sec">merged skill names (count)</div><div class="skl">`+c.skills.map(s=>`<span>${esc(s[0])} · ${s[1]}</span>`).join('')+`</div>`;
 h+=`<div class="sec">${its.length} questions</div>`;
 its.slice(0,250).forEach(it=>{const s=D.disp[it.i];
  h+=`<div class="card"><div class="h">${it.b} / ${it.s} · #${it.i}</div><div style="margin:5px 0">${esc(it.q)}</div>`+
   s.a.map(x=>`<span class="chip">${esc(x)}</span>`).join('')+
   ` <span class="tog" onclick="this.nextElementSibling.classList.toggle('open')">▾ steps</span><div class="steps">`+
   s.st.map(p=>`<div class="step">${esc(p[0])} <b>→ ${esc(p[1])}</b></div>`).join('')+
   `<div class="step" style="color:#64748b;margin-top:4px">overall: <b>${esc(s.o)}</b></div></div></div>`;});
 main.innerHTML=h;main.scrollTop=0;}
document.getElementById('q').oninput=e=>{f=e.target.value.toLowerCase().trim();rs();};
rs();show(0);
</script></body></html>"""

if __name__ == "__main__":
    raise SystemExit(main())
