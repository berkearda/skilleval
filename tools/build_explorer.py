"""Build a polished single-file HTML explorer for the step-wise taxonomy.

Embeds BOTH extraction variants (A current, B tightened), each clustered at
K=30 and K=80, plus every question with its reasoning-steps. Features in the
UI: variant + K toggles, Clusters/Questions views, live search, per-cluster
health badges, and flag-and-export for error review.

Output: taxonomy_explorer.html  (open in any browser, no server).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

DATA = Path("cdm_exploration/data/cdm_ready")
VARIANTS = {"A (current)": DATA / "skills_stepwise_pilot.json",
            "B (tightened)": DATA / "skills_stepwise_pilot_B.json"}
KS = [30, 80]
OUT = Path("taxonomy_explorer.html")


def build_variant(path, embedder, items_meta):
    recs = json.load(open(path))
    per_item = {r["item_idx"]: [s.lower() for s in (r.get("atomic_skills") or [])] for r in recs}
    item_skills = {r["item_idx"]: {
        "a": r.get("atomic_skills", []),
        "st": [[s.get("step", "")[:170], s.get("skill", "")] for s in r.get("reasoning_steps", [])],
        "o": r.get("overall_skill", ""),
    } for r in recs}
    uniq = sorted({s for ss in per_item.values() for s in ss})
    emb = embedder.encode(uniq, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    freq = Counter(s for ss in per_item.values() for s in ss)
    ks = {}
    for K in KS:
        lab = AgglomerativeClustering(n_clusters=K, metric="cosine",
                                      linkage="average").fit_predict(emb)
        p2c = {uniq[j]: int(lab[j]) for j in range(len(uniq))}
        clusters = []
        for c in range(K):
            members = [i for i in per_item if any(p2c.get(s) == c for s in per_item[i])]
            if not members:
                continue
            subs = Counter(items_meta[i]["s"] for i in members)
            skl = Counter(s for i in members for s in per_item[i] if p2c.get(s) == c)
            name = max((uniq[j] for j in range(len(uniq)) if lab[j] == c),
                       key=lambda s: freq.get(s, 0), default="?")
            clusters.append({"name": name.replace("_", " "), "size": len(members),
                             "subs": subs.most_common(), "skills": skl.most_common(40),
                             "items": members, "nsub": len(subs)})
        clusters.sort(key=lambda d: -d["size"])
        ks[str(K)] = clusters
    return {"item_skills": item_skills, "ks": ks}


def main() -> int:
    from sentence_transformers import SentenceTransformer
    full = {r["item_idx"]: r["question_full_text"]
            for r in json.load(open(DATA / "item_full_text_recovered.json"))}
    base = json.load(open(list(VARIANTS.values())[0]))
    items_meta = {r["item_idx"]: {"b": r["benchmark"], "s": r["subtask"],
                                  "q": " ".join(full.get(r["item_idx"], "").split())[:1400]}
                  for r in base}
    embedder = SentenceTransformer("all-mpnet-base-v2")
    variants = {}
    for name, p in VARIANTS.items():
        print(f"clustering {name}...", flush=True)
        variants[name] = build_variant(p, embedder, items_meta)
    data = {"items": items_meta, "variants": variants,
            "variantNames": list(VARIANTS.keys()), "ks": [str(k) for k in KS]}
    OUT.write_text(HTML.replace("/*DATA*/", json.dumps(data)))
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    return 0


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillEval Taxonomy Explorer</title>
<style>
:root{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#f8fafc;--acc:#4f46e5;--accbg:#eef2ff;
--ok:#16a34a;--warn:#d97706;--bad:#dc2626}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:#fff;border-bottom:1px solid var(--line);padding:10px 18px;position:sticky;top:0;z-index:10;
display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.title{font-weight:700;font-size:16px;letter-spacing:-.01em}
.title small{color:var(--mut);font-weight:500;margin-left:8px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;background:#fff;padding:6px 12px;cursor:pointer;font-size:13px;color:var(--mut)}
.seg button.on{background:var(--acc);color:#fff}
label.lab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-right:4px}
#q{flex:1;min-width:220px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;font-size:14px}
.stats{color:var(--mut);font-size:12px}
.wrap{display:flex;height:calc(100vh - 53px)}
#side{width:360px;min-width:360px;border-right:1px solid var(--line);overflow:auto;background:#fff}
.row{padding:10px 14px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:10px;align-items:flex-start}
.row:hover{background:var(--accbg)} .row.on{background:var(--accbg);box-shadow:inset 3px 0 0 var(--acc)}
.dot{width:9px;height:9px;border-radius:50%;margin-top:6px;flex:none}
.row .nm{font-weight:600;font-size:13.5px} .row .sub{color:var(--mut);font-size:12px;margin-top:2px}
.flagbtn{margin-left:auto;border:0;background:transparent;cursor:pointer;font-size:15px;opacity:.3}
.flagbtn.on{opacity:1}
#main{flex:1;overflow:auto;padding:20px 26px;max-width:1000px}
h2{margin:0 0 2px;font-size:20px;letter-spacing:-.01em}
.meta{color:var(--mut);font-size:13px;margin-bottom:14px}
.pill{display:inline-block;border-radius:999px;padding:2px 9px;font-size:11px;margin:2px 4px 2px 0;background:#f1f5f9;color:#334155}
.sec{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:18px 0 6px;font-weight:700}
.skl span{display:inline-block;background:var(--accbg);color:#3730a3;border-radius:6px;padding:2px 9px;margin:3px 4px 0 0;font-size:12px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0;background:#fff}
.card .h{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;display:flex;gap:8px;align-items:center}
.card .txt{margin:6px 0 8px}
.chip{display:inline-block;background:#f1f5f9;border:1px solid var(--line);border-radius:6px;padding:1px 8px;font-size:12px;margin:2px 4px 2px 0;color:#334155}
.steps{display:none;margin-top:8px;border-top:1px dashed var(--line);padding-top:7px}
.steps.open{display:block} .step{font-size:13px;margin:3px 0;color:#334155}
.step b{color:var(--acc)} .tog{cursor:pointer;color:var(--acc);font-size:12px;font-weight:600;user-select:none}
.empty{color:var(--mut);padding:40px;text-align:center}
#exp{cursor:pointer;color:var(--acc);font-size:12px;font-weight:600;border:1px solid var(--acc);border-radius:8px;padding:6px 10px;background:#fff}
</style></head><body>
<header>
  <div class="title">SkillEval Taxonomy Explorer<small>step-wise skills</small></div>
  <span><label class="lab">prompt</label><span id="vseg" class="seg"></span></span>
  <span><label class="lab">granularity</label><span id="kseg" class="seg"></span></span>
  <span><label class="lab">view</label><span id="view" class="seg">
    <button data-v="clusters" class="on">Clusters</button><button data-v="questions">Questions</button></span></span>
  <input id="q" placeholder="search skills, questions, clusters...">
  <button id="exp">Export flags</button>
  <span class="stats" id="stats"></span>
</header>
<div class="wrap"><div id="side"></div><div id="main"><div class="empty">Select an item on the left.</div></div></div>
<script>
const D=/*DATA*/;
let V=D.variantNames[0], K=D.ks[0], MODE="clusters", filter="";
const flags=JSON.parse(localStorage.getItem("seflags")||"{}");
const $=s=>document.querySelector(s), side=$("#side"), main=$("#main");
function saveFlags(){localStorage.setItem("seflags",JSON.stringify(flags))}
function fkey(type,id){return V+"|"+K+"|"+type+"|"+id}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function dotColor(nsub){return nsub<=2?'var(--ok)':nsub==3?'var(--warn)':'var(--bad)'}

// build segmented controls
function seg(el,vals,cur,cb){el.innerHTML="";vals.forEach(v=>{const b=document.createElement('button');
  b.textContent=v;if(v==cur)b.className="on";b.onclick=()=>{cb(v);};el.appendChild(b);});}
function reseg(){
  seg($("#vseg"),D.variantNames,V,v=>{V=v;reseg();render();});
  seg($("#kseg"),D.ks,K,k=>{K=k;reseg();render();});
}
reseg();
document.querySelectorAll("#view button").forEach(b=>b.onclick=()=>{
  MODE=b.dataset.v;document.querySelectorAll("#view button").forEach(x=>x.classList.toggle("on",x==b));render();});
$("#q").oninput=e=>{filter=e.target.value.toLowerCase().trim();render();};
$("#exp").onclick=()=>{
  const lines=Object.keys(flags).filter(k=>flags[k]).map(k=>k);
  const blob=new Blob([lines.join("\n")||"(no flags)"],{type:"text/plain"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="skilleval_flags.txt";a.click();};

function clusters(){return D.variants[V].ks[K]}
function render(){
  const cs=clusters();
  $("#stats").textContent=`${Object.keys(D.items).length} questions · ${cs.length} clusters · ${Object.values(flags).filter(Boolean).length} flagged`;
  if(MODE=="clusters")renderClusters(cs); else renderQuestions();
}
function renderClusters(cs){
  side.innerHTML="";
  cs.forEach((c,i)=>{
    if(filter && !(c.name.includes(filter)||c.skills.some(s=>s[0].includes(filter)))) return;
    const d=document.createElement('div');d.className="row";d.dataset.i=i;
    const fk=fkey("cluster",c.name);
    d.innerHTML=`<span class="dot" style="background:${dotColor(c.nsub)}"></span>
      <div><div class="nm">${esc(c.name)}</div>
      <div class="sub">${c.size} q · ${c.nsub} subtasks · ${c.subs.slice(0,2).map(s=>s[0]).join(', ')}</div></div>
      <button class="flagbtn ${flags[fk]?'on':''}" title="flag">⚑</button>`;
    d.querySelector('.flagbtn').onclick=ev=>{ev.stopPropagation();flags[fk]=!flags[fk];saveFlags();ev.target.classList.toggle('on');render();};
    d.onclick=()=>showCluster(i);side.appendChild(d);
  });
  if(!side.children.length)side.innerHTML='<div class="empty">no matches</div>';
}
function showCluster(i){
  document.querySelectorAll('.row').forEach(e=>e.classList.toggle('on',e.dataset.i==i));
  const c=clusters()[i], IS=D.variants[V].item_skills;
  let its=c.items.map(x=>({idx:x,...D.items[x]}));
  if(filter)its=its.filter(it=>it.q.toLowerCase().includes(filter)||(IS[it.idx].a.join(' ').toLowerCase().includes(filter)));
  let h=`<h2>${esc(c.name)}</h2><div class="meta">${c.size} questions · ${c.nsub} subtasks `+
    `<span class="dot" style="display:inline-block;background:${dotColor(c.nsub)}"></span></div>`;
  h+=`<div class="sec">subtasks</div>`+c.subs.map(s=>`<span class="pill">${s[0]}: ${s[1]}</span>`).join('');
  h+=`<div class="sec">distinct skill names merged here (count)</div><div class="skl">`+
    c.skills.map(s=>`<span>${esc(s[0])} · ${s[1]}</span>`).join('')+`</div>`;
  h+=`<div class="sec">${its.length} questions</div>`;
  its.slice(0,300).forEach(it=>{const s=IS[it.idx], fk=fkey("q",it.idx);
    h+=`<div class="card"><div class="h">${it.b} / ${it.s} · #${it.idx}
      <button class="flagbtn ${flags[fk]?'on':''}" onclick="toggleFlag('${fk}',this)">⚑</button></div>
      <div class="txt">${esc(it.q)}</div>`+
      s.a.map(x=>`<span class="chip">${esc(x)}</span>`).join('')+
      ` <span class="tog" onclick="this.nextElementSibling.classList.toggle('open')">▾ steps</span>
      <div class="steps">`+s.st.map(p=>`<div class="step">${esc(p[0])} <b>→ ${esc(p[1])}</b></div>`).join('')+
      `<div class="step" style="margin-top:5px;color:#64748b">overall: <b>${esc(s.o)}</b></div></div></div>`;});
  main.innerHTML=h;main.scrollTop=0;
}
function renderQuestions(){
  const IS=D.variants[V].item_skills;
  side.innerHTML="";
  let ids=Object.keys(D.items);
  if(filter)ids=ids.filter(i=>D.items[i].q.toLowerCase().includes(filter)||IS[i].a.join(' ').toLowerCase().includes(filter));
  ids.slice(0,600).forEach(i=>{const it=D.items[i];
    const d=document.createElement('div');d.className="row";
    d.innerHTML=`<div><div class="nm">#${i} · ${it.s}</div><div class="sub">${esc(it.q.slice(0,70))}</div></div>`;
    d.onclick=()=>showQuestion(i);side.appendChild(d);});
  if(!side.children.length)side.innerHTML='<div class="empty">no matches</div>';
}
function showQuestion(i){
  const it=D.items[i], s=D.variants[V].item_skills[i];
  let h=`<h2>#${i} <span style="font-size:14px;color:#64748b">${it.b} / ${it.s}</span></h2>`;
  h+=`<div class="card"><div class="txt">${esc(it.q)}</div>`+
    s.a.map(x=>`<span class="chip">${esc(x)}</span>`).join('')+
    `<div class="sec">reasoning steps → skill</div>`+
    s.st.map(p=>`<div class="step">${esc(p[0])} <b>→ ${esc(p[1])}</b></div>`).join('')+
    `<div class="step" style="margin-top:6px;color:#64748b">overall: <b>${esc(s.o)}</b></div></div>`;
  main.innerHTML=h;main.scrollTop=0;
}
function toggleFlag(fk,el){flags[fk]=!flags[fk];saveFlags();el.classList.toggle('on');}
render();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
