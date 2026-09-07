"""Build a single self-contained HTML to browse skills / clusters / questions.

No server needed: open the .html in any browser. Left = clusters; click one to
see its merged skill names, subtask mix, and every question (expandable to its
reasoning-steps -> skills). Search box filters across clusters/skills/questions.

Clustering shown: HAC (K configurable) on the step-wise atomic skills.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

DATA = Path("cdm_exploration/data/cdm_ready")
K = 30
OUT = Path("taxonomy_browser.html")


def main() -> int:
    from sentence_transformers import SentenceTransformer
    step = json.load(open(DATA / "skills_stepwise_pilot.json"))
    full = {r["item_idx"]: r["question_full_text"]
            for r in json.load(open(DATA / "item_full_text_recovered.json"))}

    per_item = {r["item_idx"]: [s.lower() for s in (r.get("atomic_skills") or [])]
                for r in step}
    uniq = sorted({s for ss in per_item.values() for s in ss})
    print(f"embedding {len(uniq)} unique skills + clustering K={K}...", flush=True)
    emb = SentenceTransformer("all-mpnet-base-v2").encode(
        uniq, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    lab = AgglomerativeClustering(n_clusters=K, metric="cosine",
                                  linkage="average").fit_predict(emb)
    p2c = {uniq[j]: int(lab[j]) for j in range(len(uniq))}
    gcnt = Counter(s for ss in per_item.values() for s in ss)

    # cluster names = most frequent member skill
    names = {}
    for c in range(K):
        members = [uniq[j] for j in range(len(uniq)) if lab[j] == c]
        names[c] = (max(members, key=lambda s: gcnt.get(s, 0)).replace("_", " ").title()
                    if members else f"cluster {c}")

    # item records
    items = []
    for r in step:
        i = r["item_idx"]
        cs = sorted({p2c[s] for s in per_item[i] if s in p2c})
        items.append({
            "idx": i, "bench": r["benchmark"], "subtask": r["subtask"],
            "q": " ".join(full.get(i, "").split())[:1600],
            "skills": r.get("atomic_skills", []),
            "steps": [{"s": st.get("step", "")[:160], "k": st.get("skill", "")}
                      for st in r.get("reasoning_steps", [])],
            "overall": r.get("overall_skill", ""),
            "clusters": cs,
        })

    # cluster records
    clusters = []
    for c in range(K):
        members = [it for it in items if c in it["clusters"]]
        subs = Counter(it["subtask"] for it in members)
        skl = Counter(s for it in members for s in
                      [x.lower() for x in it["skills"]] if p2c.get(s) == c)
        clusters.append({
            "id": c, "name": names[c], "size": len(members),
            "subtasks": subs.most_common(),
            "skills": skl.most_common(40),
            "items": [it["idx"] for it in members],
        })
    clusters.sort(key=lambda d: -d["size"])

    data = {"k": K, "clusters": clusters,
            "items": {it["idx"]: it for it in items}}
    html = HTML.replace("/*DATA*/", json.dumps(data))
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB, {len(items)} items, {K} clusters)")
    return 0


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>SkillEval taxonomy browser</title>
<style>
*{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a}
header{background:#1f2937;color:#fff;padding:10px 16px;position:sticky;top:0;z-index:5}
header b{font-size:16px} header span{opacity:.7;margin-left:10px;font-size:13px}
#search{margin-left:16px;padding:6px 10px;width:340px;border-radius:6px;border:1px solid #555;background:#374151;color:#fff}
.wrap{display:flex;height:calc(100vh - 46px)}
#side{width:340px;border-right:1px solid #e5e7eb;overflow:auto;background:#f9fafb}
.cl{padding:8px 12px;border-bottom:1px solid #eee;cursor:pointer}
.cl:hover{background:#eef2ff} .cl.active{background:#e0e7ff}
.cl .n{font-weight:600} .cl .m{color:#6b7280;font-size:12px}
#main{flex:1;overflow:auto;padding:18px 24px}
.badge{display:inline-block;background:#e5e7eb;border-radius:10px;padding:1px 8px;font-size:11px;margin:1px 3px 1px 0;color:#374151}
.chip{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:6px;padding:1px 7px;font-size:12px;margin:2px 4px 2px 0}
.q{border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin:10px 0;background:#fff}
.q .sub{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}
.q .txt{margin:4px 0 6px} .steps{display:none;margin-top:8px;border-top:1px dashed #ddd;padding-top:6px}
.steps.open{display:block} .step{font-size:13px;color:#374151;margin:2px 0}
.step b{color:#3730a3;font-weight:600}
.tog{cursor:pointer;color:#4f46e5;font-size:12px;user-select:none}
h2{margin:0 0 4px} .sk{color:#6b7280;font-size:13px;margin:8px 0}
.skl{margin:6px 0 14px} .skl span{display:inline-block;background:#f3f4f6;border-radius:6px;padding:2px 8px;margin:2px;font-size:12px}
</style></head><body>
<header><b>SkillEval taxonomy browser</b><span id="meta"></span>
<input id="search" placeholder="search skills / questions / clusters...">
</header>
<div class="wrap"><div id="side"></div><div id="main"></div></div>
<script>
const D=/*DATA*/;
const side=document.getElementById('side'),main=document.getElementById('main');
document.getElementById('meta').textContent=D.items&&(' '+Object.keys(D.items).length+' questions · '+D.clusters.length+' clusters (HAC K='+D.k+')');
let filter="";
function renderSide(){
  side.innerHTML="";
  D.clusters.forEach((c,i)=>{
    if(filter && !(c.name.toLowerCase().includes(filter) || c.skills.some(s=>s[0].includes(filter)))) return;
    const d=document.createElement('div');d.className='cl';d.dataset.id=c.id;
    d.innerHTML=`<div class="n">${c.name}</div><div class="m">${c.size} questions · `+
      c.subtasks.slice(0,3).map(s=>s[0]+':'+s[1]).join(', ')+`</div>`;
    d.onclick=()=>show(c.id);side.appendChild(d);
  });
}
function show(id){
  document.querySelectorAll('.cl').forEach(e=>e.classList.toggle('active',e.dataset.id==id));
  const c=D.clusters.find(x=>x.id==id);
  let h=`<h2>${c.name}</h2><div class="sk">${c.size} questions · subtasks: `+
    c.subtasks.map(s=>`<span class="badge">${s[0]}:${s[1]}</span>`).join('')+`</div>`;
  h+=`<div class="sk"><b>distinct skill names merged here:</b></div><div class="skl">`+
    c.skills.map(s=>`<span>${s[0]} (${s[1]})</span>`).join('')+`</div>`;
  let its=c.items.map(i=>D.items[i]);
  if(filter) its=its.filter(it=>it.q.toLowerCase().includes(filter)||it.skills.join(' ').toLowerCase().includes(filter));
  h+=`<div class="sk"><b>${its.length} questions</b> (click "steps" to expand)</div>`;
  its.slice(0,400).forEach(it=>{
    h+=`<div class="q"><div class="sub">${it.bench} / ${it.subtask} · #${it.idx}</div>`+
      `<div class="txt">${esc(it.q)}</div>`+
      it.skills.map(s=>`<span class="chip">${s}</span>`).join('')+
      ` <span class="tog" onclick="this.nextElementSibling.classList.toggle('open')">steps ▾</span>`+
      `<div class="steps">`+it.steps.map(s=>`<div class="step">${esc(s.s)} <b>→ ${s.k}</b></div>`).join('')+
      `<div class="step" style="margin-top:4px">overall: <b>${it.overall}</b></div></div></div>`;
  });
  main.innerHTML=h;main.scrollTop=0;
}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
document.getElementById('search').oninput=e=>{filter=e.target.value.toLowerCase().trim();renderSide();};
renderSide();if(D.clusters.length)show(D.clusters[0].id);
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
