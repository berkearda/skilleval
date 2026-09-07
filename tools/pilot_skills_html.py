"""Render the pilot skill bank to a clean HTML for manual checking, with the
Part A scorecard (good-skill metrics) and explanations at the top. No reliability
score (that is a separate downstream metric).

    .venv312/bin/python tools/pilot_skills_html.py \
        cdm_exploration/experiments/pilot_merged_drop.json \
        cdm_exploration/experiments/pilot_partA.json pilot_skills.html
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
BANKF = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/pilot_merged_drop.json"
PARTAF = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    REPO / "cdm_exploration/experiments/pilot_partA.json"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "pilot_skills.html"


def pct(x, n):
    return f"{round(100 * x / max(n, 1))}%"


def main():
    data = json.load(open(BANKF))
    pa = json.load(open(PARTAF))
    sc = pa["scorecard"]
    grain = pa["per_skill_grain"]
    dup_skills = {s for pair in pa["duplicate_pairs"] for s in pair}
    defn = {b["name"]: b["definition"] for b in data["bank"]}
    qtext = {r["item_idx"]: r["question_full_text"]
             for r in json.load(open(D / "item_full_text_recovered.json"))}

    sk_items = defaultdict(list)
    sk_bench = defaultdict(Counter)
    item_meta = {}
    for pi in data["per_item"]:
        item_meta[pi["item_idx"]] = (pi["benchmark"], pi["subtask"])
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
            sk_bench[s][pi["benchmark"]] += 1

    GCOL = {"appropriate": "#bfe3c6", "too_narrow": "#e8eec2",
            "too_broad": "#f3dada", "?": "#e9e9e9"}

    rows = []
    for name in defn:
        items = sk_items.get(name, [])
        seen, ex = set(), []
        for i in items:
            b, sub = item_meta[i]
            if b in seen and len(seen) < len(sk_bench[name]):
                continue
            seen.add(b)
            ex.append((b, sub, " ".join(qtext.get(i, "").split())[:260]))
            if len(ex) >= 3:
                break
        rows.append({"name": name, "def": defn[name], "n": len(items),
                     "bench": dict(sk_bench[name]), "grain": grain.get(name, "?"),
                     "dup": name in dup_skills, "ex": ex})
    rows.sort(key=lambda r: r["n"], reverse=True)

    trs = []
    for k, r in enumerate(rows, 1):
        bench = ", ".join(f"{b} {n}" for b, n in sorted(r["bench"].items(),
                                                        key=lambda x: -x[1]))
        exh = "".join(
            f"<div class=ex><span class=eb>{html.escape(b)}/{html.escape(sub)}</span> "
            f"{html.escape(qq)}</div>" for b, sub, qq in r["ex"])
        flag = " <span class=dup title='near-duplicate of another skill'>dup</span>" if r["dup"] else ""
        trs.append(f"""<tr data-n="{r['n']}" data-name="{html.escape(r['name'])}" data-g="{r['grain']}">
<td class=num>{k}</td>
<td class=sk>{html.escape(r['name'])}{flag}</td>
<td class=df>{html.escape(r['def'])}</td>
<td class=num>{r['n']}</td>
<td class=bm>{html.escape(bench)}</td>
<td style="background:{GCOL.get(r['grain'],'#e9e9e9')}">{html.escape(r['grain'].replace('_',' '))}</td>
<td><details><summary>show</summary>{exh}</details></td></tr>""")

    val = sc["validity"]
    gc = sc["granularity"]
    nv = sc["validity_n"]

    K = sc["skills"]
    yes_c, no_c = val.get("yes", 0), val.get("no", 0)
    appr_c = gc.get("appropriate", 0)
    corr_c = round(sc["intruder_detection"] * sc["intruder_n"])
    dup_c, near = sc["near_duplicate"], sc["near_pairs"]
    raw = sc["raw_before"]

    # scorecard rows: (metric, result, plain sentence, math formula)
    score_rows = [
        ("Tagging validity",
         f"yes {pct(yes_c,nv)}, no {pct(no_c,nv)}",
         "A judge reads random item-and-skill pairs and marks whether the item truly needs that "
         "skill.",
         f"validity = (pairs marked yes) / (pairs sampled) = {yes_c} / {nv} = {yes_c/nv:.2f}"),
        ("Granularity",
         f"{pct(appr_c,K)} appropriate",
         "A judge rates each skill as a specific operation, too broad, or too narrow.",
         f"granularity = (skills rated appropriate) / (all skills) = {appr_c} / {K} = {appr_c/K:.2f}"),
        ("Coherence",
         f"{round(100*sc['intruder_detection'])}%",
         "For some skills we add one item from a different skill, and the judge must spot the "
         "outsider.",
         f"coherence = (outsiders found) / (trials) = {corr_c} / {sc['intruder_n']} "
         f"= {corr_c/sc['intruder_n']:.2f}   (chance = 1/5 = 0.20)"),
        ("Distinctiveness",
         f"{pct(dup_c,near)} redundant",
         "Among the most similar skill pairs, how many are really the same skill.",
         f"redundancy = (duplicate pairs) / (closest pairs checked) = {dup_c} / {near} = {dup_c/near:.2f}"),
        ("Size and reduction",
         f"{K} skills ({raw/K:.1f}x)",
         "How compact the taxonomy is versus the raw extraction, with no clustering step.",
         f"reduction = (raw skill names) / (final skills) = {raw} / {K} = {raw/K:.1f}"),
        ("Coverage",
         f"{pct(int(sc['coverage']*100),100)}",
         "The share of items that received at least one skill.",
         f"coverage = (items with a skill) / (all items) = {sc['coverage']:.2f}"),
    ]
    score_html = "".join(
        f"<tr><td class=mname>{m}</td><td class=mres>{res}</td>"
        f"<td class=mmean>{sent}<div class=f>{frm}</div></td></tr>"
        for m, res, sent, frm in score_rows)

    # ---- churn section (present when rendering Step 4 output) ----
    churn = data.get("churn")
    churn_html = ""
    if churn:
        cr = [
            ("Labels retained", f"{churn['retained_frac']:.0%} of old labels kept",
             "Fraction of the pre-Step-4 labels that survived the relabel."),
            ("Item sets identical", f"{churn['identical_frac']:.0%} of items",
             "Items whose skill set was completely unchanged by Step 4."),
            ("Mean overlap (Jaccard)", f"{churn['mean_jaccard']:.2f}",
             "Average per-item overlap between old and new skill sets (1 = identical)."),
            ("Skills in use", f"old {churn['used_old']} to new {churn['used_new']} of {sc['skills']}",
             "Distinct bank skills actually used, before and after the relabel."),
            ("Labels dropped / added", f"{churn['dropped']} dropped, {churn['added']} added",
             "Per-label changes Step 4 made across all items."),
            ("Escape-hatch new skills", f"{churn['escape_new']}",
             "Skills Step 4 had to invent because no candidate fit (should be small)."),
        ]
        rows_h = "".join(f"<tr><td class=mname>{m}</td><td class=mres>{r}</td>"
                         f"<td class=mmean>{x}</td></tr>" for m, r, x in cr)
        churn_html = ("<h2>Step 4 relabel: what changed</h2>"
                      "<p class=note>Step 4 relabels every item against the frozen bank so the labels "
                      "are consistent. This compares the Step 4 labels with the pre-Step-4 labels.</p>"
                      f"<table class=score>{rows_h}</table>")

    # ---- prompts section ----
    PR = {
        "Step 2 and 4: labeling system prompt (reuse from bank, or create new)":
        ("You label a test item's reasoning steps using a bank of cognitive skills. "
         "For each step, pick the candidate skill whose definition best matches the operation "
         "that step requires, and reuse its EXACT name. If genuinely none of the candidates fits, "
         'output "NEW:snake_case_name" with a one-line definition (this should be rare). '
         "Judge by meaning, not wording."),
        "Step 4: per-item user prompt (candidates shown as name + full definition)":
        ('ITEM [benchmark/subtask]\n\nReasoning steps:\n1. <step text>\n2. ...\n\n'
         'Candidate skills (name: definition):\n- isolate_the_variable: rearrange an equation to '
         'get the unknown alone\n- ... (top 24 by similarity) ...\n\n'
         'Return JSON: {"labels": [{"skill": "<exact candidate name OR NEW:new_name>", '
         '"definition": "<one line, ONLY for NEW>"}]} with one entry per step, in order.'),
        "Step 3: merge two skills by meaning":
        ("You decide whether two cognitive-skill labels denote the SAME cognitive operation. "
         "Judge by MEANING, not wording. They are the same only if a reasoning step requiring one "
         "would require the other. Skills from different domains, or different operations, are NOT "
         "the same even if worded similarly (e.g. order_of_operations vs order_adjectives differ). "
         'Input: Skill A (name + definition + example steps), Skill B (same). '
         'Return JSON {"same": true|false}.'),
        "Part A judge: tagging validity":
        ('Item (benchmark/subtask): <question>\nSkill: <name>\nDefinition: <definition>\n\n'
         'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|partial|no"}.'),
        "Part A judge: granularity":
        ('Skill: <name>\nDefinition: <definition>\nExample items: ...\n\n'
         "Rate granularity: 'too_broad' (vague catch-all like logical_reasoning), 'appropriate' "
         "(a specific reusable cognitive operation), or 'too_narrow' (tied to one specific item). "
         'Return JSON {"grain":"too_broad|appropriate|too_narrow"}.'),
        "Part A judge: intruder / coherence":
        ("These items should all require the skill '<name>' (<definition>). Exactly one does NOT. "
         'Which number is the intruder?\n1. <q1>\n...\n5. <q5>\n\nReturn JSON {"intruder": <1-5>}.'),
        "Part A judge: distinctiveness":
        ('Skill A: <name> - <definition>\nSkill B: <name> - <definition>\n\n'
         'Near-duplicates (essentially the same cognitive operation)? Return JSON {"duplicate": true|false}.'),
    }
    prompts_html = "<h2>Prompts used</h2>" + "".join(
        f"<details class=pr><summary>{html.escape(k)}</summary><pre>{html.escape(v)}</pre></details>"
        for k, v in PR.items())

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>SkillEval pilot taxonomy</title><style>
body{{font:15px/1.55 Georgia,'Times New Roman',serif;color:#1a1a1a;margin:0;background:#fff}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 22px 70px}}
h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 6px}}
.sub{{color:#555;font-style:italic;margin:0 0 14px;font-size:13.5px}}
.note{{font-size:13px;color:#444;margin:0 0 12px}}
.f{{font-family:Menlo,monospace;font-size:11.5px;color:#222;background:#f6f6f4;border-left:3px solid #999;padding:4px 8px;margin:5px 0 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{border:1px solid #d6d6d6;padding:6px 8px;text-align:left;vertical-align:top}}
.score td{{vertical-align:top}}
.mname{{font-weight:700;white-space:nowrap;width:150px}}
.mres{{width:200px}} .mmean{{color:#444;font-size:13px}}
code{{font-family:Menlo,monospace;font-size:12px;color:#333}}
#t th{{background:#f2f2ef;position:sticky;top:0;cursor:pointer;font-weight:700}}
#t th:hover{{background:#e7e7e2}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.sk{{font-family:'SF Mono',Menlo,monospace;font-size:12.5px;font-weight:700;white-space:nowrap}}
.dup{{font-family:Georgia,serif;font-weight:400;font-size:10px;color:#a23;border:1px solid #d9a;border-radius:3px;padding:0 3px}}
.df{{color:#333;max-width:300px}} .bm{{font-size:12px;color:#555;white-space:nowrap}}
details summary{{cursor:pointer;color:#0645ad;font-size:12.5px}}
.ex{{margin:6px 0;font-size:12.5px;color:#222;max-width:560px}}
.eb{{font-family:monospace;font-size:11px;color:#777}}
.legend{{font-size:12.5px;color:#555;margin:8px 0 0}}
.chip{{display:inline-block;width:11px;height:11px;border:1px solid #bbb;vertical-align:middle}}
input{{font:14px Georgia,serif;padding:6px 9px;width:280px;border:1px solid #aaa;margin:10px 0 12px}}
pre{{white-space:pre-wrap;word-break:break-word;font:11.5px/1.45 Menlo,monospace;background:#f7f7f4;border:1px solid #ddd;padding:8px 10px;margin:6px 0;color:#222}}
details.pr{{margin:4px 0}} details.pr summary{{font-weight:700;color:#1a1a1a;font-size:13px}}
</style></head><body><div class=wrap>
<h1>SkillEval pilot taxonomy</h1>
<p class=sub>Full pipeline (running bank, meaning-merge, drop rare) on a 998-item, five-benchmark subset. For manual checking.</p>

<h2>Evaluation (by reading, not model scores)</h2>
<table class=score><tr><th>Metric</th><th>Result</th><th>How it is measured</th></tr>{score_html}</table>
<p class=note>Semantic metrics use an LLM judge on samples; human calibration pending.</p>
{prompts_html}

<h2>The {sc['skills']} skills</h2>
<p class=legend>Granularity colour:
<span class=chip style="background:#bfe3c6"></span> appropriate &nbsp;
<span class=chip style="background:#e8eec2"></span> too narrow &nbsp;
<span class=chip style="background:#f3dada"></span> too broad. &nbsp;
<span class=dup>dup</span> marks a near-duplicate pair the merge missed. Click a header to sort, type to filter.</p>
<input id=f placeholder="filter by skill name or definition" oninput="filt()">
<table id=t><thead><tr>
<th onclick="srt(0,'n')">#</th><th onclick="srt(1,'s')">skill</th>
<th onclick="srt(2,'s')">definition</th><th onclick="srt(3,'n')">items</th>
<th onclick="srt(4,'s')">benchmarks</th><th onclick="srt(5,'s')">granularity</th>
<th>examples</th></tr></thead><tbody>
{''.join(trs)}
</tbody></table></div>
<script>
const tb=document.querySelector('#t tbody');
function filt(){{const v=document.getElementById('f').value.toLowerCase();
 for(const r of tb.rows){{const t=(r.dataset.name+' '+r.cells[2].textContent).toLowerCase();
 r.style.display=t.includes(v)?'':'none';}}}}
let asc={{}};
function srt(i,kind){{const rows=[...tb.rows];asc[i]=!asc[i];const s=asc[i]?1:-1;
 rows.sort((a,b)=>{{let x,y;
 if(kind==='n'){{x=+a.dataset.n;y=+b.dataset.n;return s*(x-y);}}
 x=a.cells[i].textContent.toLowerCase();y=b.cells[i].textContent.toLowerCase();
 return s*(x<y?-1:x>y?1:0);}});
 rows.forEach(r=>tb.appendChild(r));}}
</script></body></html>"""
    OUT.write_text(doc)
    print(f"wrote {OUT}  ({len(rows)} skills)")


if __name__ == "__main__":
    main()
