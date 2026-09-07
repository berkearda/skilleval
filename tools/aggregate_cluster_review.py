"""Aggregate the 4 blind hand-reviews (A-D) of the 100 old-taxonomy clusters:
headline counts, cross-check vs the judge's validity, and a review HTML.

    python tools/aggregate_cluster_review.py
"""
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

REV = Path("/private/tmp/claude-501/-Users-berkearda-Desktop-cdmeval/1de56163-6f9a-40d8-9c62-dc65a71bce0c/scratchpad/reviews")
REPO = Path(__file__).resolve().parent.parent
E = REPO / "cdm_exploration/experiments"

rows = []
for g in "ABCD":
    f = REV / f"group{g}.json"
    if f.exists():
        rows += json.load(open(f))
print(f"loaded {len(rows)} cluster reviews")

# validity per original cluster (orig names run) and per renamed (renamed run)
vb = {c["skill"]: c["validity"] for c in json.load(open(E / "oldtax_percluster_v3.json"))["per_cluster"]}
ren = json.load(open(E / "oldtax_full_renamed.json"))["rename_map"]
va_by_new = {c["skill"]: c["validity"] for c in json.load(open(E / "oldtax_renamed_percluster.json"))["per_cluster"]}

cv = Counter(r["cluster_verdict"] for r in rows)
on = Counter(r["orig_name"] for r in rows)
rn = Counter(r["renamed"] for r in rows)
good = sum(cv[k] for k in ("coherent", "coherent_minor_contamination"))
bad = sum(cv[k] for k in ("two_groups", "multi_group_junk"))
print("\n=== cluster quality (blind hand-review) ===")
for k in ("coherent", "coherent_minor_contamination", "two_groups", "multi_group_junk"):
    print(f"  {k:28s} {cv[k]:3d}")
print(f"  --> good clusters: {good}/{len(rows)}   not-good (splittable): {bad}/{len(rows)}")
print(f"  estimated total subgroups if all split: {sum(r['subgroups'] for r in rows)}")

print("\n=== original name quality ===")
for k in ("accurate", "too_specific", "wrong"):
    print(f"  {k:14s} {on[k]:3d}")
print(f"  --> original name NOT accurate: {on['too_specific']+on['wrong']}/{len(rows)}")
print("\n=== regenerated name quality ===")
for k in ("accurate", "too_vague", "topic_not_skill", "wrong"):
    print(f"  {k:16s} {rn[k]:3d}")
print(f"  --> regenerated name NOT accurate: {len(rows)-rn['accurate']}/{len(rows)}")

# cross-check: do blind 'junk' verdicts line up with low original validity?
print("\n=== cross-check vs judge validity (original names) ===")
gv = defaultdict(list)
for r in rows:
    v = vb.get(r["original_name"])
    if v is not None:
        gv[r["cluster_verdict"]].append(v)
for k in ("coherent", "coherent_minor_contamination", "two_groups", "multi_group_junk"):
    xs = gv[k]
    if xs:
        print(f"  {k:28s} mean orig-validity {sum(xs)/len(xs):.0%}  (n={len(xs)})")

# renamed 'junk' still scoring high = the coarseness/gaming signal
print("\n=== renamed validity of clusters the reviewers call junk ===")
jr = [va_by_new.get(ren.get(r["original_name"], "")) for r in rows if r["cluster_verdict"] == "multi_group_junk"]
jr = [x for x in jr if x is not None]
if jr:
    print(f"  multi_group_junk clusters: mean RENAMED validity {sum(jr)/len(jr):.0%} (n={len(jr)})"
          f"  <- if high, validity is gamed by vague names")

# ---- HTML ----
VC = {"coherent": "#bfe3c6", "coherent_minor_contamination": "#e8eec2",
      "two_groups": "#f3dada", "multi_group_junk": "#e6b8b8"}
rows.sort(key=lambda r: (r["cluster_verdict"] != "multi_group_junk",
                         r["cluster_verdict"] != "two_groups", -r["subgroups"]))
trs = []
for k, r in enumerate(rows, 1):
    ov = vb.get(r["original_name"])
    trs.append(
        f"<tr><td class=n>{k}</td>"
        f"<td class=old>{html.escape(r['original_name'])}</td>"
        f"<td class=new>{html.escape(ren.get(r['original_name'],''))}</td>"
        f"<td style='background:{VC.get(r['cluster_verdict'],'#eee')}'>{r['cluster_verdict']}</td>"
        f"<td class=n>{r['subgroups']}</td>"
        f"<td>{r['orig_name']}</td><td>{r['renamed']}</td>"
        f"<td class=n>{'' if ov is None else round(100*ov)}</td>"
        f"<td class=sug>{html.escape(r['suggested_name'])}</td>"
        f"<td class=rat>{html.escape(r['rationale'])}</td></tr>")
doc = f"""<!doctype html><meta charset=utf-8><title>100-cluster blind hand-review</title><style>
body{{font:13px/1.45 Georgia,serif;margin:0}}.wrap{{max-width:1400px;margin:0 auto;padding:24px}}
h1{{font-size:19px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5d5d5;padding:5px 7px;vertical-align:top;text-align:left}}
th{{background:#f2f2ef;position:sticky;top:0}}.n{{text-align:right}}.old{{color:#7a3b16;max-width:210px}}
.new{{color:#1a5c1a;max-width:210px}}.sug{{max-width:260px;font-weight:700}}.rat{{max-width:360px;color:#333}}</style>
<div class=wrap><h1>Old taxonomy: blind hand-review of all 100 clusters</h1>
<p>Four independent reviewers read 12 member items per cluster, blind to validity scores. Sorted worst-first.
"orig&nbsp;val" = judge validity under the original name (per-cluster run).</p>
<table><tr><th>#</th><th>original name</th><th>regenerated name</th><th>cluster verdict</th><th>sub-groups</th>
<th>orig name</th><th>renamed</th><th>orig val</th><th>suggested name</th><th>rationale</th></tr>
{''.join(trs)}</table></div>"""
out = REPO / "cluster_review_100.html"
out.write_text(doc)
print(f"\nwrote {out}")
