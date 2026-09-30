#!/usr/bin/env python3
"""Score a hand-labelled Part A/Part B gold sheet against its answer key.

This existed only as an ad-hoc computation. The numbers it produced are quoted in
the project log 2026-09-09 (74% acceptance, 99% decoy rejection, the acceptance-by-
size table) and in the task list T-110, but no script in the repo produced them, so
under N5 they were prose without a traceable producer. This is that producer, and
it reproduces every one of those figures from the filed sheet.

Three quantities, and the third is the only one that answers T-110.

ACCEPTANCE     ticks on entries the pipeline actually assigned, over all such
               entries. This is "how often the taxonomy's answer survives a
               human looking at it".
DECOY RATE     ticks on entries drawn from unrelated codes. The null control: if
               it is not near zero, acceptance is measuring agreeableness.
SEPARATION     acceptance minus decoy rate. The headline, because acceptance
               alone cannot be told apart from ticking everything.

BY SIZE        acceptance split by how many questions the skill holds, which is
               what the floor argument turns on. Sizes come from --labels, so
               --codebook and --labels must be the pair the sheet was rendered
               against; passing a later label file silently re-bands every skill.

    python3 tools/gold_score_partb.py                      # reproduces sheet 1
    python3 tools/gold_score_partb.py --sheet gold/gold_sheet_2_labelled.md \
        --key gold/gold_partB_key_2.json \
        --codebook codebook_v2_amended_b150.json --labels item_labels_b150.jsonl
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.metrics import row_codes

P = REPO / "cdm_exploration/experiments/pipeline_v7"
GOLD = REPO / "gold"
BANDS = [(1, 9, "1-9"), (10, 19, "10-19"), (20, 49, "20-49"),
         (50, 199, "50-199"), (200, 10 ** 9, "200+")]
FLOOR = 20


def band(n):
    for lo, hi, lab in BANDS:
        if lo <= n <= hi:
            return lab
    return "0"


def parse_sheet(path):
    """Ticked ids per item, plus which ids the sheet actually showed.

    The shown list is returned so it can be checked against the key. If the sheet
    was regenerated after labelling, the options move and the ticks would be
    scored against a list the labeller never saw.
    """
    txt = path.read_text()
    ticks, shown, anchored = defaultdict(set), defaultdict(set), set()
    for blk in re.split(r"^### ", txt, flags=re.M)[1:]:
        m = re.search(r"item (\d+)", blk)
        if not m:
            continue
        i = m.group(1)
        for mm in re.finditer(r"^- \[([ xX])\] `([^`]+)`", blk, flags=re.M):
            shown[i].add(mm.group(2))
            if mm.group(1).lower() == "x":
                ticks[i].add(mm.group(2))
        # Part A is meant to be written before Part B is read. An answer that
        # cites a code id proves Part B was already visible for that question.
        parta = blk.split("**Part B**")[0]
        if re.search(r"c_\d{3,}", parta):
            anchored.add(i)
    return ticks, shown, anchored


def skill_sizes(codebook, labels):
    cb = json.loads((P / codebook).read_text())
    alias = cb.get("alias", {})
    cnt = Counter()
    for line in (P / labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        for c in row_codes(r, alias):          # deduped: the labeller repeats codes
            if c in cb["codes"]:
                cnt[c] += 1
    live = [c for c in cb["codes"] if c not in alias]
    return cb, cnt, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="gold/gold_sheet_labelled.md")
    ap.add_argument("--key", default="gold/gold_partB_key.json")
    ap.add_argument("--codebook", default="codebook_v9_double_judged.json",
                    help="must be the codebook the sheet was rendered against")
    ap.add_argument("--labels", default="item_labels_codebook_v9_double_judged.jsonl",
                    help="must match --codebook; skill sizes are counted from it")
    ap.add_argument("--out", default=None, help="defaults to <sheet>_score.json")
    a = ap.parse_args()

    sheet = REPO / a.sheet
    key = json.loads((REPO / a.key).read_text())["by_item"]
    ticks, shown, anchored = parse_sheet(sheet)
    cb, cnt, live = skill_sizes(a.codebook, a.labels)

    # the key's ids must belong to the codebook being scored, or every size band
    # is wrong and the acceptance rate describes a different taxonomy
    allids = {c for v in key.values() for c in v["shown"]}
    unknown = allids - set(cb["codes"])
    if unknown:
        raise SystemExit(
            f"{len(unknown)} of {len(allids)} ids in {a.key} are not in {a.codebook}. "
            f"The sheet was rendered against a different taxonomy (code ids restart at "
            f"c_0001 every run). Pass the matching --codebook/--labels.")
    drift = [i for i in key if shown.get(i, set()) != set(key[i]["shown"])]
    if drift:
        raise SystemExit(
            f"{len(drift)} items show different options than the key records. The sheet "
            f"was regenerated after labelling, so the ticks no longer line up.")

    n_asg = sum(len(v["assigned"]) for v in key.values())
    n_dec = sum(len(v["decoys"]) for v in key.values())
    ta = sum(1 for i, v in key.items() for c in v["assigned"] if c in ticks.get(i, ()))
    td = sum(1 for i, v in key.items() for c in v["decoys"] if c in ticks.get(i, ()))
    acc, dec = ta / max(1, n_asg), td / max(1, n_dec)

    agg = defaultdict(lambda: [0, 0])
    for i, v in key.items():
        for c in v["assigned"]:
            b = agg[band(cnt.get(c, 0))]
            b[0] += 1
            b[1] += (c in ticks.get(i, ()))

    print(f"sheet: {a.sheet}  key: {a.key}")
    print(f"codebook: {a.codebook} ({len(live)} live)  labels: {a.labels}")
    print(f"items: {len(key)}  |  Part A answers citing a code id (anchored): "
          f"{len(anchored)} of {len(key)}")
    print(f"\naccepted pipeline assignments : {ta}/{n_asg} = {acc:.1%}")
    print(f"ticked decoys                 : {td}/{n_dec} = {dec:.1%}  "
          f"(rejected {n_dec-td}/{n_dec} = {1-dec:.0%})")
    print(f"separation                    : {acc-dec:+.2f}")

    print(f"\nacceptance by skill size:")
    print(f"  {'skill holds':12s} {'shown':>6} {'accepted':>9}")
    for _, _, lab in BANDS:
        n, k = agg[lab]
        print(f"  {lab:12s} {n:>6} {f'{k/n:.0%}' if n else '-':>9}")
    und = [sum(agg[l][j] for l in ("1-9", "10-19")) for j in (0, 1)]
    atv = [sum(agg[l][j] for l in ("20-49", "50-199", "200+")) for j in (0, 1)]
    print(f"\n  below the floor of {FLOOR}: {und[1]}/{und[0]} = "
          f"{und[1]/und[0]:.0%}" if und[0] else "  below the floor: no assignments")
    print(f"  at or above          : {atv[1]}/{atv[0]} = "
          f"{atv[1]/atv[0]:.0%}" if atv[0] else "  at or above: no assignments")
    if und[0] < 20:
        print(f"\n  WARNING: only {und[0]} assignments come from skills below the floor. "
              f"That is too few to\n  settle the floor argument; the split above is "
              f"descriptive, not evidence. (the task list T-110)")

    out = Path(a.out) if a.out else (REPO / a.sheet).with_name(
        Path(a.sheet).stem + "_score.json")
    out.write_text(json.dumps(
        {"sheet": a.sheet, "key": a.key, "codebook": a.codebook, "labels": a.labels,
         "items": len(key), "part_a_anchored": sorted(anchored),
         "assigned_shown": n_asg, "assigned_ticked": ta, "acceptance": acc,
         "decoys_shown": n_dec, "decoys_ticked": td, "decoy_rate": dec,
         "separation": acc - dec,
         "by_size": {lab: {"shown": agg[lab][0], "accepted": agg[lab][1]}
                     for _, _, lab in BANDS},
         "below_floor": {"shown": und[0], "accepted": und[1]},
         "at_or_above": {"shown": atv[0], "accepted": atv[1]}}, indent=1))
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out                  # --out may legitimately point outside the repo
    print(f"\nwrote {shown}")


if __name__ == "__main__":
    main()
