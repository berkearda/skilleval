#!/usr/bin/env python3
"""Emit the Step 2 numbers block for the pipeline run log.

The run log's rule is that its numbers are written by the step's own script,
not typed by hand, so prose and data cannot drift apart. Step 2 printed its
telemetry to stdout and saved a codebook; nothing turned those into the block
the log asks for. This does, reading only the log and the artifact.

    python3 tools/step2_report.py [--codebook final] > block.md
"""
import json, re, sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "cdm_exploration/experiments/pipeline_v7"


def batches(log):
    out = []
    for l in log.open():
        m = re.match(r"\s+batch\s+(\d+):\s+(\d+) mapped,\s+(\d+) new \(\s*([\d.]+)%\),"
                     r"\s+(\d+) carried \| codes\s+(\d+)\s*\|\s*([\d,]+) tok\s*\|\s*(\d+)s", l)
        if m:
            g = m.groups()
            out.append(dict(b=int(g[0]), mapped=int(g[1]), new=int(g[2]), rate=float(g[3]),
                            carried=int(g[4]), codes=int(g[5]),
                            tok=int(g[6].replace(",", "")), sec=int(g[7])))
    return sorted({r["b"]: r for r in out}.values(), key=lambda r: r["b"])


def audits(log):
    out = []
    for l in log.open():
        if "audit v" not in l:
            continue
        ops = dict(re.findall(r"'(\w+)': (\d+)", l))
        out.append(dict(v=int(re.search(r"audit v(\d+)", l).group(1)),
                        ops={k: int(v) for k, v in ops.items()},
                        churn=float(re.search(r"churn ([\d.]+)%", l).group(1)),
                        codes=int(re.search(r"codes (\d+)", l).group(1))))
    return out


def main(tag="final"):
    log = P / "step2_run.log"
    cb = json.loads((P / f"codebook_{tag}.json").read_text())
    B, A = batches(log), audits(log)
    live = [c for c in cb["codes"] if c not in cb["alias"]]
    uses = Counter()
    for lab, cid in cb["assign"].items():
        seen = set()
        while cid in cb["alias"] and cid not in seen:
            seen.add(cid); cid = cb["alias"][cid]
        uses[cid] += 1
    sizes = sorted((uses[c] for c in live), reverse=True)
    stop = next((l.strip() for l in log.open() if l.startswith("stopped:")), "(still running)")
    jo = Counter(j["op"] for j in cb.get("journal", []))

    print(f"**Ran** {len(B)} batches of {BATCH_HINT}, {len(A)} audits. {stop}\n")
    print("| | |\n|---|---|")
    print(f"| live codes | {len(live):,} |")
    print(f"| codes created | {len(cb['codes']):,} |")
    print(f"| merged away | {len(cb['alias']):,} |")
    print(f"| raw labels assigned | {len(cb['assign']):,} |")
    if sizes:
        print(f"| code size, max / median | {sizes[0]} / {sizes[len(sizes)//2]} |")
        print(f"| singleton codes | {sum(1 for s in sizes if s == 1):,} "
              f"({sum(1 for s in sizes if s == 1)/len(sizes):.0%}) |")
    if B:
        print(f"| tokens | {B[-1]['tok']:,} |")
        print(f"| wall time | {B[-1]['sec']/3600:.1f} h |")
    print(f"| audit operations | {dict(jo)} |\n")

    print("New-code rate, mean per ten batches:\n")
    print("| batches | mean new/batch | rate |\n|---|---|---|")
    for lo in range(0, (B[-1]["b"] if B else 0) + 1, 10):
        w = [r["new"] for r in B if lo <= r["b"] < lo + 10]
        if w:
            print(f"| {lo}-{lo+9} | {sum(w)/len(w):.1f} | {sum(w)/len(w)/50:.1%} |")
    print("\nAudits:\n")
    print("| audit | operations | churn | live codes after |\n|---|---|---|---|")
    for a in A:
        print(f"| v{a['v']} | {a['ops']} | {a['churn']:.1f}% | {a['codes']} |")


BATCH_HINT = 50
if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 else "final")
