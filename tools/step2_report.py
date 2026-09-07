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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    stops = [l.strip() for l in log.open() if l.startswith("stopped:")]
    stop = stops[-1] if stops else "(still running)"   # the last one is the outcome
    jo = Counter(j["op"] for j in cb.get("journal", []))

    print(f"**Ran** {len(B)} batches, {len(A)} audits. {stop}\n")
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
        # counters reset on every process start, so after a --resume the last row
        # is the continuation segment, not the run. Sum the segments instead.
        segs, prev = [], None
        for r in B:
            if prev is None or r["tok"] < prev["tok"]:
                segs.append(r)
            else:
                segs[-1] = r
            prev = r
        print(f"| tokens | {sum(x['tok'] for x in segs):,} |")
        print(f"| wall time | {sum(x['sec'] for x in segs)/3600:.1f} h |")
        if len(segs) > 1:
            print(f"| segments (resumes) | {len(segs)} |")
    print(f"| audit operations | {dict(jo)} |\n")

    print("New-code rate, mean per ten batches:\n")
    print("| batches | mean new/batch | rate |\n|---|---|---|")
    for lo in range(0, (B[-1]["b"] if B else 0) + 1, 10):
        w = [r for r in B if lo <= r["b"] < lo + 10]
        if w:
            # the script's own rate, not new/50: a batch that absorbed a failed
            # batch's carry processes 100 labels and /50 reports it as double
            print(f"| {lo}-{lo+9} | {sum(x['new'] for x in w)/len(w):.1f} | "
                  f"{sum(x['rate'] for x in w)/len(w):.1f}% |")
    print("\nAudits:\n")
    print("| audit | operations | churn | live codes after |\n|---|---|---|---|")
    for a in A:
        print(f"| v{a['v']} | {a['ops']} | {a['churn']:.1f}% | {a['codes']} |")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="final")
    ap.add_argument("--no-gate", action="store_true",
                    help="emit numbers without running the regression suite first")
    args = ap.parse_args()
    if not args.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()
    main(args.codebook)
