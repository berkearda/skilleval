#!/usr/bin/env python3
"""Merge experiment-log entries from another machine without losing any.

The cluster's copy of `experiment_log.json` was destroyed when scratch was purged,
so a cluster run writes a file holding only its own handful of entries while the
local file holds the project's whole history. Copying that file back, with rsync
or by hand, silently destroys everything the local one records: every number in
the paper traces to an entry in it (N5).

This appends instead. It never edits or removes an existing entry, it backs the
target up before writing, and it writes through a temporary file so an interrupted
run cannot leave a truncated log.

IDENTITY. An entry is identified by the pair (experiment, timestamp), because
timestamps are per-run and the same experiment name is deliberately reused across
runs. Content-identical duplicates are skipped. An incoming entry that shares the
pair with a local entry but differs in content is a conflict: it is reported and
NOT merged, because guessing which one is right is how a result gets quietly
replaced by another.

    python3 tools/merge_experiment_log.py --incoming /tmp/euler_log.json
    python3 tools/merge_experiment_log.py --incoming /tmp/euler_log.json --apply
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "cdm_exploration/experiments/experiment_log.json"


def key(e):
    return (str(e.get("experiment")), str(e.get("timestamp")))


def canon(e):
    return json.dumps(e, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", required=True, help="the other machine's log file")
    ap.add_argument("--target", default=str(TARGET))
    ap.add_argument("--apply", action="store_true",
                    help="without this, report what would change and write nothing")
    a = ap.parse_args()

    target = Path(a.target)
    incoming = Path(a.incoming)
    if not incoming.exists():
        raise SystemExit(f"{incoming} does not exist")
    inc = json.loads(incoming.read_text())
    cur = json.loads(target.read_text()) if target.exists() else []
    if not isinstance(inc, list) or not isinstance(cur, list):
        raise SystemExit("both logs must be JSON lists of entries")

    have = {}
    for e in cur:
        have.setdefault(key(e), []).append(e)

    new, dupes, conflicts = [], [], []
    for e in inc:
        k = key(e)
        if k not in have:
            new.append(e)
        elif any(canon(x) == canon(e) for x in have[k]):
            dupes.append(k)
        else:
            conflicts.append(k)

    print(f"target   {target.relative_to(REPO) if target.is_relative_to(REPO) else target}"
          f"  ({len(cur)} entries)")
    print(f"incoming {incoming}  ({len(inc)} entries)")
    print(f"  new              {len(new)}")
    print(f"  already present  {len(dupes)}")
    print(f"  CONFLICTS        {len(conflicts)}")
    for k in conflicts:
        print(f"    same (experiment, timestamp) but different content: {k}")
    if conflicts:
        print("  Conflicting entries are NOT merged. Resolve by hand: replacing one")
        print("  result with another silently is the failure this tool exists to avoid.")
    for e in new:
        r = e.get("results", {})
        auc = r.get("test_auc", r.get("test_auc_mean"))
        print(f"    + {e.get('experiment')}  {str(e.get('timestamp'))[:19]}"
              f"  auc={auc}  verified={e.get('verified')}")

    if not a.apply:
        print("\ndry run. Re-run with --apply to write.")
        return
    if not new:
        print("\nnothing to add.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_suffix(f".json.before_merge_{stamp}")
    if target.exists():
        shutil.copy2(target, backup)
        print(f"\nbacked up to {backup.name}")
    merged = cur + new
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, default=str))
    tmp.replace(target)                      # atomic, so no truncated log
    print(f"wrote {len(merged)} entries ({len(cur)} kept + {len(new)} added)")


if __name__ == "__main__":
    main()
