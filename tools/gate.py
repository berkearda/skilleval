#!/usr/bin/env python3
"""The gate: a number is not accepted unless the tests guarding it pass.

Reporting scripts call require_tests_pass() before emitting anything. The suite
in tests/ pins defects that actually occurred, so a failure here means a number
this pipeline is about to publish is computed by code with a known-bad path.
"""
import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class GateFailed(SystemExit):
    pass


def require_tests_pass(verbose: bool = False) -> None:
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                       cwd=REPO, capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    if r.returncode != 0:
        sys.stderr.write("\n".join(tail[-25:]) + "\n")
        raise GateFailed(
            "\nGATE FAILED: the regression suite does not pass, so no number from "
            "this pipeline is accepted.\nFix the failure, or re-run with --no-gate "
            "and mark every number produced as unverified.\n")
    if verbose:
        print(f"gate: {tail[-1] if tail else 'ok'}")


if __name__ == "__main__":
    require_tests_pass(verbose=True)
    print("gate passed: numbers may be reported")
