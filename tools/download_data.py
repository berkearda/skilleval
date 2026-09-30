#!/usr/bin/env python3
"""Download the data and build everything the training scripts read.

1. Downloads the Open LLM Leaderboard v2 results packaged by RouterEval (Huang et al., 2025; MIT licence) from
   the Hugging Face Hub (linggm/RouterEval, pinned revision, about 135 MB) and checks their SHA-256.
2. Extracts the files for the five benchmarks into cdm_exploration/data/routereval/.
3. Builds the response matrix (3,811 LLMs x 9,523 items) and the item-text embeddings with
   tools/build_response_matrix_v2.py (the embedding model, about 420 MB, is downloaded on the first run).
4. Writes the Q-matrix of the paper from release/qmatrix_K100.csv.

Everything lands in cdm_exploration/data/cdm_ready/.

    python tools/download_data.py
    python tools/download_data.py --with-mmlu-pro   # also extract the MMLU-Pro results
"""
import argparse
import csv
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ID = "linggm/RouterEval"
REVISION = "0ee4de2661854b588474c211bebe97dc1e434237"
FILES = {
    "leaderboard_score.zip": "21e0429f549898afdf229dac9d7f77c49b5ff8061340d8b86ed6031a38a54077",
    "leaderboard_prompt.zip": "7b454ba0ccbe44a51dd9bdbac6653263651bb73a0be90d1e72c4887164e29240",
}
NEEDED = {"leaderboard_score.zip": ["leaderboard_new.pkl"], "leaderboard_prompt.zip": ["leaderboard_new_prompt.pkl"]}
MMLU_PRO = {"leaderboard_score.zip": ["leaderboard_mmlu_pro.pkl"], "leaderboard_prompt.zip": ["leaderboard_mmlu_pro_prompt.pkl"]}

ROOT = Path(__file__).resolve().parents[1]
ROUTEREVAL = ROOT / "cdm_exploration/data/routereval"
CDM_READY = ROOT / "cdm_exploration/data/cdm_ready"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name):
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION))
    if sha256(path) != FILES[name]:
        raise SystemExit(f"{name}: checksum mismatch, refusing to use it")
    return path


def extract(zip_path, zip_name, members):
    # each zip holds one folder; keep the layout the scripts expect: routereval/<stem>/<stem>/<file>
    stem = zip_name[: -len(".zip")]
    target = ROUTEREVAL / stem
    with zipfile.ZipFile(zip_path) as z:
        for m in members:
            out = target / stem / m
            if out.exists() and out.stat().st_size == z.getinfo(f"{stem}/{m}").file_size:
                print(f"  {out.relative_to(ROOT)} already there", flush=True)
                continue
            z.extract(f"{stem}/{m}", target)
            print(f"  extracted {out.relative_to(ROOT)}", flush=True)


def write_qmatrix():
    rows = list(csv.DictReader(open(ROOT / "release/qmatrix_K100.csv")))
    q = np.zeros((len(rows), 100), dtype=np.int64)
    for r in rows:
        for k in r["skill_ids"].split(";"):
            if k:
                q[int(r["item_idx"]), int(k)] = 1
    CDM_READY.mkdir(parents=True, exist_ok=True)
    np.save(CDM_READY / "qmatrix_v2_K100.npy", q)
    print(f"  wrote {(CDM_READY / 'qmatrix_v2_K100.npy').relative_to(ROOT)} {q.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download RouterEval and build the SkillEval data.")
    ap.add_argument("--with-mmlu-pro", action="store_true", help="also extract the MMLU-Pro results (not used in the paper)")
    a = ap.parse_args()

    print(f"Downloading from huggingface.co/datasets/{REPO_ID} (revision {REVISION[:7]})", flush=True)
    for name in FILES:
        path = fetch(name)
        members = NEEDED[name] + (MMLU_PRO[name] if a.with_mmlu_pro else [])
        extract(path, name, members)

    print("Building the response matrix and the item-text embeddings", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/build_response_matrix_v2.py"),
                    "--routereval", str(ROUTEREVAL), "--out", str(CDM_READY), "--embeddings"], check=True)
    print("Writing the Q-matrix", flush=True)
    write_qmatrix()
    print("Done. Next: python tools/train_expanded.py --config-name paper", flush=True)


if __name__ == "__main__":
    main()
