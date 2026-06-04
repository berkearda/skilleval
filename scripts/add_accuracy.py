"""Add per-model accuracy to public/data/theta_matrix.json.

Accuracy = row mean of the binary response matrix (fraction of the 9,523
items the model answered correctly). The response matrix and its LLM list
live in the cdmeval repo; the LLM order there matches the site's hf_id
order positionally (verified by assertion below).

Run:
    python3 scripts/add_accuracy.py
"""

import json
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
CDM = Path("/Users/berkearda/Desktop/cdmeval/cdm_exploration/data/cdm_ready")

theta_path = SITE / "public/data/theta_matrix.json"

R = np.load(CDM / "response_matrix_v2_full.npy")
llms = json.loads((CDM / "response_matrix_v2_full_llms.json").read_text())
models = json.loads(theta_path.read_text())

assert R.shape[0] == len(llms) == len(models), (
    f"size mismatch: R {R.shape[0]}, llms {len(llms)}, site {len(models)}"
)
assert set(np.unique(R)) <= {0.0, 1.0}, "response matrix is not binary"

acc = R.mean(axis=1)

mismatches = 0
for i, m in enumerate(models):
    if m.get("hf_id") != llms[i]:
        mismatches += 1
assert mismatches == 0, f"{mismatches} hf_id rows out of order; refusing to join"

for i, m in enumerate(models):
    m["accuracy"] = round(float(acc[i]), 4)

theta_path.write_text(json.dumps(models, separators=(",", ":")))
print(f"wrote accuracy for {len(models)} models -> {theta_path}")
print(f"accuracy range: {acc.min():.3f} .. {acc.max():.3f} (mean {acc.mean():.3f})")
