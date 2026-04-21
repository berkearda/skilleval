#!/bin/bash
# One-time venv bootstrap for the IrtNet head-to-head experiment on Pro 6000
# (Blackwell / CUDA 13). Run on an Euler login node from the repo root:
#
#     ssh euler
#     cd $SCRATCH/cdm-llm-evaluation
#     bash tools/bootstrap_pro6000_venv.sh
#
# Idempotent: re-runs skip already-installed packages where pip can detect
# them. If the venv directory already exists the script exits early — remove
# "$VENV" manually if you want a clean rebuild.

set -euo pipefail

VENV="${SCRATCH}/cdmeval_pro6000_venv"
REPO="$SCRATCH/cdm-llm-evaluation"
IRTNET_DIR="${REPO}/cdm_exploration/repos/IrtNet"
IRTNET_REQS="${IRTNET_DIR}/requirements.txt"

echo "[bootstrap] venv target: ${VENV}"
echo "[bootstrap] repo: ${REPO}"

# Load the CUDA 13 toolchain (matches run_irtnet_headtohead.sbatch exactly).
source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

# Clone IrtNet into the gitignored repos dir if absent (required by
# tools/run_irtnet_headtohead.py::import_irtnet_model_class).
if [[ ! -d "${IRTNET_DIR}" ]]; then
    echo "[bootstrap] Cloning IrtNet into ${IRTNET_DIR}"
    mkdir -p "${REPO}/cdm_exploration/repos"
    git clone https://github.com/JianhaoChen-nju/IrtNet.git "${IRTNET_DIR}"
else
    echo "[bootstrap] IrtNet clone already present: ${IRTNET_DIR}"
fi

if [[ -d "${VENV}" ]]; then
    echo "[bootstrap] ${VENV} already exists. Delete it manually if you want to"
    echo "            rebuild; exiting early (IrtNet clone was still verified above)."
    exit 0
fi

python -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# PyTorch from the cu128 index. The cu128 index no longer ships torch 2.6.0
# (IrtNet's pinned version); it starts at 2.7.0. We use 2.9.1 — stable, full
# sm_120 / sm_122 kernel support for RTX Pro 6000 Blackwell, backward
# compatible with IrtNet's training code (standard nn.Module APIs only).
# torchvision is NOT installed: neither CDMEval nor IrtNet imports it.
# Install this FIRST so pip sees torch as satisfied when processing IrtNet's
# requirements.txt (the grep filter below strips torch/torchvision/triton
# pins so pip doesn't downgrade or pull cu124 wheels from the default index).
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.9.1"

# CDMEval package (reads pyproject.toml from the repo root). We pass --no-deps
# because pyproject.toml still pins torch==2.4.1 (CUDA 12 era); that pin would
# downgrade the Blackwell-compatible torch we just installed. We install the
# actual runtime deps explicitly below.
cd "${REPO}"
python -m pip install --no-deps -e .

# Runtime deps — curated list, avoids IrtNet's requirements.txt entirely.
# IrtNet's requirements.txt pins nvidia-*-cu12==12.4.*, which conflicts with
# the nvidia-*-cu12==12.8.* libs torch 2.9.1+cu128 brings in. pip's resolver
# reacts by downgrading torch to 2.6.0+cu124 (no sm_120 kernels — fatal on
# Pro 6000). IrtNet's code only imports: torch, pandas, sklearn,
# sentence_transformers, tqdm, torch.nn/optim/utils — all covered below.
python -m pip install \
    "numpy>=1.24" "pandas>=2.0" "scipy>=1.10" "scikit-learn>=1.3" \
    "tqdm" "huggingface-hub" "tokenizers" "safetensors" \
    "transformers==4.49.0" "sentence-transformers==3.4.1" \
    "matplotlib" "seaborn" "hydra-core>=1.3,<2.0" "omegaconf>=2.3" \
    "EduCDM==0.0.13" "hdbscan==0.8.40" "umap-learn==0.5.7" \
    "anthropic"

# Strict sanity check. Aborts if torch is not 2.9.1+cu128 (e.g. silently
# downgraded by a later dep install), so the bootstrap never finishes claiming
# success with an unusable Blackwell install.
python - <<'PY'
import sys
import torch
ver = torch.__version__
print(f"torch {ver}, cuda available: {torch.cuda.is_available()}")
if not ver.startswith("2.9.1+cu128"):
    print(f"ERROR: expected torch 2.9.1+cu128, got {ver}", file=sys.stderr)
    print("       A later install likely downgraded torch. Delete the venv and", file=sys.stderr)
    print("       re-run the bootstrap after fixing the conflicting pin.", file=sys.stderr)
    sys.exit(1)
if torch.cuda.is_available():
    print(f"  cuda version: {torch.version.cuda}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
else:
    print("  (cuda_available=False on login node is expected; real check runs on GPU node)")
PY

echo
echo "[bootstrap] Done. Venv at ${VENV}."
echo "[bootstrap] Submit the sweep with:"
echo "            sbatch ${REPO}/tools/run_irtnet_headtohead.sbatch"
