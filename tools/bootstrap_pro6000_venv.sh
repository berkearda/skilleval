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

# PyTorch pinned to 2.6.0 from the cu128 index. This is the version IrtNet's
# requirements.txt asks for, and cu128 wheels include sm_120 kernels needed by
# RTX Pro 6000 Blackwell (confirmed working in PyTorch Forums threads on sm_120).
# We install it FIRST so pip sees the requirement as satisfied when it later
# processes IrtNet's requirements.txt, and does not silently reinstall
# torch==2.6.0 from the default PyPI channel (which is cu124, missing sm_120).
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.6.0" "torchvision==0.21.0"

# CDMEval package (reads pyproject.toml from the repo root). We pass --no-deps
# because pyproject.toml still pins torch==2.4.1 (CUDA 12 era); that pin is
# correct for local mps runs but would downgrade the Blackwell-compatible torch
# we just installed. IrtNet's requirements.txt covers the same scientific
# dependencies (numpy, pandas, scikit-learn, sentence-transformers, tqdm).
cd "${REPO}"
python -m pip install --no-deps -e .

# IrtNet dependencies. We strip the `torch==2.6.0` line so pip does not revisit
# torch on the default channel (which would fetch the cu124 build); the torch
# we installed above already satisfies every downstream package.
if [[ -f "${IRTNET_REQS}" ]]; then
    tmp_req=$(mktemp)
    grep -v -E '^(torch|torchvision)==' "${IRTNET_REQS}" > "${tmp_req}"
    python -m pip install -r "${tmp_req}"
    rm -f "${tmp_req}"
    # Also install the cdmeval runtime deps that we skipped with --no-deps,
    # minus torch (already handled). These are the deps from pyproject.toml
    # that IrtNet's requirements does not already cover.
    python -m pip install hydra-core omegaconf EduCDM==0.0.13 \
        hdbscan==0.8.40 umap-learn==0.5.7
else
    echo "[bootstrap] WARNING: ${IRTNET_REQS} not found after clone — aborting."
    exit 1
fi

# Light sanity check.
python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda version: {torch.version.cuda}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY

echo
echo "[bootstrap] Done. Venv at ${VENV}."
echo "[bootstrap] Submit the sweep with:"
echo "            sbatch ${REPO}/tools/run_irtnet_headtohead.sbatch"
