#!/bin/bash
#SBATCH --job-name=cdmeval_irtnet
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/irtnet_%A_%a.out
#SBATCH --error=cdm_exploration/logs/irtnet_%A_%a.err
#SBATCH --array=0-8
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Pro 6000 (Blackwell / CUDA 13) module stack — confirmed working under es_sachan.
# The cuda13pr.24h partition grants up to 24h wall; we request 12h to buffer
# the nine-config sweep (3 d_model x 3 seeds). Array indices 0..8 map to
# (d_model, seed) via tools/run_irtnet_headtohead.py::resolve_sweep_config.

set -euo pipefail

source /etc/profile                                                  # exposes `module`
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda              # CUDA 13 toolchain
module load eth_proxy                                                # pip / HF downloads

# Activate the CUDA-13 compatible venv bootstrapped by tools/bootstrap_pro6000_venv.sh.
source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation

mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/irtnet

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python -u tools/run_irtnet_headtohead.py device=cuda
