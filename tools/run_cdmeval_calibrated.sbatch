#!/bin/bash
#SBATCH --job-name=cdmeval_K100_cal
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/cdmeval_K100_cal_%a_%A.out
#SBATCH --error=cdm_exploration/logs/cdmeval_K100_cal_%a_%A.err
#SBATCH --array=0
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation
mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/expanded

SEED=42

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} SEED=${SEED}"
nvidia-smi || true

python -u tools/train_cdmeval_calibrated.py --seed $SEED --K 100 --epochs 15
