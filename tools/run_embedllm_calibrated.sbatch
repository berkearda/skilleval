#!/bin/bash
#SBATCH --job-name=cdmeval_embedllm_cal
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/embedllm_cal_%a_%A.out
#SBATCH --error=cdm_exploration/logs/embedllm_cal_%a_%A.err
#SBATCH --array=0
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation
mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/embedllm

# Use seed 42 only — single calibrated run, paired with CDMEval calibrated.
SEED=42

# Build calibrated inputs first (idempotent — skips if already there)
if [ ! -f cdm_exploration/data/embedllm/calibrated/train.csv ]; then
    echo "Building calibrated EmbedLLM inputs..."
    python -u tools/build_embedllm_inputs.py --calibrated
fi

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} SEED=${SEED}"
nvidia-smi || true

# Run EmbedLLM with calibrated split (train.csv = train_inner, test.csv = val).
# Need to point run_embedllm.py at the calibrated subdir; small flag to add.
python -u tools/run_embedllm.py --d_model 232 --seed $SEED --calibrated
