#!/bin/bash
#SBATCH --job-name=cdcat_fixA
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/cdcat_fixA_%j.out
#SBATCH --error=cdm_exploration/logs/cdcat_fixA_%j.err
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation
mkdir -p cdm_exploration/logs

echo "JOB_ID=${SLURM_JOB_ID}"
nvidia-smi || true

# Fix A periodic-refit ETA: ~3 min/LLM for full v1 scope
# Reduced: 50 LLMs x 1 seed x 5 N values = ~30-40 min
python -u tools/run_adaptive_testing_fixedA.py device=cuda \
    "+n_eval=50" "+n_repeats=1" "+cal_sizes='10,50,100,200,500'"
