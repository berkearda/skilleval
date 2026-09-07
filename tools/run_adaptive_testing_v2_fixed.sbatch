#!/bin/bash
#SBATCH --job-name=cdcat_v2_fixed
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/cdcat_v2_fixed_%j.out
#SBATCH --error=cdm_exploration/logs/cdcat_v2_fixed_%j.err
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

# Reduced scope: 50 LLMs (was 100), 1 seed (was 3), 3 criteria (drop d-optimal)
# Keeps the meaningful comparison: random vs heuristic vs trace (real autograd Fisher)
python -u tools/run_adaptive_testing_v2_fixed.py device=cuda \
    "+n_eval=50" "+n_repeats=1" \
    "+cal_sizes='10,50,100,200,500'" \
    "+criteria='random,heuristic,trace'"
