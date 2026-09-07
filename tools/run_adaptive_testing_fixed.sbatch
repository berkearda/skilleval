#!/bin/bash
#SBATCH --job-name=cdcat_fixed
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/cdcat_fixed_%j.out
#SBATCH --error=cdm_exploration/logs/cdcat_fixed_%j.err
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

python -u tools/run_adaptive_testing_fixed.py device=cuda \
    "+n_eval=100" "+n_repeats=3" "+cal_sizes='5,10,20,50,100,200,500'"
