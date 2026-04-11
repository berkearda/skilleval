#!/bin/bash
#SBATCH --job-name=cdmeval_cdcat_v3
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/cdcat_v3_%j.out
#SBATCH --error=cdm_exploration/logs/cdcat_v3_%j.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --constraint=gpu_model:RTX_4090
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

module load stack/2024-06 gcc/12.2.0 python/3.11.6
set -euo pipefail
mkdir -p cdm_exploration/logs

export PYTHONUNBUFFERED=1
echo "Hostname: $(hostname)"
nvidia-smi || true

python tools/run_adaptive_testing_v3.py device=cuda \
    '+n_eval=0' \
    "+methods='random,a_optimal_batchbald,a_optimal_balanced'"
