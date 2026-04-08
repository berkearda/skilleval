#!/bin/bash
#SBATCH --job-name=cdmeval_loo
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/loo_%A_%a.out
#SBATCH --error=cdm_exploration/logs/loo_%A_%a.err
#SBATCH --array=0,2,3,4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

module load stack/2024-06 gcc/12.2.0 python/3.11.6
set -euo pipefail
mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/leave_one_out

export PYTHONUNBUFFERED=1
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python tools/run_leave_one_out.py device=cuda
