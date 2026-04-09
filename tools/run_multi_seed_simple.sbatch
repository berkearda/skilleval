#!/bin/bash
#SBATCH --job-name=cdmeval_multiseed
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/multiseed_simple_%A_%a.out
#SBATCH --error=cdm_exploration/logs/multiseed_simple_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --constraint=gpu_model:RTX_4090
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

module load stack/2024-06 gcc/12.2.0 python/3.11.6
set -euo pipefail
mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/multi_seed

export PYTHONUNBUFFERED=1

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, SEED=${SEED}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python tools/train_expanded.py device=cuda +seed=$SEED \
    data.response_matrix=response_matrix_v2_full.npy \
    data.qmatrix=qmatrix_v2_K100.npy \
    data.text_embeddings=item_text_embeddings_v2_full.npz \
    data.llm_names=response_matrix_v2_full_llms.json \
    data.item_metadata=response_matrix_v2_full_items.json \
    data.skill_embeddings=skill_embeddings_v2_K100.npz
