#!/bin/bash
#SBATCH --job-name=cdmeval_kseed
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/kseed_%A_%a.out
#SBATCH --error=cdm_exploration/logs/kseed_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# CDMEval multi-seed K-sweep for T-021: close the asymmetric-ablation gap
# against IrtNet's 3-seed d-sweep. We already have seed-42 results at every
# K from the earlier K-ablation, and 5-seed results at K=100 from the
# multi-seed stability study. This array fills in seeds 43 and 44 at
# K=300 and K=500 so both methods are reported with 3-seed mean±std at
# both their primary config and their capacity-ceiling config.
#
#   Array idx -> (K, seed)
#     0 -> K=300, seed=43
#     1 -> K=300, seed=44
#     2 -> K=500, seed=43
#     3 -> K=500, seed=44

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation

mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/expanded

# Map SLURM_ARRAY_TASK_ID to (K, seed).
case "${SLURM_ARRAY_TASK_ID}" in
    0) K=300; SEED=43 ;;
    1) K=300; SEED=44 ;;
    2) K=500; SEED=43 ;;
    3) K=500; SEED=44 ;;
    *) echo "Unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} K=${K} SEED=${SEED}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python -u tools/run_k_ablation.py device=cuda "+ablation.K=${K}" "+seed=${SEED}"
