#!/bin/bash
#SBATCH --job-name=cdmeval_softq
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/softq_%A_%a.out
#SBATCH --error=cdm_exploration/logs/softq_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Soft-Q sweep: 5 NCDM training runs, one per softmax temperature, all
# at seed=42 with cfg.model.epochs (15) -- the same epoch budget as the
# binary K=100 baseline (run_multi_seed_42.json). Single-seed; multi-seed
# will only be added if at least one tau closes a meaningful chunk of the
# 3.08-pt EmbedLLM AUC gap.
#
# Inputs (must already exist on $SCRATCH):
#   cdm_exploration/data/cdm_ready/qmatrix_v2_K100_soft_tau{TAU}.npy
#   cdm_exploration/data/cdm_ready/raw_phrase_embeddings_v2.npz
#   cdm_exploration/data/cdm_ready/cluster_centroids_v2_K100.npy
# Plus standard v2 files (response matrix, item embeddings, etc.).
#
# Array idx -> tau:
#   0 -> tau=0.005
#   1 -> tau=0.01
#   2 -> tau=0.02
#   3 -> tau=0.05
#   4 -> tau=0.1
#
# Tau range rationale: SBERT cosine similarities live in a narrow band, so
# softmax(cos/tau) saturates fast. At tau>=0.3 the resulting Q is essentially
# uniform (row_max median ~0.025) which makes the NCDM training un-informative
# at that point. The chosen range spans row_max from ~0.52 (peaky, tau=0.005)
# to ~0.16 (soft, tau=0.1), which is the actually-meaningful interpolation
# region for this construction.
#
# Outputs (per task):
#   cdm_exploration/experiments/v2_soft_q_tau{TAU}_s42.json
#   cdm_exploration/checkpoints/expanded/text_conditioned_K100_soft_tau{TAU}_s42.pt
#   cdm_exploration/experiments/experiment_log.json (appended, verified=True)
#
# After all 5 tasks finish, run:
#   python tools/aggregate_soft_q_sweep.py --seed 42

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

TAUS=(0.005 0.01 0.02 0.05 0.1)
TAU=${TAUS[$SLURM_ARRAY_TASK_ID]}
QFILE="qmatrix_v2_K100_soft_tau${TAU}.npy"

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} TAU=${TAU} QFILE=${QFILE}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python -u tools/run_soft_q.py device=cuda \
    "+seed=42" \
    "+soft_q.tau=${TAU}" \
    "+soft_q.qmatrix_file=${QFILE}"
