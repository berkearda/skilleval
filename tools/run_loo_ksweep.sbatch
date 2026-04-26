#!/bin/bash
#SBATCH --job-name=cdmeval_loo_ksweep
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/loo_ksweep_%A_%a.out
#SBATCH --error=cdm_exploration/logs/loo_ksweep_%A_%a.err
#SBATCH --array=0-29
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Leave-one-benchmark-out at every K (except K=100 which is already done from
# the original run_leave_one_out.sbatch sweep). 30 tasks total: 6 K values
# (50, 150, 200, 300, 400, 500) × 5 LOO folds (BBH, GPQA, IFEval, MATH, MuSR).
#
# Requested on 2026-04-25 to provide LLM profiles at multiple K
# values under the LOO setting (full-data K-sweep already shipped).
#
# Array layout: idx = K_idx * 5 + bench_idx
#   K_VALUES   = [50, 150, 200, 300, 400, 500]   (6 values)
#   BENCHMARKS = [BBH, GPQA, IFEval, MATH, MuSR] (5 values, matches the order
#                                                 in run_leave_one_out.py)
#
# Memory-per-cpu is 32G to match the multi-seed K-sweep sbatch — the lazy
# triplet dataset keeps memory bounded but Pro 6000 nodes set per-job caps.

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation

mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/leave_one_out

K_VALUES=(50 150 200 300 400 500)
BENCHMARKS=(BBH GPQA IFEval MATH MuSR)

K_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

K=${K_VALUES[$K_IDX]}
BENCH=${BENCHMARKS[$BENCH_IDX]}

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Resolved: K=${K}, held_out=${BENCH}"
echo "Hostname: $(hostname)"
nvidia-smi || true

python -u tools/run_leave_one_out.py device=cuda \
    "+ablation.K=${K}" "+held_out=${BENCH}"
