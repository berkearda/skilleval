#!/bin/bash
#SBATCH --job-name=cdmeval_protoA
#SBATCH --account=es_sachan
#SBATCH --output=cdm_exploration/logs/protoA_%A_%a.out
#SBATCH --error=cdm_exploration/logs/protoA_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Protocol A reruns (interaction-wise random 80/10/10 triplet split) on the
# v2 dataset, requested on 2026-04-25 to compare seen vs unseen
# performance against the Protocol B numbers in the paper.
#
# Array layout (4 tasks):
#   idx 0: NCDM K=100               (TextConditionedNet)
#   idx 1: IrtNet d_model=232       (as published)
#   idx 2: IrtNet d_model=512       (capacity sensitivity)
#   idx 3: IrtNet d_model=1024      (capacity ceiling)
#
# Memory-per-cpu is 32G to match run_multi_seed_ksweep.sbatch — the long
# triplet table at K=100 was the OOM trigger there and the same risk applies
# here under Protocol A where every (LLM, item) cell is materialised.

set -euo pipefail

source /etc/profile
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy

source "$SCRATCH/cdmeval_pro6000_venv/bin/activate"

export HF_HOME="$SCRATCH/.huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.huggingface"
export PYTHONUNBUFFERED=1

cd $SCRATCH/cdm-llm-evaluation

mkdir -p cdm_exploration/logs cdm_exploration/checkpoints/expanded \
         cdm_exploration/checkpoints/irtnet

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
nvidia-smi || true

# Note: d_model and seed passed explicitly so the IrtNet script does not
# fall back to its SLURM_ARRAY_TASK_ID -> d_model mapping (which assumes
# array indices {0,1,2}). Here our array indices are {0,1,2,3} with index 0
# reserved for the NCDM task.
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        echo "Task 0: NCDM K=100, Protocol A"
        python -u tools/run_ncdm_protocolA.py device=cuda +seed=42
        ;;
    1)
        echo "Task 1: IrtNet d_model=232, Protocol A"
        python -u tools/run_irtnet_protocolA.py device=cuda \
            +irtnet.d_model=232 +irtnet.seed=42
        ;;
    2)
        echo "Task 2: IrtNet d_model=512, Protocol A"
        python -u tools/run_irtnet_protocolA.py device=cuda \
            +irtnet.d_model=512 +irtnet.seed=42
        ;;
    3)
        echo "Task 3: IrtNet d_model=1024, Protocol A"
        python -u tools/run_irtnet_protocolA.py device=cuda \
            +irtnet.d_model=1024 +irtnet.seed=42
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac
