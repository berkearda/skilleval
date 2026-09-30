#!/bin/bash
# Table 2 baseline (a), an external review (2026-09-22): the SkillEval network with free per-item difficulty and
# discrimination instead of text-derived ones, trained and evaluated on the interaction-wise (ID) split. Same script,
# split, seed 42, batch size 64 (config default, as the SkillEval ID run), 15 epochs and learning rate as the SkillEval
# ID number 81.73 (tools/run_ncdm_protocolA.py); only +free_items=true differs.
# Expected (written before the run): 29,033,721 training triplets, 453,651 batches per epoch, test AUC 0.84 to 0.88,
# 4 to 9 hours depending on the GPU, under 64 GB of memory (item ids and Q rows only, no text embeddings per triplet).
#SBATCH --job-name=ncdm_freeitems
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=cdm_exploration/logs/ncdm_freeitems_%j.out
#SBATCH --error=cdm_exploration/logs/ncdm_freeitems_%j.err
set -uo pipefail
source /etc/profile
module load stack/2024-06 gcc/12.2.0 python/3.11.6
echo "MODULE_LOAD_EXIT=$?"
REPO=$SCRATCH/cdm-llm-evaluation
source "$SCRATCH/cdmeval_t106_venv/bin/activate"
echo "VENV_ACTIVATE_EXIT=$?"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
cd "$REPO" || exit 1
echo "host $(hostname)  $(date)"; nvidia-smi || true
python -u tools/run_ncdm_protocolA.py device=cuda +seed=42 +free_items=true
echo "TRAIN_EXIT=$?"
echo "finished $(date)"
