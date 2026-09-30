#!/bin/bash
# Table 2 baseline (b), an external review (2026-09-22): the IRT 2PL with discrimination and difficulty computed
# from the item text by linear projections, as SkillEval's item parameters are (tools/run_irt2pl_textitems.py).
# Training settings as the existing 2PL row (lr 0.005, batch 16384, up to 15 epochs, patience 3).
# ID (protocol A): seed 42, as the 2PL and SkillEval ID values. OOD (protocol B): seeds 42 to 46, as SkillEval's
# five-seed OOD value.
# Expected (written before the run): ID test AUC between 0.78 and 0.84 (below the free 2PL's 0.8616); OOD test AUC
# between 0.68 and 0.75 with a seed spread under 0.005; item parameters and predictions not constant. Each run under
# 10 minutes on one GPU, the whole job under 1 hour, under 32 GB of RAM.
#SBATCH --job-name=irt2pl_text
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output=cdm_exploration/logs/irt2pl_text_%j.out
#SBATCH --error=cdm_exploration/logs/irt2pl_text_%j.err
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

python -u tools/run_irt2pl_textitems.py --protocol A --seed 42 --device cuda
echo "EXIT A 42 = $?"
for s in 42 43 44 45 46; do
    python -u tools/run_irt2pl_textitems.py --protocol B --seed $s --device cuda
    echo "EXIT B $s = $?"
done
echo "finished $(date)"
