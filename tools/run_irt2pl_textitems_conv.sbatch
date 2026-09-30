#!/bin/bash
# Convergence check for Table 2 baseline (b), the text-conditioned 2PL (decisions 2026-09-22): the 15-epoch runs of job
# 14890464 were still gaining 0.0002 to 0.0004 validation AUC per epoch at the end. Seed 42, both splits, up to 60 epochs
# with early stopping off (patience 1000), at the 2PL's learning rate 0.005 and at 0.02. Outputs carry the suffix
# _conv60 / _conv60_lr002, so the 15-epoch files are not touched.
# Expected (written before the run): validation AUC levels off within 60 epochs; best-epoch test AUC ID 0.775 to 0.80,
# OOD 0.730 to 0.745; lr 0.02 reaches about the same level in fewer epochs. About 30 minutes, one GPU, under 32 GB.
#SBATCH --job-name=irt2pl_conv
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=cdm_exploration/logs/irt2pl_conv_%j.out
#SBATCH --error=cdm_exploration/logs/irt2pl_conv_%j.err
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

for lr in 0.005 0.02; do
    tag=_conv60; [ "$lr" = "0.02" ] && tag=_conv60_lr002
    for p in A B; do
        python -u tools/run_irt2pl_textitems.py --protocol $p --seed 42 --device cuda --epochs 60 --patience 1000 --lr $lr --tag $tag
        echo "EXIT $p lr=$lr = $?"
    done
done
echo "finished $(date)"
