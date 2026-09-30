#!/bin/bash
# An external review, point 2 (2026-09-22): "mastered = theta > 0.5" is not identified, because predictions depend
# on theta - d only. Replacement: mastered = the main model's predicted accuracy on the skill's items is above the cut.
# 1. tools/predict_skill_accuracy.py: predictions for all 3,811 x 9,523 pairs from the main model, averaged per skill.
# 2. The co-mastery graph and the conjunctive/compensatory analysis, unchanged except for the mastery score:
#    theta > 0.5 (check: must reproduce the submitted files), predicted accuracy at 0.4 / 0.5 / 0.6, and observed
#    accuracy at 0.5 (model-free comparison). Outputs carry a suffix; the submitted files are not touched.
# Expected (written before the run): test AUC 0.7170 (the script exits 2 otherwise); Llama-3.2-1B on "Evaluating
# Truthfulness Of Nested Statements": theta 0.79, predicted 0.37, observed 0.38 (these are the Instruct model's values,
# mislabelled at launch; the script prints the base model); the theta check gives 75 skills,
# 242 links, depth 5, tertile 52.4 / 34.7 / 12.9, pattern n = 251; observed at 0.5 gives 86 skills, 278 links, depth 6
# (as in the local smoke test). Under 30 minutes, one GPU, under 10 GB of GPU memory, under 32 GB of RAM.
#SBATCH --job-name=mastery_pred
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=cdm_exploration/logs/mastery_pred_%j.out
#SBATCH --error=cdm_exploration/logs/mastery_pred_%j.err
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

python -u tools/predict_skill_accuracy.py --device cuda
rc=$?; echo "PREDICT_EXIT=$rc"
[ $rc -eq 0 ] || exit $rc

for run in "theta 0.5 +out_tag=_theta050check" "predicted 0.4" "predicted 0.5" "predicted 0.6" "observed 0.5"; do
    set -- $run
    for script in run_skill_prerequisites run_conjunctive_compensatory; do
        echo "=== $script mastery=$1 cut=$2 ${3:-}"
        python -u tools/$script.py device=cpu +mastery=$1 +mastery_cut=$2 ${3:-}
        echo "EXIT $script $1 $2 = $?"
    done
done
echo "finished $(date)"
