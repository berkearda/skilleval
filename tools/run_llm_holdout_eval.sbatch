#!/bin/bash
# T-123: cold start and adaptive item selection with the network trained on the 3,048 training LLMs only.
# Task 0: cold start on the 763 held-out LLMs (paper settings) plus the same-LLM full-calibration reference.
# Task 1: adaptive item selection, large budgets (as tools/run_adaptive_testing_v2_fixed.sbatch).
# Task 2: adaptive item selection, small budgets (as tools/run_cdcat_small_n.sbatch).
# Result files carry the suffix _llmsplit, so the submitted result files are not touched.
#SBATCH --job-name=llmsplit_eval
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=cdm_exploration/logs/llmsplit_eval_%A_%a.out
#SBATCH --error=cdm_exploration/logs/llmsplit_eval_%A_%a.err
set -uo pipefail
source /etc/profile
module load stack/2024-06 gcc/12.2.0 python/3.11.6
REPO=$SCRATCH/cdm-llm-evaluation
source "$SCRATCH/cdmeval_t106_venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export MPLBACKEND=Agg
cd "$REPO" || exit 1
CKPT=cdm_exploration/checkpoints/expanded/text_conditioned_protocolB_llmsplit3048.pt
ls -la "$CKPT" || { echo "checkpoint missing"; exit 1; }
echo "host $(hostname)  $(date)  task ${SLURM_ARRAY_TASK_ID}"; nvidia-smi || true
case "${SLURM_ARRAY_TASK_ID}" in
  0) python -u tools/cold_start_v2_fixed.py device=cuda "+ckpt=${CKPT}" "+out_tag=_llmsplit" "+full_calibration=true" ;;
  1) python -u tools/run_adaptive_testing_v2_fixed.py device=cuda "+ckpt=${CKPT}" "+out_tag=_llmsplit_largeN" \
       "+n_eval=50" "+n_repeats=1" "+cal_sizes='10,50,100,200,500'" "+criteria='random,heuristic,trace'" ;;
  2) python -u tools/run_adaptive_testing_v2_fixed.py device=cuda "+ckpt=${CKPT}" "+out_tag=_llmsplit_smallN" \
       "+n_eval=50" "+n_repeats=5" "+cal_sizes='1,2,5,10,20,30'" "+criteria='random,heuristic,trace'" ;;
esac
echo "EVAL_EXIT=$? task ${SLURM_ARRAY_TASK_ID}"
echo "finished $(date)"
