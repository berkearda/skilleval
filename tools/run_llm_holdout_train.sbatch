#!/bin/bash
# T-123: retrain the text-conditioned NCDM with the 763 cold-start LLMs excluded from fitting and validation.
# Same recipe as the main model (tools/train_expanded.py, seed 42, batch size 1024, 15 epochs, K=100 Q-matrix);
# the only difference is +holdout_llms=true. Saves expanded/text_conditioned_protocolB_llmsplit3048.pt and does
# not touch the main model. Environment and resources as in tools/run_t106_arms.sbatch, which ran in September.
# Expected (written before the run): 20,897,697 training triplets, about 20,407 batches per epoch,
# test AUC about 0.715 to 0.717 on the 3,048 training LLMs, about 10 minutes per epoch, about 105 GB of memory.
#SBATCH --job-name=llmsplit_train
#SBATCH --account=es_sachan
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7G
#SBATCH --time=08:00:00
#SBATCH --output=cdm_exploration/logs/llmsplit_train_%j.out
#SBATCH --error=cdm_exploration/logs/llmsplit_train_%j.err
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
mkdir -p cdm_exploration/logs
echo "host $(hostname)  $(date)"; nvidia-smi || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python tools/train_expanded.py device=cuda model.batch_size=1024 \
    +seed=42 +holdout_llms=true "+ckpt_tag=_llmsplit3048" \
    "+data.response_matrix=response_matrix_v2_full.npy" \
    "+data.qmatrix=qmatrix_v2_K100.npy" \
    "+data.text_embeddings=item_text_embeddings_v2_full.npz" \
    "+data.llm_names=response_matrix_v2_full_llms.json" \
    "+data.item_metadata=response_matrix_v2_full_items.json"
echo "TRAIN_EXIT=$?"
echo "finished $(date)"
