#!/bin/bash -l
#SBATCH --job-name=ao-taboo-pairwise-train
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=md748
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${REPO_DIR:-/project/md748/zh296/ao}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-labenv}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-9b-it}"
TRAIN_JSONL="${TRAIN_JSONL:-experiments/pairwise_data/taboo_pairwise_val.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-5}"
SAVE_DIR="${SAVE_DIR:-checkpoints/taboo_pairwise_gemma2_9b_firstpass}"

cd "$REPO_DIR"

echo "Job started on $(hostname) at $(date)"
echo "Repo: $REPO_DIR"
echo "Model: $MODEL_NAME"
echo "Train JSONL: $TRAIN_JSONL"
echo "Eval JSONL: ${EVAL_JSONL:-<none>}"
echo "Save dir: $SAVE_DIR"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

nvidia-smi

source /project/md748/zh296/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  python experiments/taboo_pairwise_prepare.py --split val --prompt-type direct --output "$TRAIN_JSONL"
fi

TORCHRUN_ARGS=(
  --nproc_per_node=1
  experiments/taboo_pairwise_train.py
  --model-name "$MODEL_NAME"
  --train-jsonl "$TRAIN_JSONL"
  --batch-size "$BATCH_SIZE"
  --num-epochs "$NUM_EPOCHS"
  --lr "$LR"
  --save-dir "$SAVE_DIR"
)

if [[ -n "${EVAL_JSONL}" ]]; then
  TORCHRUN_ARGS+=(--eval-jsonl "$EVAL_JSONL")
fi

torchrun "${TORCHRUN_ARGS[@]}"

echo "Job finished at $(date)"
