#!/bin/bash -l
#SBATCH --job-name=ao-taboo-pairwise-eval
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=md748
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${REPO_DIR:-/project/md748/zh296/ao}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-labenv}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-9b-it}"
VERBALIZER_LORA_PATH="${VERBALIZER_LORA_PATH:-}"
MAX_CONTEXTS="${MAX_CONTEXTS:-5}"
SPLIT="${SPLIT:-test}"
PROMPT_TYPE="${PROMPT_TYPE:-direct}"

cd "$REPO_DIR"

echo "Job started on $(hostname) at $(date)"
echo "Repo: $REPO_DIR"
echo "Model: $MODEL_NAME"
echo "Verbalizer LoRA: ${VERBALIZER_LORA_PATH:-<repo default>}"
echo "Split: $SPLIT"
echo "Prompt type: $PROMPT_TYPE"
echo "Max contexts: $MAX_CONTEXTS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

nvidia-smi

source /project/md748/zh296/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

ARGS=(
  --model-name "$MODEL_NAME"
  --split "$SPLIT"
  --prompt-type "$PROMPT_TYPE"
  --max-contexts "$MAX_CONTEXTS"
)

if [[ -n "${VERBALIZER_LORA_PATH}" ]]; then
  ARGS+=(--verbalizer-lora-path "$VERBALIZER_LORA_PATH")
fi

python experiments/taboo_pairwise_eval.py "${ARGS[@]}"

echo "Job finished at $(date)"
