#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_git_rebasin_wm_mnist_%j.out
#SBATCH --error=slurm_git_rebasin_wm_mnist_%j.err
#SBATCH --job-name=gitreb_wm_mnist
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/mnist/endpoints/standard/seed0/checkpoint-50.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/mnist/endpoints/standard/seed1/checkpoint-50.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/alignment/git_rebasin_weight_matching/seed0-seed1}"
DATA_PATH="${DATA_PATH:-./data}"
MAX_ITER="${MAX_ITER:-100}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-128}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "Git Re-Basin Weight Matching (VGG16/MNIST)"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "MAX_ITER: ${MAX_ITER}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

srun python scripts/analysis/run_git_rebasin_vgg16_mnist_weight_matching.py \
    model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
    model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
    output_root="${OUTPUT_ROOT}" \
    data_path="${DATA_PATH}" \
    max_iter="${MAX_ITER}" \
    evaluation_batch_size="${EVALUATION_BATCH_SIZE}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    num_workers="${NUM_WORKERS}" \
    seed="${SEED}" \
    device=cuda

echo ""
echo "========================================"
echo "GIT-REBASIN WEIGHT MATCHING COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
echo "Metadata: ${OUTPUT_ROOT}/metadata.json"
echo "Comparison table: ${OUTPUT_ROOT}/evaluation/comparison.json"
echo "Comparison plot: ${OUTPUT_ROOT}/evaluation/comparison.png"
