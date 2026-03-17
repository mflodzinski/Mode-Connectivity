#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_train_vgg16_mnist_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_train_vgg16_mnist_%j.err
#SBATCH --job-name=ext_sh_vgg16
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
export EXTRA_PYTHONPATH="${PROJECT_ROOT}/.cluster-pydeps"
export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH}"

if [ ! -f "${PROJECT_ROOT}/external/sinkhorn-rebasin/examples/models/vgg.py" ]; then
    echo "Missing external/sinkhorn-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/sinkhorn-rebasin"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load graphviz >/dev/null 2>&1 || true
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/original_sinkhorn_vgg16_train}"
USE_SMALL_DATASET="${USE_SMALL_DATASET:-false}"
VALIDATION_SIZE="${VALIDATION_SIZE:-5000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
TRAIN_LR="${TRAIN_LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.0005}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"

echo "========================================"
echo "Original Sinkhorn VGG16 MNIST Training"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "USE_SMALL_DATASET: ${USE_SMALL_DATASET}"
echo "VALIDATION_SIZE: ${VALIDATION_SIZE}"
echo "TRAIN_EPOCHS: ${TRAIN_EPOCHS}"
echo "OPTIMIZER_NAME: ${OPTIMIZER_NAME}"
echo "TRAIN_LR: ${TRAIN_LR}"
echo "EARLY_STOPPING_PATIENCE: ${EARLY_STOPPING_PATIENCE}"
echo "MIN_DELTA: ${MIN_DELTA}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/run_external_sinkhorn_original_train_vgg16_mnist.py \
    output_root="${OUTPUT_ROOT}" \
    use_small_dataset="${USE_SMALL_DATASET}" \
    validation_size="${VALIDATION_SIZE}" \
    train_epochs="${TRAIN_EPOCHS}" \
    optimizer_name="${OPTIMIZER_NAME}" \
    train_lr="${TRAIN_LR}" \
    weight_decay="${WEIGHT_DECAY}" \
    early_stopping_patience="${EARLY_STOPPING_PATIENCE}" \
    min_delta="${MIN_DELTA}" \
    train_batch_size="${TRAIN_BATCH_SIZE}" \
    eval_batch_size="${EVAL_BATCH_SIZE}" \
    image_size="${IMAGE_SIZE}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN VGG16 MNIST TRAINING COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
