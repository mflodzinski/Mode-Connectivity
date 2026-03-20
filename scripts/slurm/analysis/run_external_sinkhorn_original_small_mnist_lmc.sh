#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_vgg16_mnist_lmc_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_vgg16_mnist_lmc_%j.err
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

OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/original_sinkhorn_lmc}"
VGG_NAME="${VGG_NAME:-VGG16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_LR="${TRAIN_LR:-0.05}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.0005}"
ALIGNMENT_ITERATIONS="${ALIGNMENT_ITERATIONS:-20}"
ALIGNMENT_LR="${ALIGNMENT_LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-50}"
DEBUG_ONE_BATCH="${DEBUG_ONE_BATCH:-false}"

echo "========================================"
echo "Original Sinkhorn VGG16 MNIST LMC"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "VGG_NAME: ${VGG_NAME}"
echo "TRAIN_EPOCHS: ${TRAIN_EPOCHS}"
echo "TRAIN_LR: ${TRAIN_LR}"
echo "MOMENTUM: ${MOMENTUM}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "EARLY_STOPPING_PATIENCE: ${EARLY_STOPPING_PATIENCE}"
echo "MIN_DELTA: ${MIN_DELTA}"
echo "ALIGNMENT_ITERATIONS: ${ALIGNMENT_ITERATIONS}"
echo "ALIGNMENT_LR: ${ALIGNMENT_LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "DEBUG_ONE_BATCH: ${DEBUG_ONE_BATCH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/run_external_sinkhorn_original_small_mnist_lmc.py \
    output_root="${OUTPUT_ROOT}" \
    vgg_name="${VGG_NAME}" \
    train_epochs="${TRAIN_EPOCHS}" \
    train_lr="${TRAIN_LR}" \
    momentum="${MOMENTUM}" \
    weight_decay="${WEIGHT_DECAY}" \
    early_stopping_patience="${EARLY_STOPPING_PATIENCE}" \
    min_delta="${MIN_DELTA}" \
    alignment_iterations="${ALIGNMENT_ITERATIONS}" \
    alignment_lr="${ALIGNMENT_LR}" \
    batch_size="${BATCH_SIZE}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    debug_one_batch="${DEBUG_ONE_BATCH}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN ${VGG_NAME} MNIST LMC COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
