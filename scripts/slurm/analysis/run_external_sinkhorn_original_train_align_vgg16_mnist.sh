#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_train_align_mnist_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_train_align_mnist_%j.err
#SBATCH --job-name=ext_sh_train
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

OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/original_sinkhorn_vgg16_train_align}"
USE_SMALL_DATASET="${USE_SMALL_DATASET:-true}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"
ALIGNMENT_EPOCHS="${ALIGNMENT_EPOCHS:-20}"
ALIGNMENT_LR="${ALIGNMENT_LR:-0.1}"
LOSS_NAME="${LOSS_NAME:-random}"
TAU="${TAU:-1.0}"
SINKHORN_ITERS="${SINKHORN_ITERS:-20}"
SINKHORN_L="${SINKHORN_L:-1.0}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-25}"

echo "========================================"
echo "Original Sinkhorn VGG16 MNIST Train + Align"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "USE_SMALL_DATASET: ${USE_SMALL_DATASET}"
echo "TRAIN_EPOCHS: ${TRAIN_EPOCHS}"
echo "ALIGNMENT_EPOCHS: ${ALIGNMENT_EPOCHS}"
echo "LOSS_NAME: ${LOSS_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/run_external_sinkhorn_original_train_align_vgg16_mnist.py \
    output_root="${OUTPUT_ROOT}" \
    use_small_dataset="${USE_SMALL_DATASET}" \
    train_epochs="${TRAIN_EPOCHS}" \
    train_lr="${TRAIN_LR}" \
    alignment_epochs="${ALIGNMENT_EPOCHS}" \
    alignment_lr="${ALIGNMENT_LR}" \
    loss_name="${LOSS_NAME}" \
    tau="${TAU}" \
    sinkhorn_iters="${SINKHORN_ITERS}" \
    sinkhorn_l="${SINKHORN_L}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN VGG16 MNIST TRAIN + ALIGN COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
