#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_git_rebasin_orig_mnist_wm_%j.out
#SBATCH --error=slurm_git_rebasin_orig_mnist_wm_%j.err
#SBATCH --job-name=gitrb_mnist_wm

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/external/git-rebasin/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

MODEL_A="${MODEL_A:?Set MODEL_A to a W&B artifact version like v5}"
MODEL_B="${MODEL_B:?Set MODEL_B to a W&B artifact version like v4}"
SEED="${SEED:-0}"

if [ ! -f "${PROJECT_ROOT}/external/git-rebasin/src/mnist_vgg_weight_matching.py" ]; then
    echo "Missing external/git-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/git-rebasin"
    exit 1
fi

echo "========================================"
echo "Original git-rebasin MNIST VGG16 Weight Matching"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A: ${MODEL_A}"
echo "MODEL_B: ${MODEL_B}"
echo "SEED: ${SEED}"
echo "PYTHON: $(which python)"
echo ""

WANDB_MODE="${WANDB_MODE:-online}" srun python external/git-rebasin/src/mnist_vgg_weight_matching.py \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --seed "${SEED}"
