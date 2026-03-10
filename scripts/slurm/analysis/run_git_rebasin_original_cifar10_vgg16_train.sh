#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_git_rebasin_orig_cifar10_train_%j.out
#SBATCH --error=slurm_git_rebasin_orig_cifar10_train_%j.err
#SBATCH --job-name=gitrb_c10_tr

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/external/git-rebasin/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache}"
mkdir -p "${MPLCONFIGDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

SEED="${SEED:-0}"
WIDTH_MULTIPLIER="${WIDTH_MULTIPLIER:-64}"
WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${PROJECT_ROOT}/external/git-rebasin/src/cifar10_vgg_run.py" ]; then
    echo "Missing external/git-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/git-rebasin"
    exit 1
fi

echo "========================================"
echo "Original git-rebasin CIFAR10 VGG16 Train"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "SEED: ${SEED}"
echo "WIDTH_MULTIPLIER: ${WIDTH_MULTIPLIER}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "PYTHON: $(which python)"
echo ""

if [ "${WANDB_MODE}" = "disabled" ]; then
    WANDB_MODE="${WANDB_MODE}" srun python external/git-rebasin/src/cifar10_vgg_run.py \
        --seed "${SEED}" \
        --width-multiplier "${WIDTH_MULTIPLIER}" \
        --test \
        ${EXTRA_ARGS}
else
    WANDB_MODE="${WANDB_MODE}" srun python external/git-rebasin/src/cifar10_vgg_run.py \
        --seed "${SEED}" \
        --width-multiplier "${WIDTH_MULTIPLIER}" \
        ${EXTRA_ARGS}
fi
