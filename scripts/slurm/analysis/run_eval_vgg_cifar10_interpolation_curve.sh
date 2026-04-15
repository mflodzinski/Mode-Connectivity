#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_vgg_cifar10_interp_curve_%j.out
#SBATCH --error=slurm_eval_vgg_cifar10_interp_curve_%j.err
#SBATCH --job-name=eval_vgg_cv
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

VGG_NAME="${VGG_NAME:-}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-}"
REBASED_CHECKPOINT="${REBASED_CHECKPOINT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
DATA_PATH="${DATA_PATH:-./data}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-51}"

if [ -z "${VGG_NAME}" ] || [ -z "${MODEL_B_CHECKPOINT}" ] || [ -z "${REBASED_CHECKPOINT}" ] || [ -z "${OUTPUT_PATH}" ]; then
    echo "Missing required environment variables."
    echo "Required: VGG_NAME, MODEL_B_CHECKPOINT, REBASED_CHECKPOINT, OUTPUT_PATH"
    exit 1
fi

args=(
    --vgg-name "${VGG_NAME}"
    --model-b-checkpoint "${MODEL_B_CHECKPOINT}"
    --rebased-checkpoint "${REBASED_CHECKPOINT}"
    --output-path "${OUTPUT_PATH}"
    --data-path "${DATA_PATH}"
    --image-size "${IMAGE_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${SLURM_CPUS_PER_TASK}"
    --num-eval-points "${NUM_EVAL_POINTS}"
    --device cuda
)

srun python scripts/analysis/eval_vgg_cifar10_interpolation_curve.py "${args[@]}"
