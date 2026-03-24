#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_random_scale_invariance_vgg11_cifar10_%j.out
#SBATCH --error=slurm_random_scale_invariance_vgg11_cifar10_%j.err
#SBATCH --job-name=rand_scale_v11
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

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SCALE_MIN="${SCALE_MIN:-}"
SCALE_MAX="${SCALE_MAX:-}"
SCALE_SAMPLING="${SCALE_SAMPLING:-}"
SCALE_SEED="${SCALE_SEED:-}"

echo "========================================"
echo "Random Scale Invariance VGG11 CIFAR10"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_CHECKPOINT: ${MODEL_CHECKPOINT:-<from config>}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT:-<from config>}"
echo "SCALE_MIN: ${SCALE_MIN:-<from config>}"
echo "SCALE_MAX: ${SCALE_MAX:-<from config>}"
echo "SCALE_SAMPLING: ${SCALE_SAMPLING:-<from config>}"
echo "SCALE_SEED: ${SCALE_SEED:-<from config>}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

args=(
    "num_workers=${SLURM_CPUS_PER_TASK}"
    "device=cuda"
)
if [ -n "${MODEL_CHECKPOINT}" ]; then
    args+=("model_checkpoint=${MODEL_CHECKPOINT}")
fi
if [ -n "${OUTPUT_ROOT}" ]; then
    args+=("output_root=${OUTPUT_ROOT}")
fi
if [ -n "${SCALE_MIN}" ]; then
    args+=("scale_min=${SCALE_MIN}")
fi
if [ -n "${SCALE_MAX}" ]; then
    args+=("scale_max=${SCALE_MAX}")
fi
if [ -n "${SCALE_SAMPLING}" ]; then
    args+=("scale_sampling=${SCALE_SAMPLING}")
fi
if [ -n "${SCALE_SEED}" ]; then
    args+=("scale_seed=${SCALE_SEED}")
fi

srun python scripts/analysis/run_random_scale_invariance_vgg11_cifar10.py "${args[@]}"
