#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_vgg11_cifar10_train_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_vgg11_cifar10_train_%j.err
#SBATCH --job-name=ext_sh_v11_tr
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

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
TRAIN_LR="${TRAIN_LR:-}"
MOMENTUM="${MOMENTUM:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"

echo "========================================"
echo "Original Sinkhorn VGG11 CIFAR10 Train"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT:-<from config>}"
echo "TRAIN_EPOCHS: ${TRAIN_EPOCHS:-<from config>}"
echo "TRAIN_LR: ${TRAIN_LR:-<from config>}"
echo "MOMENTUM: ${MOMENTUM:-<from config>}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAY:-<from config>}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

args=(
    "num_workers=${SLURM_CPUS_PER_TASK}"
    "device=cuda"
)
if [ -n "${OUTPUT_ROOT}" ]; then
    args+=("output_root=${OUTPUT_ROOT}")
fi
if [ -n "${TRAIN_EPOCHS}" ]; then
    args+=("train_epochs=${TRAIN_EPOCHS}")
fi
if [ -n "${TRAIN_LR}" ]; then
    args+=("train_lr=${TRAIN_LR}")
fi
if [ -n "${MOMENTUM}" ]; then
    args+=("momentum=${MOMENTUM}")
fi
if [ -n "${WEIGHT_DECAY}" ]; then
    args+=("weight_decay=${WEIGHT_DECAY}")
fi

srun python scripts/analysis/run_external_sinkhorn_original_vgg11_cifar10_train.py "${args[@]}"

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN VGG11 CIFAR10 TRAIN COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT:-<output_root from config>}"
