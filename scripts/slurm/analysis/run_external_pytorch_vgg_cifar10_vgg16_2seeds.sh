#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_pytorch_vgg_cifar10_vgg16_2seeds_%j.out
#SBATCH --error=slurm_external_pytorch_vgg_cifar10_vgg16_2seeds_%j.err
#SBATCH --job-name=ext_vgg16_2s
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
EXTERNAL_ROOT="${PROJECT_ROOT}/external/pytorch-vgg-cifar10"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
export EXTRA_PYTHONPATH="${PROJECT_ROOT}/.cluster-pydeps"
export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH}"

if [ ! -f "${EXTERNAL_ROOT}/run.sh" ]; then
    echo "Missing external/pytorch-vgg-cifar10 in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/pytorch-vgg-cifar10"
    exit 1
fi

ARCH="${ARCH:-vgg16}"

echo "========================================"
echo "External pytorch-vgg-cifar10 VGG16 x2"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "EXTERNAL_ROOT: ${EXTERNAL_ROOT}"
echo "ARCH: ${ARCH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "PYTHON: $(command -v python || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

cd "${EXTERNAL_ROOT}"
srun env ARCH="${ARCH}" bash run.sh

echo ""
echo "========================================"
echo "EXTERNAL PYTORCH-VGG-CIFAR10 COMPLETE"
echo "========================================"
echo "Outputs written under: ${EXTERNAL_ROOT}"
