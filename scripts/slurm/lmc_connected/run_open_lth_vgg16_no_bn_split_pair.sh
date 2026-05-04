#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_open_lth_no_bn_split_%x_%j.out
#SBATCH --error=slurm_open_lth_no_bn_split_%x_%j.err
#SBATCH --job-name=open_lth_no_bn_split
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <split_iter>"
  exit 1
fi

SPLIT_ITER="$1"

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
DATASET_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/open_lth_datasets"
OUTPUT_ROOT="results/vgg16/cifar10/endpoints/open_lth_shared_split_no_bn"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

if [ ! -f "${PROJECT_ROOT}/external/open_lth/open_lth.py" ]; then
  echo "Missing external/open_lth submodule."
  echo "Run: git submodule update --init --recursive external/open_lth"
  exit 1
fi

mkdir -p "${DATASET_ROOT}"

echo "========================================"
echo "open_lth VGG16/CIFAR10 shared split pair (no BN)"
echo "========================================"
echo "Split iteration: ${SPLIT_ITER}"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "DATASET_ROOT: ${DATASET_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}/iter${SPLIT_ITER}"
echo "Augmentation: enabled (repo default)"
echo ""

srun python scripts/train/run_open_lth_vgg16_split_pair.py \
  --model-name "cifar_vgg_no_bn_16" \
  --split-iter "${SPLIT_ITER}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --shared-seed 42 \
  --branch-seeds 0 1 \
  --num-workers 1 \
  --batch-size 128 \
  --num-eval-points 31
