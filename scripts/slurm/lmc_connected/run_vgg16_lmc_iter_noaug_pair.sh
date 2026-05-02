#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_pair_%x_%j.out
#SBATCH --error=slurm_lmc_iter_pair_%x_%j.err
#SBATCH --job-name=lmc_iter_pair
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <split_iter>"
  exit 1
fi

SPLIT_ITER="$1"

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH}"

OUTPUT_ROOT="results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug"

echo "========================================"
echo "Iteration split pair training"
echo "========================================"
echo "Split iteration: ${SPLIT_ITER}"
echo "Output root: ${OUTPUT_ROOT}/iter${SPLIT_ITER}"
echo "No train augmentation"
echo ""

srun python scripts/train/run_lmc_connected_pair_by_iteration.py \
    --mode pair \
    --output-root "${OUTPUT_ROOT}" \
    --split-iter "${SPLIT_ITER}" \
    --dataset CIFAR10 \
    --data-path ./data \
    --transform VGG \
    --model VGG16 \
    --shared-seed 42 \
    --split-seeds 0 1 \
    --final-epochs 200 \
    --batch-size 128 \
    --num-workers 4 \
    --lr 0.05 \
    --momentum 0.9 \
    --wd 5e-4 \
    --no-train-aug \
    --save-freq-epochs 50
