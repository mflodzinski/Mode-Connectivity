#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_shared_%j.out
#SBATCH --error=slurm_lmc_iter_shared_%j.err
#SBATCH --job-name=lmc_iter_shared
#SBATCH --gres=gpu:a40:1

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH}"

OUTPUT_ROOT="results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug"
SHARED_ITERS=(0 25 100 1000 5000)

echo "========================================"
echo "Shared trunk training with iteration checkpoints"
echo "========================================"
echo "Output root: ${OUTPUT_ROOT}"
echo "Shared checkpoints: ${SHARED_ITERS[*]}"
echo "No train augmentation"
echo ""

srun python scripts/train/run_lmc_connected_pair_by_iteration.py \
    --mode shared \
    --output-root "${OUTPUT_ROOT}" \
    --shared-iters "${SHARED_ITERS[@]}" \
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
