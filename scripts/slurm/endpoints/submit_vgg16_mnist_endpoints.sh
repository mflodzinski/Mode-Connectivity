#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_vgg16_mnist_endpoints_%j.out
#SBATCH --error=slurm_vgg16_mnist_endpoints_%j.err
#SBATCH --job-name=vgg16_mnist
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/endpoints/standard}"
DATA_PATH="${DATA_PATH:-./data}"
EPOCHS="${EPOCHS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"
LR="${LR:-0.05}"
MOMENTUM="${MOMENTUM:-0.9}"
WD="${WD:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEEDS="${SEEDS:-[0,1]}"
USE_WANDB="${USE_WANDB:-false}"

mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "VGG16 MNIST Endpoints"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "DATA_PATH: ${DATA_PATH}"
echo "EPOCHS: ${EPOCHS}"
echo "SEEDS: ${SEEDS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

srun python scripts/train/run_vgg16_mnist_endpoints.py \
    output_root="${OUTPUT_ROOT}" \
    data_path="${DATA_PATH}" \
    epochs="${EPOCHS}" \
    save_freq="${SAVE_FREQ}" \
    lr="${LR}" \
    momentum="${MOMENTUM}" \
    wd="${WD}" \
    batch_size="${BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    seeds="${SEEDS}" \
    use_wandb="${USE_WANDB}"

echo ""
echo "========================================"
echo "VGG16 MNIST ENDPOINT TRAINING COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
