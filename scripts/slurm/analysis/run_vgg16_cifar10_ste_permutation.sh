#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_vgg16_cifar10_ste_perm_%j.out
#SBATCH --error=slurm_vgg16_cifar10_ste_perm_%j.err
#SBATCH --job-name=vgg16_c10_ste
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/cifar10/alignment/ste_permutation/seed0-seed1}"
DATA_PATH="${DATA_PATH:-./data}"
ALIGNMENT_STEPS="${ALIGNMENT_STEPS:-500}"
ALIGNMENT_BATCH_SIZE="${ALIGNMENT_BATCH_SIZE:-128}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-2048}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-128}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEED="${SEED:-0}"
LR="${LR:-1e-2}"
TAU="${TAU:-1.0}"
SINKHORN_ITERS="${SINKHORN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"

mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "VGG16 CIFAR10 STE Permutation"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "ALIGNMENT_STEPS: ${ALIGNMENT_STEPS}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

srun python scripts/analysis/run_vgg16_cifar10_ste_permutation.py \
    model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
    model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
    output_root="${OUTPUT_ROOT}" \
    data_path="${DATA_PATH}" \
    alignment_steps="${ALIGNMENT_STEPS}" \
    alignment_batch_size="${ALIGNMENT_BATCH_SIZE}" \
    calibration_size="${CALIBRATION_SIZE}" \
    evaluation_batch_size="${EVALUATION_BATCH_SIZE}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    num_workers="${NUM_WORKERS}" \
    seed="${SEED}" \
    lr="${LR}" \
    tau="${TAU}" \
    sinkhorn_iters="${SINKHORN_ITERS}" \
    log_interval="${LOG_INTERVAL}" \
    device=cuda

echo ""
echo "========================================"
echo "VGG16 CIFAR10 STE RUN COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
echo "Comparison table: ${OUTPUT_ROOT}/evaluation/comparison.json"
echo "Comparison plot: ${OUTPUT_ROOT}/evaluation/comparison.png"
