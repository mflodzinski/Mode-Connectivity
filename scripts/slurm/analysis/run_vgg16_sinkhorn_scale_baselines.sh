#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_vgg16_sinkhorn_scale_%j.out
#SBATCH --error=slurm_vgg16_sinkhorn_scale_%j.err
#SBATCH --job-name=vgg16_sinkhorn_scale
#SBATCH --gres=gpu:a40:1

# Run the VGG16/CIFAR10 baseline comparison pipeline:
# 1. No alignment
# 2. Sinkhorn permutation-only re-basin
# 3. Sinkhorn + diagonal scaling

set -euo pipefail

# Activate virtual environment
source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

# Navigate to project directory
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

# Add project root to Python path
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

# Cluster-friendly defaults. Override via environment variables before sbatch if needed.
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/cifar10/alignment/sinkhorn_scale_prototype/seed0-seed1}"
DATA_PATH="${DATA_PATH:-./data}"
ALIGNMENT_STEPS="${ALIGNMENT_STEPS:-500}"
ALIGNMENT_BATCH_SIZE="${ALIGNMENT_BATCH_SIZE:-128}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-2048}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEED="${SEED:-0}"
TAU="${TAU:-1.0}"
SINKHORN_ITERS="${SINKHORN_ITERS:-20}"
LR="${LR:-1e-2}"
LAMBDA_SCALE="${LAMBDA_SCALE:-1e-5}"

mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "VGG16/CIFAR10 Sinkhorn Baselines"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "ALIGNMENT_STEPS: ${ALIGNMENT_STEPS}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

srun python scripts/analysis/run_vgg16_sinkhorn_scale_baselines.py \
    model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
    model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
    output_root="${OUTPUT_ROOT}" \
    data_path="${DATA_PATH}" \
    alignment_steps="${ALIGNMENT_STEPS}" \
    alignment_batch_size="${ALIGNMENT_BATCH_SIZE}" \
    calibration_size="${CALIBRATION_SIZE}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    evaluation_batch_size="${EVALUATION_BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    seed="${SEED}" \
    tau="${TAU}" \
    sinkhorn_iters="${SINKHORN_ITERS}" \
    lr="${LR}" \
    lambda_scale="${LAMBDA_SCALE}" \
    device=cuda

echo ""
echo "========================================"
echo "BASELINE PIPELINE COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
echo "Summary: ${OUTPUT_ROOT}/pipeline_summary.json"
echo "Comparison table: ${OUTPUT_ROOT}/evaluation/comparison.json"
echo "Comparison plot: ${OUTPUT_ROOT}/evaluation/comparison.png"
