#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_lmc_mnist_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_lmc_mnist_%j.err
#SBATCH --job-name=ext_sh_lmc_mn
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

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/mnist/endpoints/standard/seed0/checkpoint-50.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/mnist/endpoints/standard/seed1/checkpoint-50.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/vgg16/mnist/alignment/external_sinkhorn_original_lmc/seed0-seed1}"
DATA_PATH="${DATA_PATH:-./data}"
ALIGNMENT_EPOCHS="${ALIGNMENT_EPOCHS:-100}"
ALIGNMENT_BATCH_SIZE="${ALIGNMENT_BATCH_SIZE:-128}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-2048}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-128}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEED="${SEED:-0}"
LOSS_NAME="${LOSS_NAME:-midpoint}"
LR="${LR:-2e-2}"
TAU="${TAU:-1.0}"
SINKHORN_ITERS="${SINKHORN_ITERS:-20}"
SINKHORN_L="${SINKHORN_L:-1.0}"
IDENTITY_INIT="${IDENTITY_INIT:-false}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "Original External Sinkhorn LMC (MNIST)"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "ALIGNMENT_EPOCHS: ${ALIGNMENT_EPOCHS}"
echo "LOSS_NAME: ${LOSS_NAME}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/run_external_sinkhorn_original_lmc_vgg16_mnist.py \
    model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
    model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
    output_root="${OUTPUT_ROOT}" \
    data_path="${DATA_PATH}" \
    alignment_epochs="${ALIGNMENT_EPOCHS}" \
    alignment_batch_size="${ALIGNMENT_BATCH_SIZE}" \
    calibration_size="${CALIBRATION_SIZE}" \
    evaluation_batch_size="${EVALUATION_BATCH_SIZE}" \
    num_eval_points="${NUM_EVAL_POINTS}" \
    num_workers="${NUM_WORKERS}" \
    seed="${SEED}" \
    loss_name="${LOSS_NAME}" \
    lr="${LR}" \
    tau="${TAU}" \
    sinkhorn_iters="${SINKHORN_ITERS}" \
    sinkhorn_l="${SINKHORN_L}" \
    identity_init="${IDENTITY_INIT}" \
    log_interval="${LOG_INTERVAL}" \
    device=cuda

echo ""
echo "========================================"
echo "ORIGINAL EXTERNAL SINKHORN LMC MNIST RUN COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
echo "Metadata: ${OUTPUT_ROOT}/metadata.json"
echo "Comparison table: ${OUTPUT_ROOT}/evaluation/comparison.json"
echo "Comparison plot: ${OUTPUT_ROOT}/evaluation/comparison.png"
