#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_sweep_%A_%a.out
#SBATCH --error=slurm_external_sinkhorn_sweep_%A_%a.err
#SBATCH --job-name=ext_sinkhorn_sw
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-23

# Hyperparameter sweep for the vendored external/sinkhorn-rebasin baseline.
# Grid:
#   alignment_steps: 500, 1000, 2000
#   tau: 0.5, 1.0
#   lr: 0.3, 1.0, 3.0, 10.0
#   sinkhorn_l: 1.0

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

if [ ! -f "${PROJECT_ROOT}/external/sinkhorn-rebasin/examples/models/vgg.py" ]; then
    echo "Missing external/sinkhorn-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/sinkhorn-rebasin"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load graphviz >/dev/null 2>&1 || true
fi

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-results/vgg16/cifar10/alignment/external_sinkhorn_rebasin_sweep/seed0-seed1}"
DATA_PATH="${DATA_PATH:-./data}"
ALIGNMENT_BATCH_SIZE="${ALIGNMENT_BATCH_SIZE:-128}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-2048}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-128}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
SEED="${SEED:-0}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-midpoint}"
MIDPOINT_ALPHA="${MIDPOINT_ALPHA:-0.5}"
IDENTITY_INIT="${IDENTITY_INIT:-true}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
LOG_EVAL_BATCHES="${LOG_EVAL_BATCHES:-4}"

ALIGNMENT_STEPS_VALUES=(500 1000 2000)
TAU_VALUES=(0.5 1.0)
LR_VALUES=(0.3 1.0 3.0 10.0)
SINKHORN_L_VALUES=(1.0)

NUM_STEPS=${#ALIGNMENT_STEPS_VALUES[@]}
NUM_TAU=${#TAU_VALUES[@]}
NUM_LR=${#LR_VALUES[@]}
NUM_SINKHORN_L=${#SINKHORN_L_VALUES[@]}
TOTAL_RUNS=$((NUM_STEPS * NUM_TAU * NUM_LR * NUM_SINKHORN_L))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "${TASK_ID}" -ge "${TOTAL_RUNS}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID} exceeds total runs ${TOTAL_RUNS}"
    exit 1
fi

INDEX=${TASK_ID}
STEPS_INDEX=$((INDEX / (NUM_TAU * NUM_LR * NUM_SINKHORN_L)))
INDEX=$((INDEX % (NUM_TAU * NUM_LR * NUM_SINKHORN_L)))
TAU_INDEX=$((INDEX / (NUM_LR * NUM_SINKHORN_L)))
INDEX=$((INDEX % (NUM_LR * NUM_SINKHORN_L)))
LR_INDEX=$((INDEX / NUM_SINKHORN_L))
SINKHORN_L_INDEX=$((INDEX % NUM_SINKHORN_L))

ALIGNMENT_STEPS="${ALIGNMENT_STEPS_VALUES[$STEPS_INDEX]}"
TAU="${TAU_VALUES[$TAU_INDEX]}"
LR="${LR_VALUES[$LR_INDEX]}"
SINKHORN_L="${SINKHORN_L_VALUES[$SINKHORN_L_INDEX]}"

sanitize_float() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/_}"
    printf '%s' "${value}"
}

OUTPUT_TAG="steps${ALIGNMENT_STEPS}_tau$(sanitize_float "${TAU}")_lr$(sanitize_float "${LR}")_l$(sanitize_float "${SINKHORN_L}")"
OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${OUTPUT_TAG}"
mkdir -p "${OUTPUT_ROOT}"

echo "========================================"
echo "External Sinkhorn-Rebasin Sweep"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "TASK_ID: ${TASK_ID}/${TOTAL_RUNS}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "ALIGNMENT_STEPS: ${ALIGNMENT_STEPS}"
echo "TAU: ${TAU}"
echo "LR: ${LR}"
echo "SINKHORN_L: ${SINKHORN_L}"
echo "TRAIN_OBJECTIVE: ${TRAIN_OBJECTIVE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo ""

srun python scripts/analysis/run_external_sinkhorn_baseline.py \
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
    sinkhorn_iters=20 \
    sinkhorn_l="${SINKHORN_L}" \
    train_objective="${TRAIN_OBJECTIVE}" \
    midpoint_alpha="${MIDPOINT_ALPHA}" \
    log_interval="${LOG_INTERVAL}" \
    log_eval_batches="${LOG_EVAL_BATCHES}" \
    identity_init="${IDENTITY_INIT}" \
    device=cuda

echo ""
echo "========================================"
echo "EXTERNAL SINKHORN SWEEP RUN COMPLETE"
echo "========================================"
echo "Artifacts written under: ${OUTPUT_ROOT}"
echo "Metadata: ${OUTPUT_ROOT}/metadata.json"
echo "Comparison table: ${OUTPUT_ROOT}/evaluation/comparison.json"
echo "Comparison plot: ${OUTPUT_ROOT}/evaluation/comparison.png"
