#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_xor_train_linear_barriers_%j.out
#SBATCH --error=slurm_xor_train_linear_barriers_%j.err
#SBATCH --job-name=xor_train_lin

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

OUTPUT_DIR="${OUTPUT_DIR:-results/xor/xor_5h_trained_linear_pairs}"
HIDDEN_SIZE="${HIDDEN_SIZE:-5}"
NUM_NETWORKS="${NUM_NETWORKS:-10}"
SEEDS="${SEEDS:-}"
PAIRS="${PAIRS:-}"
MAX_ENDPOINT_LOSS="${MAX_ENDPOINT_LOSS:-0.02}"
CURVE_EVAL_POINTS="${CURVE_EVAL_POINTS:-61}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS:-}"
TRAIN_LR="${TRAIN_LR:-}"
VERBOSE="${VERBOSE:-false}"

echo "========================================"
echo "XOR Train + Linear Pair Barriers"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "HIDDEN_SIZE: ${HIDDEN_SIZE}"
echo "NUM_NETWORKS: ${NUM_NETWORKS}"
echo "SEEDS: ${SEEDS:-<default>}"
echo "PAIRS: ${PAIRS:-<all kept>}"
echo "MAX_ENDPOINT_LOSS: ${MAX_ENDPOINT_LOSS}"
echo "CURVE_EVAL_POINTS: ${CURVE_EVAL_POINTS}"
echo "TRAIN_MAX_EPOCHS: ${TRAIN_MAX_EPOCHS:-<default>}"
echo "TRAIN_LR: ${TRAIN_LR:-<default>}"
echo "VERBOSE: ${VERBOSE}"
echo ""

args=(
    --output "${OUTPUT_DIR}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-networks "${NUM_NETWORKS}"
    --max-endpoint-loss "${MAX_ENDPOINT_LOSS}"
    --curve-eval-points "${CURVE_EVAL_POINTS}"
)

if [ -n "${SEEDS}" ]; then
    args+=(--seeds "${SEEDS}")
fi
if [ -n "${PAIRS}" ]; then
    args+=(--pairs "${PAIRS}")
fi
if [ -n "${TRAIN_MAX_EPOCHS}" ]; then
    args+=(--train-max-epochs "${TRAIN_MAX_EPOCHS}")
fi
if [ -n "${TRAIN_LR}" ]; then
    args+=(--train-lr "${TRAIN_LR}")
fi
if [ "${VERBOSE}" = "true" ]; then
    args+=(--verbose)
fi

srun python scripts/experiments/xor_train_linear_barriers.py "${args[@]}"

echo ""
echo "========================================"
echo "XOR TRAIN + LINEAR PAIR BARRIERS COMPLETE"
echo "========================================"
echo "Results written under: ${OUTPUT_DIR}"
