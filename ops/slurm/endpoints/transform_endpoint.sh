#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_transform_endpoint_%j.out
#SBATCH --error=slurm_transform_endpoint_%j.err
#SBATCH --job-name=endpoint_transform
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch $0 <mirror|random> <source-checkpoint> <output-checkpoint> [extra network_transform args...]"
  exit 1
fi

MODE="$1"
SOURCE_CHECKPOINT="$2"
OUTPUT_CHECKPOINT="$3"
shift 3

MODEL_NAME="${MODEL_NAME:-VGG16}"
DATASET_NAME="${DATASET_NAME:-CIFAR10}"
DATA_PATH="${DATA_PATH:-./data}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PERM_SEED="${PERM_SEED:-42}"
VERIFY="${VERIFY:-true}"
FULL_DATASET_VERIFY="${FULL_DATASET_VERIFY:-true}"
EVALUATE_LINEAR="${EVALUATE_LINEAR:-true}"
NUM_POINTS="${NUM_POINTS:-61}"

mc_setup_python_env
mc_banner "Endpoint Transform"
echo "MODE: ${MODE}"
echo "SOURCE_CHECKPOINT: ${SOURCE_CHECKPOINT}"
echo "OUTPUT_CHECKPOINT: ${OUTPUT_CHECKPOINT}"
echo "EVALUATE_LINEAR: ${EVALUATE_LINEAR}"
echo ""

transform_args=(
  --mode "${MODE}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --output "${OUTPUT_CHECKPOINT}"
  --model "${MODEL_NAME}"
  --dataset "${DATASET_NAME}"
  --data-path "${DATA_PATH}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)
if [ "${MODE}" = "random" ]; then
  transform_args+=(--perm-seed "${PERM_SEED}")
fi
if [ "${VERIFY}" = "true" ]; then
  transform_args+=(--verify)
fi
if [ "${FULL_DATASET_VERIFY}" = "true" ]; then
  transform_args+=(--full-dataset-verify)
fi
transform_args+=("$@")

mc_run_module tools.verification.network_transform "${transform_args[@]}"

if [ "${EVALUATE_LINEAR}" = "true" ]; then
  EVAL_DIR="${EVAL_DIR:-$(dirname "${OUTPUT_CHECKPOINT}")/../evaluations}"
  mkdir -p "${EVAL_DIR}"
  mc_run_module experiments.curves.evaluate_paths \
    --mode linear \
    --dir "${EVAL_DIR}" \
    --init-start "${SOURCE_CHECKPOINT}" \
    --init-end "${OUTPUT_CHECKPOINT}" \
    --num-points "${NUM_POINTS}" \
    --dataset "${DATASET_NAME}" \
    --data-path "${DATA_PATH}" \
    --model "${MODEL_NAME}" \
    --transform "${MC_TRANSFORM:-VGG}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --use-test
fi
