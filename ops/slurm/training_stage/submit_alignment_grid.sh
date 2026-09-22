#!/bin/bash
# Re-run only Sinkhorn and fixed-permutation Sinkhorn+scale alignment for the
# already-trained VGG11/CIFAR-10 training-stage checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"
mc_setup_python_env

SOURCE_ROOT="${1:?Usage: $0 SOURCE_ROOT [SWEEP_ROOT]}"
SWEEP_ROOT="${2:-results/training_stage_vgg11_alignment_grid}"

# The retained VGG11 sweep found useful rates around 0.005--0.01, whereas the
# original training-stage default was 0.1.  These environment variables make
# the compact grid editable without changing Python code.
BASE_LRS="${BASE_LRS:-0.005,0.01,0.05}"
TAUS="${TAUS:-1.0,1.5}"
SINKHORN_L="${SINKHORN_L:-1.0}"
SCALE_LRS="${SCALE_LRS:-0.01,0.05}"
LAMBDA_SCALES="${LAMBDA_SCALES:-0.0001,0.001,0.01}"

PARTITION="${MC_PARTITION:-general}"
QOS="${MC_QOS:-short}"
GRES="${MC_GRES:-gpu:a40:1}"
CONCURRENCY="${MC_CONCURRENCY:-4}"
SCHEDULER_ARGS=("--partition=${PARTITION}" "--qos=${QOS}")
if [ -n "${MC_ACCOUNT:-}" ]; then
  SCHEDULER_ARGS+=("--account=${MC_ACCOUNT}")
fi

mkdir -p "${SWEEP_ROOT}/logs"
COMMON_ARGS=(
  --source-root "${SOURCE_ROOT}"
  --sweep-root "${SWEEP_ROOT}"
  --base-lrs "${BASE_LRS}"
  --taus "${TAUS}"
  --sinkhorn-l "${SINKHORN_L}"
  --scale-lrs "${SCALE_LRS}"
  --lambda-scales "${LAMBDA_SCALES}"
)

BASE_COUNT="$(python -m experiments.training_stage.alignment_grid count "${COMMON_ARGS[@]}" --phase base)"
SCALE_COUNT="$(python -m experiments.training_stage.alignment_grid count "${COMMON_ARGS[@]}" --phase scale)"
if ! [[ "${BASE_COUNT}" =~ ^[1-9][0-9]*$ && "${SCALE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid grid sizes: base=${BASE_COUNT}, scale=${SCALE_COUNT}" >&2
  exit 1
fi

echo "Source checkpoints: ${SOURCE_ROOT}"
echo "Sweep output:       ${SWEEP_ROOT}"
echo "Base tasks:         ${BASE_COUNT} (${BASE_LRS}; tau=${TAUS}; l=${SINKHORN_L})"
echo "Scale tasks:        ${SCALE_COUNT} (${SCALE_LRS}; lambda=${LAMBDA_SCALES})"

BASE_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_sinkhorn_grid --ntasks=1 --cpus-per-task=2 --mem=4GB \
  --time=03:00:00 --signal=USR1@120 --gres="${GRES}" \
  --array="0-$((BASE_COUNT - 1))%${CONCURRENCY}" \
  --output="${SWEEP_ROOT}/logs/%x_%A_%a.out" \
  --error="${SWEEP_ROOT}/logs/%x_%A_%a.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${SCRIPT_DIR}/run_alignment_grid.sh" base "${COMMON_ARGS[@]}")"
BASE_JOB="${BASE_JOB%%;*}"

SELECT_BASE_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_select_sinkhorn --ntasks=1 --cpus-per-task=1 --mem=4GB \
  --time=00:20:00 --dependency="afterok:${BASE_JOB}" \
  --output="${SWEEP_ROOT}/logs/%x_%j.out" \
  --error="${SWEEP_ROOT}/logs/%x_%j.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${SCRIPT_DIR}/run_alignment_grid.sh" select-base "${COMMON_ARGS[@]}")"
SELECT_BASE_JOB="${SELECT_BASE_JOB%%;*}"

SCALE_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_scale_grid --ntasks=1 --cpus-per-task=2 --mem=4GB \
  --time=01:00:00 --signal=USR1@120 --gres="${GRES}" \
  --dependency="afterok:${SELECT_BASE_JOB}" \
  --array="0-$((SCALE_COUNT - 1))%${CONCURRENCY}" \
  --output="${SWEEP_ROOT}/logs/%x_%A_%a.out" \
  --error="${SWEEP_ROOT}/logs/%x_%A_%a.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${SCRIPT_DIR}/run_alignment_grid.sh" scale "${COMMON_ARGS[@]}")"
SCALE_JOB="${SCALE_JOB%%;*}"

SELECT_SCALE_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_select_scale --ntasks=1 --cpus-per-task=1 --mem=4GB \
  --time=00:20:00 --dependency="afterok:${SCALE_JOB}" \
  --output="${SWEEP_ROOT}/logs/%x_%j.out" \
  --error="${SWEEP_ROOT}/logs/%x_%j.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${SCRIPT_DIR}/run_alignment_grid.sh" select-scale "${COMMON_ARGS[@]}")"
SELECT_SCALE_JOB="${SELECT_SCALE_JOB%%;*}"

echo "Submitted base array:       ${BASE_JOB}"
echo "Submitted base selection:   ${SELECT_BASE_JOB}"
echo "Submitted scale array:      ${SCALE_JOB}"
echo "Submitted scale selection:  ${SELECT_SCALE_JOB}"
echo "Final artifacts will be linked under ${SWEEP_ROOT}/selected/."
