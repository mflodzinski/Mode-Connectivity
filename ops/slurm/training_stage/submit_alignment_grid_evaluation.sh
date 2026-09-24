#!/bin/bash
# Evaluate the globally selected VGG11 Sinkhorn and scale artifacts on frozen
# train/test subsets without rerunning alignment optimization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"
mc_setup_python_env

SOURCE_ROOT="${1:?Usage: $0 SOURCE_ROOT [SWEEP_ROOT]}"
SWEEP_ROOT="${2:-results/training_stage_vgg11_alignment_grid}"
PARTITION="${MC_PARTITION:-general}"
QOS="${MC_QOS:-short}"
GRES="${MC_GRES:-gpu:a40:1}"
CONCURRENCY="${MC_CONCURRENCY:-4}"
SCHEDULER_ARGS=("--partition=${PARTITION}" "--qos=${QOS}")
if [ -n "${MC_ACCOUNT:-}" ]; then
  SCHEDULER_ARGS+=("--account=${MC_ACCOUNT}")
fi

COMMON_ARGS=(--source-root "${SOURCE_ROOT}" --sweep-root "${SWEEP_ROOT}")
COUNT="$(python -m experiments.training_stage.alignment_grid_evaluation count "${COMMON_ARGS[@]}")"
if ! [[ "${COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid selected-pair count: ${COUNT}" >&2
  exit 1
fi

LOGS="${SWEEP_ROOT}/selected/evaluation_logs"
mkdir -p "${LOGS}"
RUNNER="${SCRIPT_DIR}/run_alignment_grid_evaluation.sh"

EVALUATE_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_grid_evaluate --ntasks=1 --cpus-per-task=2 --mem=4GB \
  --time=00:30:00 --signal=USR1@120 --gres="${GRES}" \
  --array="0-$((COUNT - 1))%${CONCURRENCY}" \
  --output="${LOGS}/%x_%A_%a.out" \
  --error="${LOGS}/%x_%A_%a.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${RUNNER}" evaluate "${COMMON_ARGS[@]}")"
EVALUATE_JOB="${EVALUATE_JOB%%;*}"

REPORT_JOB="$(sbatch --parsable \
  "${SCHEDULER_ARGS[@]}" \
  --job-name=stage_grid_report --ntasks=1 --cpus-per-task=2 --mem=4GB \
  --time=00:20:00 --dependency="afterok:${EVALUATE_JOB}" \
  --output="${LOGS}/%x_%j.out" \
  --error="${LOGS}/%x_%j.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
  "${RUNNER}" report "${COMMON_ARGS[@]}")"
REPORT_JOB="${REPORT_JOB%%;*}"

echo "Selected pairs: ${COUNT}"
echo "Submitted evaluation array: ${EVALUATE_JOB}"
echo "Submitted report:           ${REPORT_JOB}"
echo "Report destination:         ${SWEEP_ROOT}/selected/report"
