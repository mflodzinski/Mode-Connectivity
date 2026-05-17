#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_polychain_%x_%j.out
#SBATCH --error=slurm_polychain_%x_%j.err
#SBATCH --job-name=polychain
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch $0 <polygon|symmetry_plane|random_plane> <config-name> [hydra overrides...]"
  exit 1
fi

EXPERIMENT_KIND="$1"
CONFIG_NAME="$2"
shift 2

case "${EXPERIMENT_KIND}" in
  polygon)
    MODULE_NAME="experiments.curves.garipov_polygon"
    ;;
  symmetry_plane)
    MODULE_NAME="experiments.curves.symmetry_plane"
    ;;
  random_plane)
    MODULE_NAME="experiments.curves.random_plane"
    ;;
  *)
    echo "Unknown polychain experiment kind: ${EXPERIMENT_KIND}"
    exit 1
    ;;
esac

mc_setup_python_env
mc_banner "PolyChain Experiment"
echo "EXPERIMENT_KIND: ${EXPERIMENT_KIND}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo ""

mc_run_module "${MODULE_NAME}" --config-name "${CONFIG_NAME}" "$@"

if [ "${EVALUATE_AFTER_TRAIN:-false}" = "true" ]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-}"
  CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
  if [ -z "${CHECKPOINT_PATH}" ]; then
    if [ -z "${OUTPUT_ROOT}" ] || [ -z "${CHECKPOINT_EPOCH}" ]; then
      echo "Set CHECKPOINT_PATH directly, or set OUTPUT_ROOT and CHECKPOINT_EPOCH to enable evaluation."
      exit 1
    fi
    CHECKPOINT_PATH="${OUTPUT_ROOT}/checkpoint-${CHECKPOINT_EPOCH}.pt"
  fi
  EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/evaluations}"
  mc_eval_curve_checkpoint "${CHECKPOINT_PATH}" "${EVAL_DIR}" "PolyChain" "${NUM_BENDS:-3}"
fi
