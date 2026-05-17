#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_bezier_%x_%j.out
#SBATCH --error=slurm_bezier_%x_%j.err
#SBATCH --job-name=bezier_curve
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 <config-name> [hydra overrides...]"
  echo "Example: sbatch $0 curves/runs/curve_seed0_seed1_reg"
  exit 1
fi

CONFIG_NAME="$1"
shift

TRAIN_ONLY="${TRAIN_ONLY:-false}"
EVAL_ONLY="${EVAL_ONLY:-false}"

mc_setup_python_env
mc_banner "Bezier Curve Experiment"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "TRAIN_ONLY: ${TRAIN_ONLY}"
echo "EVAL_ONLY: ${EVAL_ONLY}"
echo ""

if [ "${EVAL_ONLY}" != "true" ]; then
  mc_run_module experiments.curves.garipov_curve --config-name "${CONFIG_NAME}" "$@"
fi

if [ "${TRAIN_ONLY}" != "true" ]; then
  mc_run_module experiments.curves.evaluate_garipov_curve --config-name "${CONFIG_NAME}" "$@"
fi
