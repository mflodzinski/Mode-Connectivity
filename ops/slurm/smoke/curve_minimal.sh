#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_curve_%j.out
#SBATCH --error=slurm_smoke_curve_%j.err
#SBATCH --job-name=smoke_curve
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_banner "Smoke Check: Curve Training"

CONFIG_NAME="${CONFIG_NAME:-curves/runs/curve_seed0_seed1_reg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/smoke/curves/curve_seed0_seed1_reg_2epochs/checkpoints}"
EPOCHS="${EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-1}"

echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "EPOCHS: ${EPOCHS}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo ""

mc_run_module experiments.curves.garipov_curve \
  --config-name "${CONFIG_NAME}" \
  output_root="${OUTPUT_ROOT}" \
  epochs="${EPOCHS}" \
  save_freq="${SAVE_FREQ}" \
  use_wandb=false \
  no_train_aug=true

mc_run_module experiments.curves.evaluate_garipov_curve \
  --config-name "${CONFIG_NAME}" \
  output_root="${OUTPUT_ROOT}" \
  epochs="${EPOCHS}" \
  save_freq="${SAVE_FREQ}" \
  use_wandb=false \
  no_train_aug=true
