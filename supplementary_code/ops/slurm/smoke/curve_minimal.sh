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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Smoke Check: Curve Training"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/smoke/curves/curve_seed0_seed1_reg_2epochs/checkpoints}"
EPOCHS="${EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-1}"

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "EPOCHS: ${EPOCHS}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo ""

mc_run_module experiments.curves.garipov_curve \
  ++output_root="${OUTPUT_ROOT}" \
  ++epochs="${EPOCHS}" \
  ++save_freq="${SAVE_FREQ}" \
  ++num_workers="${SLURM_CPUS_PER_TASK}" \
  ++use_wandb=false \
  ++no_train_aug=true

mc_run_module experiments.curves.evaluate_garipov_curve \
  ++output_root="${OUTPUT_ROOT}" \
  ++epochs="${EPOCHS}" \
  ++save_freq="${SAVE_FREQ}" \
  ++num_workers="${SLURM_CPUS_PER_TASK}" \
  ++use_wandb=false \
  ++no_train_aug=true
