#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_lmc_resume_%j.out
#SBATCH --error=slurm_smoke_lmc_resume_%j.err
#SBATCH --job-name=smoke_lmc_res
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_banner "Smoke Check: LMC Resume"

CONFIG_NAME="${CONFIG_NAME:-lmc/runs/resume_shared_checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/smoke/lmc/resume_shared_checkpoint_151}"
FINAL_EPOCHS="${FINAL_EPOCHS:-151}"

echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "FINAL_EPOCHS: ${FINAL_EPOCHS}"
echo ""

mc_run_module experiments.lmc.pytorch_vgg16_lmc_connected_pair \
  --config-name "${CONFIG_NAME}" \
  output_root="${OUTPUT_ROOT}" \
  final_epochs="${FINAL_EPOCHS}" \
  save_every=1 \
  epoch_print_freq=1
