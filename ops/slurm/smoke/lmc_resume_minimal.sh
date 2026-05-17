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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Smoke Check: LMC Resume"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/smoke/lmc/resume_shared_checkpoint_151}"
FINAL_EPOCHS="${FINAL_EPOCHS:-151}"

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "FINAL_EPOCHS: ${FINAL_EPOCHS}"
echo ""

mc_run_module experiments.lmc.pytorch_vgg16_lmc_connected_pair \
  ++output_root="${OUTPUT_ROOT}" \
  ++final_epochs="${FINAL_EPOCHS}" \
  ++save_every=1 \
  ++epoch_print_freq=1
