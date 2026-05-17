#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_resume_%x_%j.out
#SBATCH --error=slurm_lmc_resume_%x_%j.err
#SBATCH --job-name=lmc_resume
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 <config-name> [hydra overrides...]"
  exit 1
fi

CONFIG_NAME="$1"
shift

mc_setup_python_env
mc_banner "LMC Shared Checkpoint Resume"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo ""

mc_run_module experiments.lmc.pytorch_vgg16_lmc_connected_pair --config-name "${CONFIG_NAME}" "$@"
