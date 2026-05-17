#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_endpoints_%x_%j.out
#SBATCH --error=slurm_endpoints_%x_%j.err
#SBATCH --job-name=endpoints
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

CONFIG_NAME="${1:-curves/runs/endpoints_standard}"
if [ "$#" -gt 0 ]; then
  shift
fi

mc_setup_python_env
mc_banner "Endpoint Training"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo ""

mc_run_module experiments.curves.garipov_endpoints --config-name "${CONFIG_NAME}" "$@"
