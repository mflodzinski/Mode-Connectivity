#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_sinkhorn_sweep_%x_%j.out
#SBATCH --error=slurm_sinkhorn_sweep_%x_%j.err
#SBATCH --job-name=sinkhorn_sweep
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

CONFIG_NAME="${1:-sinkhorn/runs/vgg11_cifar_perm_only}"
if [ "$#" -gt 0 ]; then
  shift
fi

mc_setup_python_env
mc_require_external_file "external/sinkhorn-rebasin/examples/models/vgg.py"
mc_banner "Sinkhorn Alignment Sweep"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo ""

mc_run_module experiments.sinkhorn.vgg_cifar_alignment_sweep --config-name "${CONFIG_NAME}" "$@"
