#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_lmc_resume_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_lmc_resume_%x_%j.err
#SBATCH --job-name=pytorch_vgg_lmc_resume
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <config-name>"
  exit 1
fi

CONFIG_NAME="$1"

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "PyTorch VGG16 shared-checkpoint split training"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo ""

srun python scripts/train/run_pytorch_vgg16_lmc_connected_pair.py \
  --config-name "${CONFIG_NAME}"
