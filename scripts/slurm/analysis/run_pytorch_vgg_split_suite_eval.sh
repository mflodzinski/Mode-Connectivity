#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_split_eval_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_split_eval_%x_%j.err
#SBATCH --job-name=pytorch_vgg_split_eval
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "Evaluate pytorch-vgg split suite"
echo "========================================"
echo ""

srun python scripts/analysis/evaluate_pytorch_vgg_split_suite.py \
  --data-root ./data \
  --batch-size 128 \
  --workers 2 \
  --num-points 61
