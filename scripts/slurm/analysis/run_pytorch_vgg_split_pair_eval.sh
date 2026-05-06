#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_pair_eval_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_pair_eval_%x_%j.err
#SBATCH --job-name=pytorch_vgg_pair_eval
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch $0 <label>"
  echo "Labels: 100/100 80/120 30/170 8/192 6/194 5/195 4/196 3/197 2/198 1/199 0/200 independent"
  exit 1
fi

LABEL="$1"

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "Evaluate pytorch-vgg split pair"
echo "========================================"
echo "Label: ${LABEL}"
echo ""

srun python scripts/analysis/evaluate_pytorch_vgg_split_suite.py \
  --labels "${LABEL}" \
  --data-root ./data \
  --batch-size 128 \
  --workers 1 \
  --num-points 21
