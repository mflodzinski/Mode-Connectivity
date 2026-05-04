#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_split_plot_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_split_plot_%x_%j.err
#SBATCH --job-name=pytorch_vgg_split_plot

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "Plot pytorch-vgg split suite"
echo "========================================"
echo ""

srun python scripts/plot/plot_pytorch_vgg_split_suite.py \
  --output-prefix plots/pytorch_vgg_split_suite \
  --csv-output plots/pytorch_vgg_split_suite.csv
