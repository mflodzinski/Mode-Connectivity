#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_open_lth_split_barriers_%x_%j.out
#SBATCH --error=slurm_open_lth_split_barriers_%x_%j.err
#SBATCH --job-name=open_lth_split_barriers

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "open_lth shared-split barrier plots"
echo "========================================"
echo "Results root: results/vgg16/cifar10/endpoints/open_lth_shared_split"
echo "Splits: 0 25 100 500 1000"
echo "Metrics: train_loss_barrier test_loss_barrier train_acc_barrier test_acc_barrier"
echo ""

srun python scripts/plot/plot_open_lth_split_barriers.py \
  --results-root results/vgg16/cifar10/endpoints/open_lth_shared_split \
  --splits 0 25 100 500 1000 \
  --metrics train_loss_barrier test_loss_barrier train_acc_barrier test_acc_barrier \
  --output-prefix plots/open_lth_shared_split \
  --csv-output plots/open_lth_shared_split_barriers.csv
