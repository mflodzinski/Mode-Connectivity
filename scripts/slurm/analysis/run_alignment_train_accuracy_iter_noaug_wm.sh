#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=lmc_iter_wm_plot
#SBATCH --output=slurm_lmc_iter_wm_plot_%j.out
#SBATCH --error=slurm_lmc_iter_wm_plot_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "Controlled benchmark WM aggregation plot"
echo "========================================"
echo "Results root: results/analysis/alignment_benchmark_iter_noaug"
echo "Splits: 0 25 100 1000 5000"
echo "WM seeds: 0 1 2"

srun python scripts/plot/plot_alignment_train_accuracy_iter_noaug_wm.py \
  --results-root results/analysis/alignment_benchmark_iter_noaug \
  --splits 0 25 100 1000 5000 \
  --wm-seeds 0 1 2 \
  --output plots/alignment_train_accuracy_iter_noaug_wm.png \
  --csv-output plots/alignment_train_accuracy_iter_noaug_wm.csv
