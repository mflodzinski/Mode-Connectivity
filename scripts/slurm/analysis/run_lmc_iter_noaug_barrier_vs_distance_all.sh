#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_plot_all_%j.out
#SBATCH --error=slurm_lmc_iter_plot_all_%j.err
#SBATCH --job-name=lmc_iter_plot_all
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

echo "========================================"
echo "Iteration-split barrier vs distance plots"
echo "========================================"
echo "Metrics: test_loss_barrier_rel train_loss_barrier_rel test_acc_barrier_rel train_acc_barrier_rel"
echo ""

srun python scripts/plot/plot_iter_split_barrier_vs_distance.py \
    --endpoints-root results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug \
    --splits 0 25 100 1000 5000 \
    --metrics test_loss_barrier_rel train_loss_barrier_rel test_acc_barrier_rel train_acc_barrier_rel \
    --output-prefix plots/lmc_iter_noaug \
    --csv-output plots/lmc_iter_noaug_barrier_vs_distance.csv

echo ""
echo "Done. Outputs written to plots/."
