#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_plot_all_%j.out
#SBATCH --error=slurm_lmc_iter_plot_all_%j.err
#SBATCH --job-name=lmc_iter_plot_all

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

METRICS=(
  test_loss_barrier_rel
  train_loss_barrier_rel
  test_acc_barrier_rel
  train_acc_barrier_rel
)

echo "========================================"
echo "Iteration-split barrier vs distance plots"
echo "========================================"
echo "Metrics: ${METRICS[*]}"
echo ""

for METRIC in "${METRICS[@]}"; do
  echo "Running metric: ${METRIC}"
  srun --exclusive -N1 -n1 python scripts/plot/plot_iter_split_barrier_vs_distance.py \
      --endpoints-root results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug \
      --splits 0 25 100 1000 5000 \
      --metric "${METRIC}" \
      --output "plots/lmc_iter_noaug_${METRIC}_vs_distance.png" \
      --csv-output "plots/lmc_iter_noaug_barrier_vs_distance.csv"
done

echo ""
echo "Done. Outputs written to plots/."
