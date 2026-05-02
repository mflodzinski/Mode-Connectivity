#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_plot_%x_%j.out
#SBATCH --error=slurm_lmc_iter_plot_%x_%j.err
#SBATCH --job-name=lmc_iter_plot

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <metric>"
  echo "Metrics: test_loss_barrier_rel train_loss_barrier_rel test_acc_barrier_rel train_acc_barrier_rel"
  exit 1
fi

METRIC="$1"

case "${METRIC}" in
  test_loss_barrier_rel|train_loss_barrier_rel|test_acc_barrier_rel|train_acc_barrier_rel)
    ;;
  *)
    echo "Unsupported metric: ${METRIC}"
    exit 1
    ;;
esac

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}"

OUTPUT_PNG="plots/lmc_iter_noaug_${METRIC}_vs_distance.png"
OUTPUT_CSV="plots/lmc_iter_noaug_barrier_vs_distance.csv"

echo "========================================"
echo "Iteration-split barrier vs distance plot"
echo "========================================"
echo "Metric: ${METRIC}"
echo "Output PNG: ${OUTPUT_PNG}"
echo "Output CSV: ${OUTPUT_CSV}"
echo ""

srun python scripts/plot/plot_iter_split_barrier_vs_distance.py \
    --endpoints-root results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug \
    --splits 0 25 100 1000 5000 \
    --metric "${METRIC}" \
    --output "${OUTPUT_PNG}" \
    --csv-output "${OUTPUT_CSV}"
