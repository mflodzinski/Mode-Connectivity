#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_iter_bench_%x_%j.out
#SBATCH --error=slurm_lmc_iter_bench_%x_%j.err
#SBATCH --job-name=lmc_iter_bench
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <split_iter> [wm_seed]"
  exit 1
fi

SPLIT_ITER="$1"
WM_SEED="${2:-0}"

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

ENDPOINT_ROOT="results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug/iter${SPLIT_ITER}"
BENCHMARK_DIR="results/analysis/alignment_benchmark_iter_noaug/iter${SPLIT_ITER}/wm_seed${WM_SEED}"
W0="${ENDPOINT_ROOT}/seed0/checkpoint-200.pt"
W1="${ENDPOINT_ROOT}/seed1/checkpoint-200.pt"

mkdir -p "${BENCHMARK_DIR}"

echo "========================================"
echo "Controlled alignment benchmark"
echo "========================================"
echo "Split iteration: ${SPLIT_ITER}"
echo "Weight matching seed: ${WM_SEED}"
echo "w0: ${W0}"
echo "w1: ${W1}"
echo "Output: ${BENCHMARK_DIR}/results.json"
echo ""

srun python scripts/analysis/benchmark_alignment.py \
    --w0 "${W0}" \
    --w1 "${W1}" \
    --perm-seed 42 \
    --wm-seed "${WM_SEED}" \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 61 \
    --data-path ./data \
    --batch-size 128 \
    --output "${BENCHMARK_DIR}/results.json"
