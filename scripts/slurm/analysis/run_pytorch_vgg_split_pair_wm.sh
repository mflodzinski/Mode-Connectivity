#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_pair_wm_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_pair_wm_%x_%j.err
#SBATCH --job-name=pytorch_vgg_pair_wm
#SBATCH --gres=gpu:a40:1

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <label> [wm_seed]"
  echo "Labels: 100/100 80/120 30/170 8/192 6/194 0/200 independent"
  exit 1
fi

LABEL="$1"
WM_SEED="${2:-0}"

if [ "${LABEL}" = "100/100" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_100split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/100_100"
elif [ "${LABEL}" = "80/120" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_80split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/80_120"
elif [ "${LABEL}" = "30/170" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_30split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/30_170"
elif [ "${LABEL}" = "8/192" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_8split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/8_192"
elif [ "${LABEL}" = "6/194" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_6split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/6_194"
elif [ "${LABEL}" = "0/200" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_0split"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/0_200"
elif [ "${LABEL}" = "independent" ]; then
  PAIR_ROOT="results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing"
  RESULT_DIR="results/analysis/pytorch_vgg_split_wm/independent"
else
  echo "Unknown label: ${LABEL}"
  exit 1
fi

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mpl-${USER}"
mkdir -p "${MPLCONFIGDIR}" "${RESULT_DIR}"

echo "========================================"
echo "PyTorch-VGG split WM benchmark"
echo "========================================"
echo "Label: ${LABEL}"
echo "WM seed: ${WM_SEED}"
echo "Pair root: ${PAIR_ROOT}"
echo "Output: ${RESULT_DIR}/results.json"
echo ""

srun python scripts/analysis/benchmark_alignment.py \
  --w0 "${PAIR_ROOT}/seed0/checkpoint-200.pt" \
  --w1 "${PAIR_ROOT}/seed1/checkpoint-200.pt" \
  --perm-seed 42 \
  --wm-seed "${WM_SEED}" \
  --method weight_matching \
  --max-iter 100 \
  --num-eval-points 31 \
  --data-path ./data \
  --batch-size 128 \
  --output "${RESULT_DIR}/results.json"
