#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_verify_%x_%j.out
#SBATCH --error=slurm_lmc_verify_%x_%j.err
#SBATCH --job-name=lmc_verify
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 <pair-root> [num-points]"
  echo "Example: sbatch $0 results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_30split"
  exit 1
fi

PAIR_ROOT="$1"
NUM_POINTS="${2:-61}"
ENDPOINT0="${PAIR_ROOT}/seed0/checkpoint-200.pt"
ENDPOINT1="${PAIR_ROOT}/seed1/checkpoint-200.pt"
OUTPUT_DIR="${PAIR_ROOT}/evaluation"

mc_setup_python_env
mc_banner "Verify LMC Connectivity"
echo "PAIR_ROOT: ${PAIR_ROOT}"
echo "ENDPOINT0: ${ENDPOINT0}"
echo "ENDPOINT1: ${ENDPOINT1}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"
mc_run_module experiments.curves.evaluate_paths \
  --mode linear \
  --init-start "${ENDPOINT0}" \
  --init-end "${ENDPOINT1}" \
  --num-points "${NUM_POINTS}" \
  --dataset "${MC_DATASET:-CIFAR10}" \
  --model "${MC_MODEL:-VGG16}" \
  --data-path "${MC_DATA_PATH:-./data}" \
  --dir "${OUTPUT_DIR}" \
  --use-test \
  --transform "${MC_TRANSFORM:-VGG}"
