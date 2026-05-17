#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_curve_eval_%x_%j.out
#SBATCH --error=slurm_curve_eval_%x_%j.err
#SBATCH --job-name=curve_eval
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

if [ "$#" -ne 4 ]; then
  echo "Usage: sbatch $0 <checkpoint-path> <output-dir> <Bezier|PolyChain> <num-bends>"
  exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"
CURVE_NAME="$3"
NUM_BENDS="$4"

mc_setup_python_env
mc_banner "Curve Evaluation"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "CURVE_NAME: ${CURVE_NAME}"
echo "NUM_BENDS: ${NUM_BENDS}"
echo ""

mc_eval_curve_checkpoint "${CHECKPOINT_PATH}" "${OUTPUT_DIR}" "${CURVE_NAME}" "${NUM_BENDS}"
