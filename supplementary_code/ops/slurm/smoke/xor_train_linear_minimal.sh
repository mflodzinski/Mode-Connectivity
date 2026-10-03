#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_xor_linear_%j.out
#SBATCH --error=slurm_smoke_xor_linear_%j.err
#SBATCH --job-name=smoke_xor_ln

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Smoke Check: XOR Train Linear Barriers"

OUTPUT_DIR="${OUTPUT_DIR:-results/smoke/xor/train_linear}"

echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

mc_run_module experiments.xor.train_linear_barriers \
  --output "${OUTPUT_DIR}" \
  --hidden-size 5 \
  --num-networks 6 \
  --seeds 0,1,2,3,4,5 \
  --max-endpoint-loss 0.10 \
  --train-max-epochs 2000 \
  --train-lr 0.03 \
  --curve-eval-points 31
