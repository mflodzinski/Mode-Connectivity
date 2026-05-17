#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_xor_linear_%j.out
#SBATCH --error=slurm_xor_linear_%j.err
#SBATCH --job-name=xor_linear

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_banner "XOR Train Linear Barriers"

args=(
  --output "${OUTPUT_DIR:-results/xor/xor_5h_trained_linear_pairs}"
  --hidden-size "${HIDDEN_SIZE:-5}"
  --num-networks "${NUM_NETWORKS:-10}"
  --max-endpoint-loss "${MAX_ENDPOINT_LOSS:-0.02}"
  --curve-eval-points "${CURVE_EVAL_POINTS:-61}"
)
if [ -n "${SEEDS:-}" ]; then args+=(--seeds "${SEEDS}"); fi
if [ -n "${PAIRS:-}" ]; then args+=(--pairs "${PAIRS}"); fi
if [ -n "${TRAIN_MAX_EPOCHS:-}" ]; then args+=(--train-max-epochs "${TRAIN_MAX_EPOCHS}"); fi
if [ -n "${TRAIN_LR:-}" ]; then args+=(--train-lr "${TRAIN_LR}"); fi
if [ "${VERBOSE:-false}" = "true" ]; then args+=(--verbose); fi

mc_run_module experiments.xor.train_linear_barriers "${args[@]}"
