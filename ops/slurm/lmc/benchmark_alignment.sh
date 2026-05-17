#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_benchmark_%x_%j.out
#SBATCH --error=slurm_lmc_benchmark_%x_%j.err
#SBATCH --job-name=lmc_benchmark
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch $0 <w0-checkpoint> <w1-checkpoint> <output-json> [extra benchmark args...]"
  exit 1
fi

W0_CHECKPOINT="$1"
W1_CHECKPOINT="$2"
OUTPUT_JSON="$3"
shift 3

mc_setup_python_env
mc_banner "LMC Alignment Benchmark"
echo "W0_CHECKPOINT: ${W0_CHECKPOINT}"
echo "W1_CHECKPOINT: ${W1_CHECKPOINT}"
echo "OUTPUT_JSON: ${OUTPUT_JSON}"
echo ""

mc_run_module experiments.lmc.benchmark_alignment \
  --w0 "${W0_CHECKPOINT}" \
  --w1 "${W1_CHECKPOINT}" \
  --perm-seed "${PERM_SEED:-42}" \
  --wm-seed "${WM_SEED:-0}" \
  --method "${METHOD:-weight_matching}" \
  --max-iter "${MAX_ITER:-100}" \
  --num-eval-points "${NUM_EVAL_POINTS:-61}" \
  --data-path "${DATA_PATH:-./data}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --output "${OUTPUT_JSON}" \
  "$@"
