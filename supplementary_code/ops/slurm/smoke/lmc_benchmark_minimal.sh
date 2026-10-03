#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_lmc_bench_%j.out
#SBATCH --error=slurm_smoke_lmc_bench_%j.err
#SBATCH --job-name=smoke_lmc_bm
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Smoke Check: LMC Alignment Benchmark"

PAIR_ROOT="${PAIR_ROOT:-results/smoke/lmc/from_scratch_split0_1epoch}"
FINAL_EPOCH="${FINAL_EPOCH:-1}"
W0_CHECKPOINT="${W0_CHECKPOINT:-${PAIR_ROOT}/seed0/checkpoint-${FINAL_EPOCH}.pt}"
W1_CHECKPOINT="${W1_CHECKPOINT:-${PAIR_ROOT}/seed1/checkpoint-${FINAL_EPOCH}.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${PAIR_ROOT}/benchmark_alignment/results.json}"

echo "W0_CHECKPOINT: ${W0_CHECKPOINT}"
echo "W1_CHECKPOINT: ${W1_CHECKPOINT}"
echo "OUTPUT_JSON: ${OUTPUT_JSON}"
echo ""

mc_run_module experiments.lmc.benchmark_alignment \
  --w0 "${W0_CHECKPOINT}" \
  --w1 "${W1_CHECKPOINT}" \
  --perm-seed 42 \
  --wm-seed 0 \
  --method weight_matching \
  --max-iter 20 \
  --num-eval-points 11 \
  --data-path ./data \
  --batch-size 128 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output "${OUTPUT_JSON}"
