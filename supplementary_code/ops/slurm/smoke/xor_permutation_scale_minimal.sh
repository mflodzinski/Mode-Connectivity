#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_xor_perm_scale_%j.out
#SBATCH --error=slurm_smoke_xor_perm_scale_%j.err
#SBATCH --job-name=smoke_xor_ps

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Smoke Check: XOR Permutation vs Scale"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-results/smoke/xor/train_linear/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke/xor/permutation_scale}"

echo "CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

mc_run_module experiments.xor.permutation_scale \
  --output "${OUTPUT_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --curve-eval-points 31 \
  --sinkhorn-opt-steps 50 \
  --sinkhorn-opt-lr 0.01 \
  --sinkhorn-opt-t-points 21 \
  --sinkhorn-iters 10 \
  --sinkhorn-search-steps 50 \
  --sinkhorn-search-lrs 0.01 \
  --sinkhorn-search-taus 1.0 \
  --sinkhorn-search-identity-strengths 1.0 \
  --sinkhorn-search-patience 20 \
  --perm-scale-search-steps 50 \
  --perm-scale-search-lrs 0.01 \
  --perm-scale-search-regs 0.0 \
  --perm-scale-search-patience 20 \
  --sinkhorn-perm-scale-search-steps 50 \
  --sinkhorn-perm-scale-search-lrs 0.01 \
  --sinkhorn-perm-scale-search-taus 1.0 \
  --sinkhorn-perm-scale-search-identity-strengths 1.0 \
  --sinkhorn-perm-scale-search-regs 0.0 \
  --sinkhorn-perm-scale-search-patience 20
