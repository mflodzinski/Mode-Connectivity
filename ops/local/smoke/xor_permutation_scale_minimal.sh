#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.mplcache}"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-results/smoke_local/xor/train_linear/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_local/xor/permutation_scale}"

echo "========================================"
echo "Local XOR Smoke: Permutation vs Scale"
echo "========================================"
echo "CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

python -m experiments.xor.permutation_scale \
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
