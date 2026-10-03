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

OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_local/xor/train_linear}"

echo "========================================"
echo "Local XOR Smoke: Train Linear Barriers"
echo "========================================"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

python -m experiments.xor.train_linear_barriers \
  --output "${OUTPUT_DIR}" \
  --hidden-size 5 \
  --num-networks 6 \
  --seeds 0,1,2,3,4,5 \
  --max-endpoint-loss 0.10 \
  --train-max-epochs 2000 \
  --train-lr 0.03 \
  --curve-eval-points 31
