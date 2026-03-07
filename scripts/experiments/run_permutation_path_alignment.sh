#!/bin/bash
set -euo pipefail

ENDPOINT_A="${1:-${ENDPOINT_A:-results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt}}"
ENDPOINT_B="${2:-${ENDPOINT_B:-results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt}}"
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-results/vgg16/cifar10/alignment/permutation_path}}"
EXPERIMENT_NAME="${4:-${EXPERIMENT_NAME:-seed0-seed1_polychain_path_alignment}}"

if [ "$#" -gt 4 ]; then
  EXTRA_ARGS=("${@:5}")
else
  EXTRA_ARGS=()
fi

mkdir -p "${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

echo "========================================"
echo "Permutation Path Alignment"
echo "========================================"
echo "endpoint_a: ${ENDPOINT_A}"
echo "endpoint_b: ${ENDPOINT_B}"
echo "output_root: ${OUTPUT_ROOT}"
echo "experiment_name: ${EXPERIMENT_NAME}"
echo ""

CMD=(
  python scripts/analysis/run_permutation_path_alignment.py
  endpoint_a="${ENDPOINT_A}"
  endpoint_b="${ENDPOINT_B}"
  output_root="${OUTPUT_ROOT}"
  experiment_name="${EXPERIMENT_NAME}"
  "${EXTRA_ARGS[@]}"
)
"${CMD[@]}"

echo ""
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: ${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
