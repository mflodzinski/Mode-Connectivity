#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-results/xor/xor_2h_15seeds/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_local/xor/joint_scale_polygonal}"

python -m experiments.xor.joint_scale_polygonal \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --output "$OUTPUT_DIR" \
  --seeds 2,4,9,10 \
  --pairs 4-10 \
  --polygon-bends 1,2,4 \
  --steps 60 \
  --num-t-samples 11 \
  --eval-points 121 \
  --restarts 2 \
  --positive-control-pair 2-9
