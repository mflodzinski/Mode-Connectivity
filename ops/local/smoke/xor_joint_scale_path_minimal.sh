#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-results/xor/xor_2h_15seeds/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_local/xor/joint_scale_path}"

python -m experiments.xor.joint_scale_path \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --output "$OUTPUT_DIR" \
  --seeds 2,4,9 \
  --pairs 2-4 \
  --steps 80 \
  --num-t-samples 7 \
  --eval-points 31 \
  --restarts 2 \
  --restart-std 0.02 \
  --scale-penalty 0.0001 \
  --positive-control-pair 2-9 \
  --positive-control-log-scale 3.0
