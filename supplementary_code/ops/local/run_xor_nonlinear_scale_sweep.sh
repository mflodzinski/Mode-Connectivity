#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

hidden_size="${HIDDEN_SIZE:-3}"
curve_type="${CURVE_TYPE:-polygonal}"
scale_endpoints="${SCALE_ENDPOINTS:-second}"
internal_parameterization="${INTERNAL_PARAMETERIZATION:-affine_residual}"
if [ "${hidden_size}" = "2" ]; then
  default_checkpoints_dir="results/xor/xor_2h_15seeds/checkpoints"
else
  default_checkpoints_dir="results/xor/xor_${hidden_size}h_trained_linear_pairs/checkpoints"
fi
checkpoints_dir="${CHECKPOINTS_DIR:-${default_checkpoints_dir}}"
experiment_tag="${EXPERIMENT_TAG:-sweep}"
output_dir="${OUTPUT_DIR:-results/xor/xor_${hidden_size}h_joint_scale_${curve_type}_${experiment_tag}}"

args=(
  --checkpoints-dir "${checkpoints_dir}"
  --hidden-size "${hidden_size}"
  --no-permutation
  --output "${output_dir}"
  --curve-type "${curve_type}"
  --scale-endpoints "${scale_endpoints}"
  --internal-parameterization "${internal_parameterization}"
  --internal-points "${INTERNAL_POINTS:-1,2,4,6}"
  --steps "${STEPS:-1500}"
  --lr "${LR:-0.05}"
  --num-t-samples "${NUM_T_SAMPLES:-31}"
  --stochastic-max-samples "${STOCHASTIC_MAX_SAMPLES:-8}"
  --grid-refresh-every "${GRID_REFRESH_EVERY:-25}"
  --grid-refresh-points "${GRID_REFRESH_POINTS:-31}"
  --eval-points "${EVAL_POINTS:-501}"
  --restarts "${RESTARTS:-1}"
  --restart-std "${RESTART_STD:-0.05}"
  --scale-penalty "${SCALE_PENALTY:-0.0001}"
  --skip-positive-control
  --skip-plots
  --compact-restarts
)
if [ -n "${SEEDS:-}" ]; then args+=(--seeds "${SEEDS}"); fi
if [ -n "${PAIRS:-}" ]; then args+=(--pairs "${PAIRS}"); fi
if [ "${VERBOSE:-false}" = "true" ]; then args+=(--verbose); fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
env PYTHONPATH="src:.${PYTHONPATH:+:${PYTHONPATH}}" \
  .venv/bin/python -m experiments.xor.joint_scale_polygonal "${args[@]}"
