#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_xor_joint_scale_path_%j.out
#SBATCH --error=slurm_xor_joint_scale_path_%j.err
#SBATCH --job-name=xor_joint_scale_path

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "XOR Nonlinear Path vs Path + Scale"

hidden_size="${HIDDEN_SIZE:-7}"
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
  --curve-type "${curve_type}"
  --scale-endpoints "${scale_endpoints}"
  --internal-parameterization "${internal_parameterization}"
  --output "${output_dir}"
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
  --compact-restarts
)
if [ -n "${SEEDS:-}" ]; then args+=(--seeds "${SEEDS}"); fi
if [ -n "${PAIRS:-}" ]; then args+=(--pairs "${PAIRS}"); fi
if [ "${VERBOSE:-false}" = "true" ]; then args+=(--verbose); fi

mc_run_module experiments.xor.joint_scale_polygonal "${args[@]}"
