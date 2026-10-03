#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_xor_perm_scale_%j.out
#SBATCH --error=slurm_xor_perm_scale_%j.err
#SBATCH --job-name=xor_perm_scale

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "XOR Permutation vs Scale"

args=(
  --output "${OUTPUT_DIR:-results/xor/xor_5h_perm_vs_scale}"
  --checkpoints-dir "${CHECKPOINTS_DIR:-results/xor/xor_5h_trained_linear_pairs/checkpoints}"
  --curve-eval-points "${CURVE_EVAL_POINTS:-61}"
  --sinkhorn-search-config "${SINKHORN_SEARCH_CONFIG:-configs/experiments/xor/search/sinkhorn_permutation.yaml}"
  --sinkhorn-opt-steps "${SINKHORN_OPT_STEPS:-1000}"
  --sinkhorn-opt-lr "${SINKHORN_OPT_LR:-0.05}"
  --sinkhorn-opt-t-points "${SINKHORN_OPT_T_POINTS:-31}"
  --sinkhorn-tau "${SINKHORN_TAU:-1.0}"
  --sinkhorn-iters "${SINKHORN_ITERS:-20}"
  --sinkhorn-identity-strength "${SINKHORN_IDENTITY_STRENGTH:-1.0}"
  --sinkhorn-patience "${SINKHORN_PATIENCE:-150}"
  --sinkhorn-min-delta "${SINKHORN_MIN_DELTA:-1e-6}"
  --perm-scale-search-config "${PERM_SCALE_SEARCH_CONFIG:-configs/experiments/xor/search/permutation_scale.yaml}"
  --sinkhorn-perm-scale-search-config "${SINKHORN_PERM_SCALE_SEARCH_CONFIG:-configs/experiments/xor/search/sinkhorn_permutation_scale.yaml}"
  --scale-opt-steps "${SCALE_OPT_STEPS:-1000}"
  --scale-opt-lr "${SCALE_OPT_LR:-0.05}"
  --scale-opt-t-points "${SCALE_OPT_T_POINTS:-31}"
  --scale-reg "${SCALE_REG:-0.0}"
  --scale-patience "${SCALE_PATIENCE:-150}"
  --scale-min-delta "${SCALE_MIN_DELTA:-1e-6}"
)
if [ -n "${HIDDEN_SIZE:-}" ]; then args+=(--hidden-size "${HIDDEN_SIZE}"); fi
if [ -n "${SEEDS:-}" ]; then args+=(--seeds "${SEEDS}"); fi
if [ -n "${PAIRS:-}" ]; then args+=(--pairs "${PAIRS}"); fi
if [ -n "${SINKHORN_SEARCH_STEPS:-}" ]; then args+=(--sinkhorn-search-steps "${SINKHORN_SEARCH_STEPS}"); fi
if [ -n "${SINKHORN_SEARCH_LRS:-}" ]; then args+=(--sinkhorn-search-lrs "${SINKHORN_SEARCH_LRS}"); fi
if [ -n "${SINKHORN_SEARCH_TAUS:-}" ]; then args+=(--sinkhorn-search-taus "${SINKHORN_SEARCH_TAUS}"); fi
if [ -n "${SINKHORN_SEARCH_IDENTITY_STRENGTHS:-}" ]; then args+=(--sinkhorn-search-identity-strengths "${SINKHORN_SEARCH_IDENTITY_STRENGTHS}"); fi
if [ -n "${SINKHORN_SEARCH_PATIENCE:-}" ]; then args+=(--sinkhorn-search-patience "${SINKHORN_SEARCH_PATIENCE}"); fi
if [ -n "${SINKHORN_SEARCH_MIN_DELTA:-}" ]; then args+=(--sinkhorn-search-min-delta "${SINKHORN_SEARCH_MIN_DELTA}"); fi
if [ -n "${PERM_SCALE_TARGET_EPSILON:-}" ]; then args+=(--perm-scale-target-epsilon "${PERM_SCALE_TARGET_EPSILON}"); fi
if [ -n "${PERM_SCALE_SEARCH_STEPS:-}" ]; then args+=(--perm-scale-search-steps "${PERM_SCALE_SEARCH_STEPS}"); fi
if [ -n "${PERM_SCALE_SEARCH_LRS:-}" ]; then args+=(--perm-scale-search-lrs "${PERM_SCALE_SEARCH_LRS}"); fi
if [ -n "${PERM_SCALE_SEARCH_REGS:-}" ]; then args+=(--perm-scale-search-regs "${PERM_SCALE_SEARCH_REGS}"); fi
if [ -n "${PERM_SCALE_SEARCH_PATIENCE:-}" ]; then args+=(--perm-scale-search-patience "${PERM_SCALE_SEARCH_PATIENCE}"); fi
if [ -n "${PERM_SCALE_SEARCH_MIN_DELTA:-}" ]; then args+=(--perm-scale-search-min-delta "${PERM_SCALE_SEARCH_MIN_DELTA}"); fi
if [ -n "${SINKHORN_PERM_SCALE_TARGET_EPSILON:-}" ]; then args+=(--sinkhorn-perm-scale-target-epsilon "${SINKHORN_PERM_SCALE_TARGET_EPSILON}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_STEPS:-}" ]; then args+=(--sinkhorn-perm-scale-search-steps "${SINKHORN_PERM_SCALE_SEARCH_STEPS}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_LRS:-}" ]; then args+=(--sinkhorn-perm-scale-search-lrs "${SINKHORN_PERM_SCALE_SEARCH_LRS}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_TAUS:-}" ]; then args+=(--sinkhorn-perm-scale-search-taus "${SINKHORN_PERM_SCALE_SEARCH_TAUS}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS:-}" ]; then args+=(--sinkhorn-perm-scale-search-identity-strengths "${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_REGS:-}" ]; then args+=(--sinkhorn-perm-scale-search-regs "${SINKHORN_PERM_SCALE_SEARCH_REGS}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_PATIENCE:-}" ]; then args+=(--sinkhorn-perm-scale-search-patience "${SINKHORN_PERM_SCALE_SEARCH_PATIENCE}"); fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA:-}" ]; then args+=(--sinkhorn-perm-scale-search-min-delta "${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA}"); fi
if [ "${RUN_JOINT_PERM_SCALE:-false}" = "true" ]; then args+=(--run-joint-perm-scale); fi
if [ "${VERBOSE:-false}" = "true" ]; then args+=(--verbose); fi

mc_run_module experiments.xor.permutation_scale "${args[@]}"
