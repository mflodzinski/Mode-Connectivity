#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_xor_permutation_scale_experiment_%j.out
#SBATCH --error=slurm_xor_permutation_scale_experiment_%j.err
#SBATCH --job-name=xor_perm_sc

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

OUTPUT_DIR="${OUTPUT_DIR:-results/xor/xor_5h_perm_vs_scale}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-results/xor/xor_5h_trained_linear_pairs/checkpoints}"
HIDDEN_SIZE="${HIDDEN_SIZE:-}"
SEEDS="${SEEDS:-}"
PAIRS="${PAIRS:-}"
CURVE_EVAL_POINTS="${CURVE_EVAL_POINTS:-61}"
SINKHORN_SEARCH_CONFIG="${SINKHORN_SEARCH_CONFIG:-configs/analysis/xor_sinkhorn_permutation_search.yaml}"
SINKHORN_OPT_STEPS="${SINKHORN_OPT_STEPS:-1000}"
SINKHORN_OPT_LR="${SINKHORN_OPT_LR:-0.05}"
SINKHORN_OPT_T_POINTS="${SINKHORN_OPT_T_POINTS:-31}"
SINKHORN_TAU="${SINKHORN_TAU:-1.0}"
SINKHORN_ITERS="${SINKHORN_ITERS:-20}"
SINKHORN_IDENTITY_STRENGTH="${SINKHORN_IDENTITY_STRENGTH:-1.0}"
SINKHORN_PATIENCE="${SINKHORN_PATIENCE:-150}"
SINKHORN_MIN_DELTA="${SINKHORN_MIN_DELTA:-1e-6}"
SINKHORN_SEARCH_STEPS="${SINKHORN_SEARCH_STEPS:-}"
SINKHORN_SEARCH_LRS="${SINKHORN_SEARCH_LRS:-}"
SINKHORN_SEARCH_TAUS="${SINKHORN_SEARCH_TAUS:-}"
SINKHORN_SEARCH_IDENTITY_STRENGTHS="${SINKHORN_SEARCH_IDENTITY_STRENGTHS:-}"
SINKHORN_SEARCH_PATIENCE="${SINKHORN_SEARCH_PATIENCE:-}"
SINKHORN_SEARCH_MIN_DELTA="${SINKHORN_SEARCH_MIN_DELTA:-}"
PERM_SCALE_SEARCH_CONFIG="${PERM_SCALE_SEARCH_CONFIG:-configs/analysis/xor_permutation_scale_search.yaml}"
SINKHORN_PERM_SCALE_SEARCH_CONFIG="${SINKHORN_PERM_SCALE_SEARCH_CONFIG:-configs/analysis/xor_sinkhorn_permutation_scale_search.yaml}"
SCALE_OPT_STEPS="${SCALE_OPT_STEPS:-1000}"
SCALE_OPT_LR="${SCALE_OPT_LR:-0.05}"
SCALE_OPT_T_POINTS="${SCALE_OPT_T_POINTS:-31}"
SCALE_REG="${SCALE_REG:-0.0}"
SCALE_PATIENCE="${SCALE_PATIENCE:-150}"
SCALE_MIN_DELTA="${SCALE_MIN_DELTA:-1e-6}"
PERM_SCALE_TARGET_EPSILON="${PERM_SCALE_TARGET_EPSILON:-}"
PERM_SCALE_SEARCH_STEPS="${PERM_SCALE_SEARCH_STEPS:-}"
PERM_SCALE_SEARCH_LRS="${PERM_SCALE_SEARCH_LRS:-}"
PERM_SCALE_SEARCH_REGS="${PERM_SCALE_SEARCH_REGS:-}"
PERM_SCALE_SEARCH_PATIENCE="${PERM_SCALE_SEARCH_PATIENCE:-}"
PERM_SCALE_SEARCH_MIN_DELTA="${PERM_SCALE_SEARCH_MIN_DELTA:-}"
SINKHORN_PERM_SCALE_TARGET_EPSILON="${SINKHORN_PERM_SCALE_TARGET_EPSILON:-}"
SINKHORN_PERM_SCALE_SEARCH_STEPS="${SINKHORN_PERM_SCALE_SEARCH_STEPS:-}"
SINKHORN_PERM_SCALE_SEARCH_LRS="${SINKHORN_PERM_SCALE_SEARCH_LRS:-}"
SINKHORN_PERM_SCALE_SEARCH_TAUS="${SINKHORN_PERM_SCALE_SEARCH_TAUS:-}"
SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS="${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS:-}"
SINKHORN_PERM_SCALE_SEARCH_REGS="${SINKHORN_PERM_SCALE_SEARCH_REGS:-}"
SINKHORN_PERM_SCALE_SEARCH_PATIENCE="${SINKHORN_PERM_SCALE_SEARCH_PATIENCE:-}"
SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA="${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA:-}"
RUN_JOINT_PERM_SCALE="${RUN_JOINT_PERM_SCALE:-false}"
VERBOSE="${VERBOSE:-false}"

echo "========================================"
echo "XOR Permutation vs Scale Experiment"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}"
echo "HIDDEN_SIZE: ${HIDDEN_SIZE:-<from checkpoints>}"
echo "SEEDS: ${SEEDS:-<all available>}"
echo "PAIRS: ${PAIRS:-<all>}"
echo "CURVE_EVAL_POINTS: ${CURVE_EVAL_POINTS}"
echo "SINKHORN_SEARCH_CONFIG: ${SINKHORN_SEARCH_CONFIG:-<disabled>}"
echo "SINKHORN_OPT_STEPS: ${SINKHORN_OPT_STEPS}"
echo "SINKHORN_OPT_LR: ${SINKHORN_OPT_LR}"
echo "SINKHORN_OPT_T_POINTS: ${SINKHORN_OPT_T_POINTS}"
echo "SINKHORN_TAU: ${SINKHORN_TAU}"
echo "SINKHORN_ITERS: ${SINKHORN_ITERS}"
echo "SINKHORN_IDENTITY_STRENGTH: ${SINKHORN_IDENTITY_STRENGTH}"
echo "SINKHORN_PATIENCE: ${SINKHORN_PATIENCE}"
echo "SINKHORN_MIN_DELTA: ${SINKHORN_MIN_DELTA}"
echo "SINKHORN_SEARCH_STEPS: ${SINKHORN_SEARCH_STEPS:-<disabled>}"
echo "SINKHORN_SEARCH_LRS: ${SINKHORN_SEARCH_LRS:-<disabled>}"
echo "SINKHORN_SEARCH_TAUS: ${SINKHORN_SEARCH_TAUS:-<disabled>}"
echo "SINKHORN_SEARCH_IDENTITY_STRENGTHS: ${SINKHORN_SEARCH_IDENTITY_STRENGTHS:-<disabled>}"
echo "SINKHORN_TARGET_EPSILON: <best exact permutation barrier per pair>"
echo "SINKHORN_SEARCH_PATIENCE: ${SINKHORN_SEARCH_PATIENCE:-<from yaml/default>}"
echo "SINKHORN_SEARCH_MIN_DELTA: ${SINKHORN_SEARCH_MIN_DELTA:-<from yaml/default>}"
echo "PERM_SCALE_SEARCH_CONFIG: ${PERM_SCALE_SEARCH_CONFIG:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_CONFIG: ${SINKHORN_PERM_SCALE_SEARCH_CONFIG:-<disabled>}"
echo "SCALE_OPT_STEPS: ${SCALE_OPT_STEPS}"
echo "SCALE_OPT_LR: ${SCALE_OPT_LR}"
echo "SCALE_OPT_T_POINTS: ${SCALE_OPT_T_POINTS}"
echo "SCALE_REG: ${SCALE_REG}"
echo "SCALE_PATIENCE: ${SCALE_PATIENCE}"
echo "SCALE_MIN_DELTA: ${SCALE_MIN_DELTA}"
echo "PERM_SCALE_TARGET_EPSILON: ${PERM_SCALE_TARGET_EPSILON:-<disabled>}"
echo "PERM_SCALE_SEARCH_STEPS: ${PERM_SCALE_SEARCH_STEPS:-<disabled>}"
echo "PERM_SCALE_SEARCH_LRS: ${PERM_SCALE_SEARCH_LRS:-<disabled>}"
echo "PERM_SCALE_SEARCH_REGS: ${PERM_SCALE_SEARCH_REGS:-<disabled>}"
echo "PERM_SCALE_SEARCH_PATIENCE: ${PERM_SCALE_SEARCH_PATIENCE:-<from yaml/default>}"
echo "PERM_SCALE_SEARCH_MIN_DELTA: ${PERM_SCALE_SEARCH_MIN_DELTA:-<from yaml/default>}"
echo "SINKHORN_PERM_SCALE_TARGET_EPSILON: ${SINKHORN_PERM_SCALE_TARGET_EPSILON:-<from yaml/default>}"
echo "SINKHORN_PERM_SCALE_SEARCH_STEPS: ${SINKHORN_PERM_SCALE_SEARCH_STEPS:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_LRS: ${SINKHORN_PERM_SCALE_SEARCH_LRS:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_TAUS: ${SINKHORN_PERM_SCALE_SEARCH_TAUS:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS: ${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_REGS: ${SINKHORN_PERM_SCALE_SEARCH_REGS:-<disabled>}"
echo "SINKHORN_PERM_SCALE_SEARCH_PATIENCE: ${SINKHORN_PERM_SCALE_SEARCH_PATIENCE:-<from yaml/default>}"
echo "SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA: ${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA:-<from yaml/default>}"
echo "RUN_JOINT_PERM_SCALE: ${RUN_JOINT_PERM_SCALE}"
echo "VERBOSE: ${VERBOSE}"
echo "PYTHON: $(command -v python || echo missing)"
echo ""

args=(
    --output "${OUTPUT_DIR}"
    --checkpoints-dir "${CHECKPOINTS_DIR}"
    --curve-eval-points "${CURVE_EVAL_POINTS}"
    --sinkhorn-search-config "${SINKHORN_SEARCH_CONFIG}"
    --sinkhorn-opt-steps "${SINKHORN_OPT_STEPS}"
    --sinkhorn-opt-lr "${SINKHORN_OPT_LR}"
    --sinkhorn-opt-t-points "${SINKHORN_OPT_T_POINTS}"
    --sinkhorn-tau "${SINKHORN_TAU}"
    --sinkhorn-iters "${SINKHORN_ITERS}"
    --sinkhorn-identity-strength "${SINKHORN_IDENTITY_STRENGTH}"
    --sinkhorn-patience "${SINKHORN_PATIENCE}"
    --sinkhorn-min-delta "${SINKHORN_MIN_DELTA}"
    --perm-scale-search-config "${PERM_SCALE_SEARCH_CONFIG}"
    --sinkhorn-perm-scale-search-config "${SINKHORN_PERM_SCALE_SEARCH_CONFIG}"
    --scale-opt-steps "${SCALE_OPT_STEPS}"
    --scale-opt-lr "${SCALE_OPT_LR}"
    --scale-opt-t-points "${SCALE_OPT_T_POINTS}"
    --scale-reg "${SCALE_REG}"
    --scale-patience "${SCALE_PATIENCE}"
    --scale-min-delta "${SCALE_MIN_DELTA}"
)

if [ -n "${HIDDEN_SIZE}" ]; then
    args+=(--hidden-size "${HIDDEN_SIZE}")
fi
if [ -n "${SEEDS}" ]; then
    args+=(--seeds "${SEEDS}")
fi
if [ -n "${PAIRS}" ]; then
    args+=(--pairs "${PAIRS}")
fi
if [ -n "${SINKHORN_SEARCH_STEPS}" ]; then
    args+=(--sinkhorn-search-steps "${SINKHORN_SEARCH_STEPS}")
fi
if [ -n "${SINKHORN_SEARCH_LRS}" ]; then
    args+=(--sinkhorn-search-lrs "${SINKHORN_SEARCH_LRS}")
fi
if [ -n "${SINKHORN_SEARCH_TAUS}" ]; then
    args+=(--sinkhorn-search-taus "${SINKHORN_SEARCH_TAUS}")
fi
if [ -n "${SINKHORN_SEARCH_IDENTITY_STRENGTHS}" ]; then
    args+=(--sinkhorn-search-identity-strengths "${SINKHORN_SEARCH_IDENTITY_STRENGTHS}")
fi
if [ -n "${SINKHORN_SEARCH_PATIENCE}" ]; then
    args+=(--sinkhorn-search-patience "${SINKHORN_SEARCH_PATIENCE}")
fi
if [ -n "${SINKHORN_SEARCH_MIN_DELTA}" ]; then
    args+=(--sinkhorn-search-min-delta "${SINKHORN_SEARCH_MIN_DELTA}")
fi
if [ -n "${PERM_SCALE_TARGET_EPSILON}" ]; then
    args+=(--perm-scale-target-epsilon "${PERM_SCALE_TARGET_EPSILON}")
fi
if [ -n "${PERM_SCALE_SEARCH_STEPS}" ]; then
    args+=(--perm-scale-search-steps "${PERM_SCALE_SEARCH_STEPS}")
fi
if [ -n "${PERM_SCALE_SEARCH_LRS}" ]; then
    args+=(--perm-scale-search-lrs "${PERM_SCALE_SEARCH_LRS}")
fi
if [ -n "${PERM_SCALE_SEARCH_REGS}" ]; then
    args+=(--perm-scale-search-regs "${PERM_SCALE_SEARCH_REGS}")
fi
if [ -n "${PERM_SCALE_SEARCH_PATIENCE}" ]; then
    args+=(--perm-scale-search-patience "${PERM_SCALE_SEARCH_PATIENCE}")
fi
if [ -n "${PERM_SCALE_SEARCH_MIN_DELTA}" ]; then
    args+=(--perm-scale-search-min-delta "${PERM_SCALE_SEARCH_MIN_DELTA}")
fi
if [ -n "${SINKHORN_PERM_SCALE_TARGET_EPSILON}" ]; then
    args+=(--sinkhorn-perm-scale-target-epsilon "${SINKHORN_PERM_SCALE_TARGET_EPSILON}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_STEPS}" ]; then
    args+=(--sinkhorn-perm-scale-search-steps "${SINKHORN_PERM_SCALE_SEARCH_STEPS}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_LRS}" ]; then
    args+=(--sinkhorn-perm-scale-search-lrs "${SINKHORN_PERM_SCALE_SEARCH_LRS}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_TAUS}" ]; then
    args+=(--sinkhorn-perm-scale-search-taus "${SINKHORN_PERM_SCALE_SEARCH_TAUS}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS}" ]; then
    args+=(--sinkhorn-perm-scale-search-identity-strengths "${SINKHORN_PERM_SCALE_SEARCH_IDENTITY_STRENGTHS}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_REGS}" ]; then
    args+=(--sinkhorn-perm-scale-search-regs "${SINKHORN_PERM_SCALE_SEARCH_REGS}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_PATIENCE}" ]; then
    args+=(--sinkhorn-perm-scale-search-patience "${SINKHORN_PERM_SCALE_SEARCH_PATIENCE}")
fi
if [ -n "${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA}" ]; then
    args+=(--sinkhorn-perm-scale-search-min-delta "${SINKHORN_PERM_SCALE_SEARCH_MIN_DELTA}")
fi
if [ "${RUN_JOINT_PERM_SCALE}" = "true" ]; then
    args+=(--run-joint-perm-scale)
fi
if [ "${VERBOSE}" = "true" ]; then
    args+=(--verbose)
fi

srun python scripts/experiments/xor_permutation_scale_experiment.py "${args[@]}"

echo ""
echo "========================================"
echo "XOR PERMUTATION VS SCALE COMPLETE"
echo "========================================"
echo "Results written under: ${OUTPUT_DIR}"
