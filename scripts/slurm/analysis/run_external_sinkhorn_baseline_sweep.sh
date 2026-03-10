#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_sweep_%A_%a.out
#SBATCH --error=slurm_external_sinkhorn_sweep_%A_%a.err
#SBATCH --job-name=ext_sinkhorn_sw
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-23

# Hyperparameter sweep for the vendored external/sinkhorn-rebasin baseline.
# The sweep values are defined in
#   configs/analysis/external_sinkhorn_rebasin_vgg16_sweep.yaml
# This array size (24 runs) matches the current config.

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

if [ ! -f "${PROJECT_ROOT}/external/sinkhorn-rebasin/examples/models/vgg.py" ]; then
    echo "Missing external/sinkhorn-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/sinkhorn-rebasin"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load graphviz >/dev/null 2>&1 || true
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "========================================"
echo "External Sinkhorn-Rebasin Sweep"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "TASK_ID: ${TASK_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo ""

srun python scripts/analysis/run_external_sinkhorn_baseline_sweep.py \
    sweep_task_id="${TASK_ID}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "EXTERNAL SINKHORN SWEEP RUN COMPLETE"
echo "========================================"
echo "Results are written under the sweep base output root configured in:"
echo "  configs/analysis/external_sinkhorn_rebasin_vgg16_sweep.yaml"
