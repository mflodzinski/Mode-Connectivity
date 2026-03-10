#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_sweep_all_%j.out
#SBATCH --error=slurm_external_sinkhorn_sweep_all_%j.err
#SBATCH --job-name=ext_sinkhorn_all
#SBATCH --gres=gpu:a40:1

# Run the full configured external Sinkhorn sweep sequentially in one Slurm job.
# The sweep grid is defined in:
#   configs/analysis/external_sinkhorn_rebasin_vgg16_sweep.yaml

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

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-23}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

echo "========================================"
echo "External Sinkhorn Full Sweep"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "START_INDEX: ${START_INDEX}"
echo "END_INDEX: ${END_INDEX}"
echo "CONTINUE_ON_ERROR: ${CONTINUE_ON_ERROR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo ""

srun python scripts/analysis/run_external_sinkhorn_baseline_sweep_all.py \
    start_index="${START_INDEX}" \
    end_index="${END_INDEX}" \
    continue_on_error="${CONTINUE_ON_ERROR}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "EXTERNAL SINKHORN FULL SWEEP COMPLETE"
echo "========================================"
echo "Results are written under the sweep base output root configured in:"
echo "  configs/analysis/external_sinkhorn_rebasin_vgg16_sweep.yaml"
