#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_example_main_lmc_cnn_%j.out
#SBATCH --error=slurm_external_sinkhorn_example_main_lmc_cnn_%j.err
#SBATCH --job-name=ext_sh_ex_lmc
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
SINKHORN_ROOT="${PROJECT_ROOT}/external/sinkhorn-rebasin"
EXAMPLE_ROOT="${PROJECT_ROOT}/external/sinkhorn-rebasin/examples"
cd "${EXAMPLE_ROOT}"

export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
export EXTRA_PYTHONPATH="${PROJECT_ROOT}/.cluster-pydeps"
export PYTHONPATH="${EXTRA_PYTHONPATH}:${SINKHORN_ROOT}:${EXAMPLE_ROOT}:${PYTHONPATH:-}"

if [ ! -f "${EXAMPLE_ROOT}/main_lmc_cnn.py" ]; then
    echo "Missing external/sinkhorn-rebasin/examples/main_lmc_cnn.py in this checkout."
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load graphviz >/dev/null 2>&1 || true
fi

echo "========================================"
echo "Original Sinkhorn Example main_lmc_cnn.py"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "SINKHORN_ROOT: ${SINKHORN_ROOT}"
echo "EXAMPLE_ROOT: ${EXAMPLE_ROOT}"
echo "PWD: $(pwd)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python main_lmc_cnn.py

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN EXAMPLE main_lmc_cnn.py COMPLETE"
echo "========================================"
echo "Outputs are written relative to: ${EXAMPLE_ROOT}"
