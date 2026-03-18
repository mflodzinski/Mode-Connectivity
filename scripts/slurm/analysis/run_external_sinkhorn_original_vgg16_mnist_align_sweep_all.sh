#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_vgg16_mnist_align_sweep_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_vgg16_mnist_align_sweep_%j.err
#SBATCH --job-name=ext_sh_v16_sw
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"
export EXTRA_PYTHONPATH="${PROJECT_ROOT}/.cluster-pydeps"
export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH}"

if [ ! -f "${PROJECT_ROOT}/external/sinkhorn-rebasin/examples/models/vgg.py" ]; then
    echo "Missing external/sinkhorn-rebasin in this checkout."
    echo "Initialize the submodule first, for example:"
    echo "  git submodule update --init --recursive external/sinkhorn-rebasin"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load graphviz >/dev/null 2>&1 || true
fi

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg16/mnist/original_sinkhorn_lmc/model_a.pt}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-results/vgg16/mnist/original_sinkhorn_lmc/model_b.pt}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-results/vgg16/mnist/original_sinkhorn_lmc_align_sweep}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-null}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

echo "========================================"
echo "Original Sinkhorn VGG16 MNIST Align Sweep"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "BASE_OUTPUT_ROOT: ${BASE_OUTPUT_ROOT}"
echo "START_INDEX: ${START_INDEX}"
echo "END_INDEX: ${END_INDEX}"
echo "CONTINUE_ON_ERROR: ${CONTINUE_ON_ERROR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/run_external_sinkhorn_original_vgg16_mnist_align_sweep_all.py \
    model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
    model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
    base_output_root="${BASE_OUTPUT_ROOT}" \
    start_index="${START_INDEX}" \
    end_index="${END_INDEX}" \
    continue_on_error="${CONTINUE_ON_ERROR}" \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    device=cuda

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN VGG16 MNIST ALIGN SWEEP COMPLETE"
echo "========================================"
echo "Results written under: ${BASE_OUTPUT_ROOT}"
