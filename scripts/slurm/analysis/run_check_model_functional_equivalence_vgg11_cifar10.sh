#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_check_model_functional_equivalence_vgg11_cifar10_%j.out
#SBATCH --error=slurm_check_model_functional_equivalence_vgg11_cifar10_%j.err
#SBATCH --job-name=chk_v11_eq
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

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-results/vgg11/cifar10/original_sinkhorn_lmc/model_a.pt}"
REBASED_CHECKPOINT="${REBASED_CHECKPOINT:-results/vgg11/cifar10/raw_pth_align_sweep_scale/steps150_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p005/rebased_model.pt}"
DATA_PATH="${DATA_PATH:-./data}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_JSON="${OUTPUT_JSON:-results/vgg11/cifar10/raw_pth_align_sweep_scale/steps150_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p005/functional_equivalence.json}"

echo "========================================"
echo "Check VGG11 CIFAR10 Functional Equivalence"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "REBASED_CHECKPOINT: ${REBASED_CHECKPOINT}"
echo "DATA_PATH: ${DATA_PATH}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "OUTPUT_JSON: ${OUTPUT_JSON}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/check_model_functional_equivalence_vgg11_cifar10.py \
    --model-a-checkpoint "${MODEL_A_CHECKPOINT}" \
    --rebased-checkpoint "${REBASED_CHECKPOINT}" \
    --data-path "${DATA_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --device cuda \
    --output-json "${OUTPUT_JSON}"

echo ""
echo "========================================"
echo "FUNCTIONAL EQUIVALENCE CHECK COMPLETE"
echo "========================================"
echo "Results written to: ${OUTPUT_JSON}"
