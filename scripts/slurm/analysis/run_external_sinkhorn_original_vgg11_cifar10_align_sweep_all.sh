#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_orig_vgg11_cifar10_align_sweep_%j.out
#SBATCH --error=slurm_external_sinkhorn_orig_vgg11_cifar10_align_sweep_%j.err
#SBATCH --job-name=ext_sh_v11_sw
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

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-}"
SCALE_INVARIANT="${SCALE_INVARIANT:-}"
LAMBDA_SCALE="${LAMBDA_SCALE:-}"
FINETUNE_MODE="${FINETUNE_MODE:-}"
STARTING_ALIGNMENT_ARTIFACT="${STARTING_ALIGNMENT_ARTIFACT:-}"
STARTING_PERMUTATION_KIND="${STARTING_PERMUTATION_KIND:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-null}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

echo "========================================"
echo "Original Sinkhorn VGG11 CIFAR10 Align Sweep"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT:-<from config>}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT:-<from config>}"
echo "BASE_OUTPUT_ROOT: ${BASE_OUTPUT_ROOT:-<from config>}"
echo "SCALE_INVARIANT: ${SCALE_INVARIANT:-<from config>}"
echo "LAMBDA_SCALE: ${LAMBDA_SCALE:-<from config>}"
echo "FINETUNE_MODE: ${FINETUNE_MODE:-<from config>}"
echo "STARTING_ALIGNMENT_ARTIFACT: ${STARTING_ALIGNMENT_ARTIFACT:-<from config>}"
echo "STARTING_PERMUTATION_KIND: ${STARTING_PERMUTATION_KIND:-<from config>}"
echo "START_INDEX: ${START_INDEX}"
echo "END_INDEX: ${END_INDEX}"
echo "CONTINUE_ON_ERROR: ${CONTINUE_ON_ERROR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

args=(
    "start_index=${START_INDEX}"
    "end_index=${END_INDEX}"
    "continue_on_error=${CONTINUE_ON_ERROR}"
    "num_workers=${SLURM_CPUS_PER_TASK}"
    "device=cuda"
)
if [ -n "${MODEL_A_CHECKPOINT}" ]; then
    args+=("model_a_checkpoint=${MODEL_A_CHECKPOINT}")
fi
if [ -n "${MODEL_B_CHECKPOINT}" ]; then
    args+=("model_b_checkpoint=${MODEL_B_CHECKPOINT}")
fi
if [ -n "${BASE_OUTPUT_ROOT}" ]; then
    args+=("base_output_root=${BASE_OUTPUT_ROOT}")
fi
if [ -n "${SCALE_INVARIANT}" ]; then
    args+=("scale_invariant=${SCALE_INVARIANT}")
fi
if [ -n "${LAMBDA_SCALE}" ]; then
    args+=("lambda_scale=${LAMBDA_SCALE}")
fi
if [ -n "${FINETUNE_MODE}" ]; then
    args+=("finetune_mode=${FINETUNE_MODE}")
fi
if [ -n "${STARTING_ALIGNMENT_ARTIFACT}" ]; then
    args+=("starting_alignment_artifact=${STARTING_ALIGNMENT_ARTIFACT}")
fi
if [ -n "${STARTING_PERMUTATION_KIND}" ]; then
    args+=("starting_permutation_kind=${STARTING_PERMUTATION_KIND}")
fi

srun python scripts/analysis/run_external_sinkhorn_original_vgg11_cifar10_align_sweep_all.py "${args[@]}"

echo ""
echo "========================================"
echo "ORIGINAL SINKHORN VGG11 CIFAR10 ALIGN SWEEP COMPLETE"
echo "========================================"
echo "Results written under: ${BASE_OUTPUT_ROOT:-<base_output_root from config>}"
