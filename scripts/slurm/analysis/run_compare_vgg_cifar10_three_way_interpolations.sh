#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_compare_vgg_cifar10_three_way_interpolations_%j.out
#SBATCH --error=slurm_compare_vgg_cifar10_three_way_interpolations_%j.err
#SBATCH --job-name=cmp_vgg3way
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

VGG_NAME="${VGG_NAME:-VGG13}"
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg13_seed0/model_final_state_dict.pth}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg13_seed1/model_final_state_dict.pth}"
REBASED_PERM_CHECKPOINT="${REBASED_PERM_CHECKPOINT:-results/vgg13/cifar10/raw_pth_align_sweep_joint_permutation_cor_def/steps150_tau1p0_lr0p75_l1p0_lossmidpoint/rebased_model.pt}"
REBASED_SCALE_CHECKPOINT="${REBASED_SCALE_CHECKPOINT:-results/vgg13/cifar10/raw_pth_align_sweep_joint_scale_cor_def/steps150_tau2p5_lr0p05_l1p0_lossmidpoint_lam0p003/rebased_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vgg13/cifar10/interpolation_comparison_three_way}"
DATA_PATH="${DATA_PATH:-./data}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-51}"
SKIP_PLOTS="${SKIP_PLOTS:-true}"

echo "========================================"
echo "Compare VGG CIFAR10 Three-Way Interpolations"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "VGG_NAME: ${VGG_NAME}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "REBASED_PERM_CHECKPOINT: ${REBASED_PERM_CHECKPOINT}"
echo "REBASED_SCALE_CHECKPOINT: ${REBASED_SCALE_CHECKPOINT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "DATA_PATH: ${DATA_PATH}"
echo "IMAGE_SIZE: ${IMAGE_SIZE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "SKIP_PLOTS: ${SKIP_PLOTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

args=(
    --vgg-name "${VGG_NAME}"
    --model-a-checkpoint "${MODEL_A_CHECKPOINT}"
    --model-b-checkpoint "${MODEL_B_CHECKPOINT}"
    --rebased-perm-checkpoint "${REBASED_PERM_CHECKPOINT}"
    --rebased-scale-checkpoint "${REBASED_SCALE_CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --data-path "${DATA_PATH}"
    --image-size "${IMAGE_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${SLURM_CPUS_PER_TASK}"
    --num-eval-points "${NUM_EVAL_POINTS}"
    --device cuda
)

if [ "${SKIP_PLOTS}" = "true" ]; then
    args+=(--skip-plots)
fi

srun python scripts/analysis/compare_vgg_cifar10_three_way_interpolations.py "${args[@]}"

echo ""
echo "========================================"
echo "COMPARE VGG CIFAR10 THREE-WAY INTERPOLATIONS COMPLETE"
echo "========================================"
echo "Results written under: ${OUTPUT_DIR}"
