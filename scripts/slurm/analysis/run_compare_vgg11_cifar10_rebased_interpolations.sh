#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_compare_vgg11_cifar10_rebased_interpolations_%j.out
#SBATCH --error=slurm_compare_vgg11_cifar10_rebased_interpolations_%j.err
#SBATCH --job-name=cmp_v11_interp
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

MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-VGG11_cifar10_0.911.pth}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-VGG11_cifar10_0.9139.pth}"
REBASED_NO_SCALE_CHECKPOINT="${REBASED_NO_SCALE_CHECKPOINT:-results/vgg11/cifar10/raw_pth_align_sweep/steps150_tau1p0_lr0p1_l1p0_lossmidpoint/rebased_model.pt}"
REBASED_SCALE_CHECKPOINT="${REBASED_SCALE_CHECKPOINT:-results/vgg11/cifar10/raw_pth_align_sweep_scale/steps150_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p005/rebased_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vgg11/cifar10/rebased_interpolation_comparison}"
DATA_PATH="${DATA_PATH:-./data}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-50}"

echo "========================================"
echo "Compare VGG11 CIFAR10 Rebased Interpolations"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "REBASED_NO_SCALE_CHECKPOINT: ${REBASED_NO_SCALE_CHECKPOINT}"
echo "REBASED_SCALE_CHECKPOINT: ${REBASED_SCALE_CHECKPOINT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "DATA_PATH: ${DATA_PATH}"
echo "IMAGE_SIZE: ${IMAGE_SIZE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

srun python scripts/analysis/compare_vgg11_cifar10_rebased_interpolations.py \
    --model-a-checkpoint "${MODEL_A_CHECKPOINT}" \
    --model-b-checkpoint "${MODEL_B_CHECKPOINT}" \
    --rebased-no-scale-checkpoint "${REBASED_NO_SCALE_CHECKPOINT}" \
    --rebased-scale-checkpoint "${REBASED_SCALE_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --data-path "${DATA_PATH}" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --num-eval-points "${NUM_EVAL_POINTS}" \
    --device cuda

echo ""
echo "========================================"
echo "COMPARE VGG11 CIFAR10 REBASED INTERPOLATIONS COMPLETE"
echo "========================================"
echo "Results written under: ${OUTPUT_DIR}"
