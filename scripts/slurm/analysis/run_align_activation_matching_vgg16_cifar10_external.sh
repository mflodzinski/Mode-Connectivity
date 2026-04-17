#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_align_activation_matching_vgg16_cifar10_external_%j.out
#SBATCH --error=slurm_align_activation_matching_vgg16_cifar10_external_%j.err
#SBATCH --job-name=align_vgg16_ext
#SBATCH --gres=gpu:a40:1

set -euo pipefail

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplcache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.mplcache"

METHOD="${METHOD:-weight_matching}"
MODEL_A="${MODEL_A:-external/pytorch-vgg-cifar10/save_vgg16_seed0/model_final_state_dict.pth}"
MODEL_B="${MODEL_B:-external/pytorch-vgg-cifar10/save_vgg16_seed1/model_final_state_dict.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vgg16/cifar10/alignment/weight_matching_external_seed0_seed1}"
DATA_PATH="${DATA_PATH:-./data}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_BATCHES="${MAX_BATCHES:-100}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-8192}"
WM_MAX_ITER="${WM_MAX_ITER:-100}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-21}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
LMC_THRESHOLD="${LMC_THRESHOLD:-0.1}"
SEED="${SEED:-0}"
FUNCTIONAL_ATOL="${FUNCTIONAL_ATOL:-1e-5}"
FUNCTIONAL_RTOL="${FUNCTIONAL_RTOL:-1e-4}"

echo "========================================"
echo "Align Activation/Weight Matching VGG16 CIFAR10 External"
echo "========================================"
echo "METHOD: ${METHOD}"
echo "MODEL_A: ${MODEL_A}"
echo "MODEL_B: ${MODEL_B}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "DATA_PATH: ${DATA_PATH}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MAX_BATCHES: ${MAX_BATCHES}"
echo "MAX_ROWS_PER_BATCH: ${MAX_ROWS_PER_BATCH}"
echo "WM_MAX_ITER: ${WM_MAX_ITER}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS}"
echo "EVAL_MAX_BATCHES: ${EVAL_MAX_BATCHES}"
echo "SEED: ${SEED}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

srun python -u scripts/analysis/align_activation_matching.py \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --output-dir "${OUTPUT_DIR}" \
    --method "${METHOD}" \
    --data-path "${DATA_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-batches "${MAX_BATCHES}" \
    --max-rows-per-batch "${MAX_ROWS_PER_BATCH}" \
    --wm-max-iter "${WM_MAX_ITER}" \
    --num-eval-points "${NUM_EVAL_POINTS}" \
    --eval-max-batches "${EVAL_MAX_BATCHES}" \
    --lmc-threshold "${LMC_THRESHOLD}" \
    --seed "${SEED}" \
    --functional-atol "${FUNCTIONAL_ATOL}" \
    --functional-rtol "${FUNCTIONAL_RTOL}"

echo ""
echo "========================================"
echo "ALIGNMENT COMPLETE"
echo "========================================"
echo "Results written under: ${OUTPUT_DIR}"
