#!/bin/bash
# Common launcher for Sinkhorn experiment-matrix modes.

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_experiment_matrix_%j.out
#SBATCH --error=slurm_external_sinkhorn_experiment_matrix_%j.err
#SBATCH --job-name=ext_sh_matrix
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

CONFIG_NAME="${CONFIG_NAME:-}"
DATASET="${DATASET:-}"
VGG_NAME="${VGG_NAME:-}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-}"
DATA_PATH="${DATA_PATH:-}"
SEED="${SEED:-}"
SPLIT_SEED="${SPLIT_SEED:-}"
VAL_FRACTION="${VAL_FRACTION:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_EVAL_POINTS="${NUM_EVAL_POINTS:-}"
PERMUTATION_ONLY_ROOT="${PERMUTATION_ONLY_ROOT:-}"
MODEL_0_NAME="${MODEL_0_NAME:-}"
MODEL_0_CHECKPOINT="${MODEL_0_CHECKPOINT:-}"
MODEL_1_NAME="${MODEL_1_NAME:-}"
MODEL_1_CHECKPOINT="${MODEL_1_CHECKPOINT:-}"
MODEL_2_NAME="${MODEL_2_NAME:-}"
MODEL_2_CHECKPOINT="${MODEL_2_CHECKPOINT:-}"

if [ -z "${CONFIG_NAME}" ]; then
    echo "CONFIG_NAME must be set."
    exit 1
fi

echo "========================================"
echo "Sinkhorn Experiment Matrix"
echo "========================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "DATASET: ${DATASET:-<from config>}"
echo "VGG_NAME: ${VGG_NAME:-<from config>}"
echo "BASE_OUTPUT_ROOT: ${BASE_OUTPUT_ROOT:-<from config>}"
echo "DATA_PATH: ${DATA_PATH:-<from config>}"
echo "SEED: ${SEED:-<from config>}"
echo "SPLIT_SEED: ${SPLIT_SEED:-<from config>}"
echo "VAL_FRACTION: ${VAL_FRACTION:-<from config>}"
echo "BATCH_SIZE: ${BATCH_SIZE:-<from config>}"
echo "NUM_EVAL_POINTS: ${NUM_EVAL_POINTS:-<from config>}"
echo "PERMUTATION_ONLY_ROOT: ${PERMUTATION_ONLY_ROOT:-<from config>}"
echo "MODEL_0: ${MODEL_0_NAME:-<from config>} ${MODEL_0_CHECKPOINT:-<from config>}"
echo "MODEL_1: ${MODEL_1_NAME:-<from config>} ${MODEL_1_CHECKPOINT:-<from config>}"
echo "MODEL_2: ${MODEL_2_NAME:-<from config>} ${MODEL_2_CHECKPOINT:-<from config>}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "dot: $(command -v dot || echo missing)"
echo "EXTRA_PYTHONPATH: ${EXTRA_PYTHONPATH}"
echo ""

args=(
    "--config-name" "${CONFIG_NAME}"
    "num_workers=${SLURM_CPUS_PER_TASK}"
    "device=cuda"
)
if [ -n "${DATASET}" ]; then
    args+=("dataset=${DATASET}")
fi
if [ -n "${VGG_NAME}" ]; then
    args+=("vgg_name=${VGG_NAME}")
fi
if [ -n "${BASE_OUTPUT_ROOT}" ]; then
    args+=("base_output_root=${BASE_OUTPUT_ROOT}")
fi
if [ -n "${DATA_PATH}" ]; then
    args+=("data_path=${DATA_PATH}")
fi
if [ -n "${SEED}" ]; then
    args+=("seed=${SEED}")
fi
if [ -n "${SPLIT_SEED}" ]; then
    args+=("split_seed=${SPLIT_SEED}")
fi
if [ -n "${VAL_FRACTION}" ]; then
    args+=("val_fraction=${VAL_FRACTION}")
fi
if [ -n "${BATCH_SIZE}" ]; then
    args+=("batch_size=${BATCH_SIZE}")
fi
if [ -n "${NUM_EVAL_POINTS}" ]; then
    args+=("num_eval_points=${NUM_EVAL_POINTS}")
fi
if [ -n "${PERMUTATION_ONLY_ROOT}" ]; then
    args+=("permutation_only_root=${PERMUTATION_ONLY_ROOT}")
fi
if [ -n "${MODEL_0_NAME}" ]; then
    args+=("endpoint_models[0].name=${MODEL_0_NAME}")
fi
if [ -n "${MODEL_0_CHECKPOINT}" ]; then
    args+=("endpoint_models[0].checkpoint=${MODEL_0_CHECKPOINT}")
fi
if [ -n "${MODEL_1_NAME}" ]; then
    args+=("endpoint_models[1].name=${MODEL_1_NAME}")
fi
if [ -n "${MODEL_1_CHECKPOINT}" ]; then
    args+=("endpoint_models[1].checkpoint=${MODEL_1_CHECKPOINT}")
fi
if [ -n "${MODEL_2_NAME}" ]; then
    args+=("endpoint_models[2].name=${MODEL_2_NAME}")
fi
if [ -n "${MODEL_2_CHECKPOINT}" ]; then
    args+=("endpoint_models[2].checkpoint=${MODEL_2_CHECKPOINT}")
fi

srun python scripts/analysis/run_external_sinkhorn_experiment_matrix.py "${args[@]}"

echo ""
echo "========================================"
echo "SINKHORN EXPERIMENT MATRIX COMPLETE"
echo "========================================"
