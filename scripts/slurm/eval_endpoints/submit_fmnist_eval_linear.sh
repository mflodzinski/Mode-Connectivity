#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_fmnist_eval_linear_%j.out
#SBATCH --error=slurm_fmnist_eval_linear_%j.err
#SBATCH --job-name=fmnist_eval_linear
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root and scripts to Python path
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:$PYTHONPATH
ENDPOINT0="${PROJECT_ROOT}/results/convfc/fmnist/endpoints/standard/seed0/checkpoint-200.pt"
ENDPOINT1="${PROJECT_ROOT}/results/convfc/fmnist/endpoints/standard/seed1/checkpoint-200.pt"
EVAL_DIR="${PROJECT_ROOT}/results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations"

# Verify endpoints exist
if [ ! -f "${ENDPOINT0}" ]; then
    echo "ERROR: Endpoint 0 not found: ${ENDPOINT0}"
    exit 1
fi

if [ ! -f "${ENDPOINT1}" ]; then
    echo "ERROR: Endpoint 1 not found: ${ENDPOINT1}"
    exit 1
fi

echo "Evaluating Linear Interpolation for Fashion-MNIST ConvFC seed0-seed1"
echo "Endpoint 0: ${ENDPOINT0}"
echo "Endpoint 1: ${ENDPOINT1}"

# Create output directory
mkdir -p "${EVAL_DIR}"

# Change to scripts/eval directory to run the script
cd scripts/eval

# Run linear evaluation (using absolute paths for files)
srun python evaluate.py \
    --mode linear \
    --dir "${EVAL_DIR}" \
    --init_start "${ENDPOINT0}" \
    --init_end "${ENDPOINT1}" \
    --num_points 61 \
    --dataset FashionMNIST \
    --data_path "${PROJECT_ROOT}/data" \
    --model ConvFC \
    --transform VGG \
    --batch_size 128 \
    --use_test

echo "Evaluation complete. Output saved to: ${EVAL_DIR}/linear.npz"
