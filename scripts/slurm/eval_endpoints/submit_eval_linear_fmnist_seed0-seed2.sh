#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_linear_fmnist_seed0-seed2_%j.out
#SBATCH --error=slurm_eval_linear_fmnist_seed0-seed2_%j.err
#SBATCH --job-name=eval_linear_fmnist_seed0-seed2
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:$PYTHONPATH

# Set paths
ENDPOINT0="results/convfc/fmnist/endpoints/standard/seed0/checkpoints/checkpoint-200.pt"
ENDPOINT1="results/convfc/fmnist/endpoints/standard/seed2/checkpoints/checkpoint-200.pt"
OUTPUT_DIR="results/convfc/fmnist/endpoints/standard/seed0-seed2/evaluations"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to scripts/eval directory to run the script
cd scripts/eval

# Run linear evaluation
srun python evaluate.py \
    --mode linear \
    --init-start "../../${ENDPOINT0}" \
    --init-end "../../${ENDPOINT1}" \
    --num-points 61 \
    --dataset FashionMNIST \
    --model ConvFC \
    --data-path ../../data \
    --dir "../../${OUTPUT_DIR}"

echo "Linear evaluation complete. Output saved to: ${OUTPUT_DIR}/linear.npz"
