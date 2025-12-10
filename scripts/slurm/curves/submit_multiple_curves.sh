#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_multiple_curves_%j.out
#SBATCH --error=slurm_multiple_curves_%j.err
#SBATCH --job-name=multiple_curves
#SBATCH --gres=gpu:a40:1

# Multi-seed Bezier curve training script
# This script trains multiple Bezier curves with different random seeds
# to study the effect of stochasticity on mode connectivity

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# =============================================================================

# Base config to use (without .yaml extension)
CONFIG_NAME="vgg16_curve_seed0-seed1_noreg_multirun"

# Random seeds to use (space-separated)
SEEDS="0 42 123"

# Base output directory (will append _seedX for each run)
# IMPORTANT: Each seed gets its own directory to prevent overwriting
BASE_OUTPUT_DIR="results/vgg16/cifar10/curve_seed0-seed1_noreg_multirun"

# =============================================================================
# SETUP
# =============================================================================

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# =============================================================================
# RUN MULTI-SEED TRAINING
# =============================================================================

echo "================================================================================"
echo "MULTI-SEED BEZIER CURVE TRAINING"
echo "================================================================================"
echo "Config: ${CONFIG_NAME}"
echo "Seeds: ${SEEDS}"
echo "Base output: ${BASE_OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "================================================================================"

# Run the multi-seed training script
srun python scripts/train/run_multiple_curves.py \
    --config-name "${CONFIG_NAME}" \
    --seeds ${SEEDS} \
    --base-output-dir "${BASE_OUTPUT_DIR}"

echo ""
echo "================================================================================"
echo "TRAINING COMPLETED"
echo "================================================================================"
echo "Results saved to: ${BASE_OUTPUT_DIR}_seed*/checkpoints"
echo ""
echo "To compare the curves, run:"
echo "python scripts/analysis/compare_curves.py \\"
for seed in ${SEEDS}; do
    echo "    --checkpoint-dirs ${BASE_OUTPUT_DIR}_seed${seed}/checkpoints \\"
done
echo "    --checkpoint-name checkpoint-50.pt"
echo "================================================================================"
