#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_fmnist_curve_bezier_%j.out
#SBATCH --error=slurm_fmnist_curve_bezier_%j.err
#SBATCH --job-name=fmnist_curve_bezier
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directories if they don't exist
mkdir -p results/convfc/fmnist/curves/standard/seed0-seed1_bezier/checkpoints
mkdir -p results/convfc/fmnist/curves/standard/seed0-seed1_bezier/evaluations
mkdir -p results/convfc/fmnist/curves/standard/seed0-seed1_bezier/figures

# Run the curve training script
srun python scripts/train/run_garipov_curve.py --config-name fmnist_convfc_curve_seed0-seed1_bezier --config-path ../../configs/garipov/curves
