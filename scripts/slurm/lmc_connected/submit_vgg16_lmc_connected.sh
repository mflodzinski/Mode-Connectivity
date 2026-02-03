#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --job-name=vgg16_lmc_connected
#SBATCH --gres=gpu:a40:1

# Train LMC-connected pair using shared initialization with split training
# Stage 1: Train shared init for 10 epochs
# Stage 2: Split training with different batch seeds to convergence (200 epochs)

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Run the LMC connected pair training script
srun python scripts/train/run_lmc_connected_pair.py
