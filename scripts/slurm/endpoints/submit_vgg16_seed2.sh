#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_vgg16_seed2_%j.out
#SBATCH --error=slurm_vgg16_seed2_%j.err
#SBATCH --job-name=vgg16_seed2
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Run the endpoint training script for seed2
srun python scripts/train/run_garipov_endpoints.py --config-name vgg16_seed2
