#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_randomplane_random_seed0-seed1_%j.out
#SBATCH --error=slurm_randomplane_random_seed0-seed1_%j.err
#SBATCH --job-name=vgg16_randomplane_random
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Run the random plane (random anchor) optimization script
srun python scripts/train/run_random_plane.py --config-name vgg16_randomplane_random_seed0-seed1
