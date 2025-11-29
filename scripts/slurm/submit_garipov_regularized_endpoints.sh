#!/bin/bash
#SBATCH --partition=general          # Request GPU partition
#SBATCH --qos=long               # Long job (up to 48 hours)
#SBATCH --time=1:00:00          # Request 24 hours (adjust if needed)
#SBATCH --ntasks=1               # Single task
#SBATCH --cpus-per-task=4        # 4 CPU cores
#SBATCH --mem=4GB               # 16GB RAM
#SBATCH --mail-type=END,FAIL     # Email on completion/failure
#SBATCH --output=slurm_regularized_endpoints_%j.out    # Output log
#SBATCH --error=slurm_regularized_endpoints_%j.err     # Error log
#SBATCH --job-name=garipov_regularized_endpoints # Job name
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Run the training script with regularized endpoints config
srun python scripts/train/run_garipov_endpoints.py \
    --config-name=vgg16_regularized_endpoints
