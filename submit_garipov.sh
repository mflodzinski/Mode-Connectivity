#!/bin/bash
#SBATCH --partition=gpu          # Request GPU partition
#SBATCH --qos=long               # Long job (up to 48 hours)
#SBATCH --time=24:00:00          # Request 24 hours (adjust if needed)
#SBATCH --ntasks=1               # Single task
#SBATCH --cpus-per-task=4        # 4 CPU cores
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --mem=16GB               # 16GB RAM
#SBATCH --mail-type=END,FAIL     # Email on completion/failure
#SBATCH --output=slurm_%j.out    # Output log
#SBATCH --error=slurm_%j.err     # Error log
#SBATCH --job-name=garipov_vgg16 # Job name

# Load modules (adjust based on DAIC's available modules)
module load cuda/11.8            # Or latest CUDA version
module load python/3.10          # Or use conda

# Navigate to project directory
cd $HOME/Mode-Connectivity

# Set up Python environment (choose ONE option):

# Option A: Using Poetry
poetry install
srun poetry run python run_garipov_endpoints.py

# Option B: Using pip/venv (if poetry doesn't work)
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt  # You'll need to create this
# srun python run_garipov_endpoints.py

# Option C: Using conda (if available)
# conda env create -f environment.yml
# conda activate mode-connectivity
# srun python run_garipov_endpoints.py