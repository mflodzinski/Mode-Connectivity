#!/bin/bash
#SBATCH --partition=general      # Use general partition (contains GPU nodes)
#SBATCH --qos=long               # Long job (up to 48 hours)
#SBATCH --time=24:00:00          # Request 24 hours
#SBATCH --ntasks=1               # Single task
#SBATCH --cpus-per-task=4        # 4 CPU cores
#SBATCH --gpus=1                 # Request 1 GPU (will get a gpu## node)
#SBATCH --mem=16GB               # 16GB RAM
#SBATCH --mail-type=END,FAIL     # Email on completion/failure
#SBATCH --output=slurm_%j.out    # Output log
#SBATCH --error=slurm_%j.err     # Error log
#SBATCH --job-name=garipov_vgg16 # Job name

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd $HOME/Mode-Connectivity

# Run the training script
srun python script.py