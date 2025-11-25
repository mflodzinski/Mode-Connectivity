#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_curve_%j.out
#SBATCH --error=slurm_curve_%j.err
#SBATCH --job-name=garipov_curve

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd $HOME/Mode-Connectivity

# Run the curve training script
srun python scripts/train/run_garipov_curve.py
