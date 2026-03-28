#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_external_sinkhorn_experiment_matrix_perm_then_scale_%j.out
#SBATCH --error=slurm_external_sinkhorn_experiment_matrix_perm_then_scale_%j.err
#SBATCH --job-name=sh_mat_pscl
#SBATCH --gres=gpu:a40:1

CONFIG_NAME="${CONFIG_NAME:-external_sinkhorn_experiment_matrix_perm_then_scale_only}"
export CONFIG_NAME
exec bash "$(dirname "$0")/run_external_sinkhorn_experiment_matrix_mode.sh"
