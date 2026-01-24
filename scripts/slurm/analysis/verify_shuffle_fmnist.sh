#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_verify_shuffle_fmnist_%j.out
#SBATCH --error=slurm_verify_shuffle_fmnist_%j.err
#SBATCH --job-name=verify_shuffle_fmnist
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo ""
echo "========================================"
echo "Verifying Shuffle Effect on FashionMNIST ConvFC"
echo "========================================"

srun python scripts/analysis/verify_shuffle_effect.py \
  --checkpoint0 results/convfc/fmnist/endpoints/standard/seed0/checkpoints/checkpoint-200.pt \
  --checkpoint1 results/convfc/fmnist/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
  --dataset FashionMNIST \
  --data-path ./data \
  --model ConvFC \
  --transform VGG \
  --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Shuffle verification failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "VERIFICATION COMPLETED"
echo "========================================"
