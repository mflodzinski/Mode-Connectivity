#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --job-name=verify_lmc
#SBATCH --gres=gpu:a40:1

# Verify LMC connectivity between w_0 and w_1 from shared init training
# Evaluates linear interpolation barrier to confirm low barrier (<1%)

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Checkpoints
W0="results/vgg16/cifar10/endpoints/lmc_connected/seed0/checkpoint-200.pt"
W1="results/vgg16/cifar10/endpoints/lmc_connected/seed1/checkpoint-200.pt"
OUTPUT_DIR="results/vgg16/cifar10/endpoints/lmc_connected/evaluations"

mkdir -p $OUTPUT_DIR

# Evaluate linear interpolation using connect.py
echo "Evaluating linear interpolation between w_0 and w_1..."
srun python external/dnn-mode-connectivity/connect.py \
    --dir $OUTPUT_DIR \
    --dataset CIFAR10 \
    --data_path ./data \
    --transform VGG \
    --model VGG16 \
    --ckpt1 $W0 \
    --ckpt2 $W1 \
    --num_points 21 \
    --use_test

echo ""
echo "Done! Check $OUTPUT_DIR for results."
