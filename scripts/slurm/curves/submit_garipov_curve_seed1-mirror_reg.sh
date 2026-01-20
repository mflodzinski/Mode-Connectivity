#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_curve_seed1-mirror_reg_%j.out
#SBATCH --error=slurm_curve_seed1-mirror_reg_%j.err
#SBATCH --job-name=curve_seed1-mirror_reg
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directories if they don't exist
mkdir -p results/vgg16/cifar10/curves/standard/seed1-mirror_reg/checkpoints
mkdir -p results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations
mkdir -p results/vgg16/cifar10/curves/standard/seed1-mirror_reg/figures

echo ""
echo "========================================"
echo "STEP 1: Training Bezier Curve"
echo "========================================"
# Run the curve training script with mirror config
srun python scripts/train/run_garipov_curve.py --config-name vgg16_curve_seed1-mirror_reg

if [ $? -ne 0 ]; then
    echo "Bezier curve training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Linear Interpolation"
echo "========================================"
srun python scripts/eval/eval_linear.py \
    --dir results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations \
    --init_start results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/standard/seed1_mirrored/checkpoints/checkpoint-200.pt \
    --num_points 61 \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --batch_size 128 \
    --num_workers 4 \
    --use_test

if [ $? -ne 0 ]; then
    echo "Linear evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 3: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_seed1-mirror_reg

if [ $? -ne 0 ]; then
    echo "Bezier curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "TRAINING AND EVALUATION COMPLETED"
echo "========================================"
