#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_vgg16_curve_seed1-seed2_bezier_%j.out
#SBATCH --error=slurm_vgg16_curve_seed1-seed2_bezier_%j.err
#SBATCH --job-name=vgg16_curve_seed1-seed2_bezier
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directories if they don't exist
mkdir -p results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/checkpoints
mkdir -p results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/evaluations
mkdir -p results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/figures

echo ""
echo "========================================"
echo "STEP 1: Training Bezier Curve"
echo "========================================"
srun python scripts/train/run_garipov_curve.py --config-name vgg16_curve_seed1-seed2_bezier --config-path ../../configs/garipov/curves

if [ $? -ne 0 ]; then
    echo "Bezier curve training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_seed1-seed2_bezier --config-path ../../configs/garipov/curves

if [ $? -ne 0 ]; then
    echo "Bezier curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "TRAINING AND EVALUATION COMPLETED"
echo "========================================"
