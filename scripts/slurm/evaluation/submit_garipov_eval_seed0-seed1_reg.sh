#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_seed0-seed1_reg_%j.out
#SBATCH --error=slurm_eval_seed0-seed1_reg_%j.err
#SBATCH --job-name=eval_seed0-seed1_reg
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "========================================"
echo "STEP 1: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name curves/vgg16_curve_seed0-seed1_reg

# Check if curve evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Linear Interpolation"
echo "========================================"
srun python scripts/eval/eval_linear.py \
    --dir results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed1/checkpoint-200.pt \
    --num_points 61 \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --batch_size 128 \
    --use_test

# Check if linear evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Linear evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "EVALUATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Bezier curve: results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/curve.npz"
echo "  - Linear path: results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/linear.npz"
echo ""
echo "To download results:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/*.npz ."
echo ""
echo "To plot comparison (creates figure in results/vgg16/cifar10/curve_seed0-seed1_reg/figures/):"
echo "  python scripts/plot/plot_connectivity.py --linear results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/linear.npz --curve results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/curve.npz --l2_evolution results/vgg16/cifar10/curve_seed0-seed1_reg/evaluations/middle_point_l2_norms.npz --output results/vgg16/cifar10/curve_seed0-seed1_reg/figures/connectivity_comparison.png"
