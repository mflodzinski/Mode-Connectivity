#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_l2_test_%j.out
#SBATCH --error=slurm_eval_l2_test_%j.err
#SBATCH --job-name=garipov_eval_l2_test
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "========================================"
echo "L2 Norm Test: Evaluating Bezier Curve"
echo "Endpoints: seed0 and seed0_mirrored"
echo "Config: vgg16_curve_l2_test"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_l2_test

# Check if curve evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Curve evaluation failed"
    exit 1
fi

echo ""
echo "========================================"
echo "L2 Norm Test: Evaluating Linear Path"
echo "========================================"
srun python scripts/eval/eval_linear.py \
    --dir results/vgg16/cifar10/curve_l2_test/evaluations \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
    --num_points 61 \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --batch_size 128 \
    --use_test

# Check if linear evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Linear evaluation failed"
    exit 1
fi

echo ""
echo "========================================"
echo "L2 Norm Test: Evaluation Complete"
echo "========================================"
echo "Results saved to:"
echo "  - Bezier curve: results/vgg16/cifar10/curve_l2_test/evaluations/curve.npz"
echo "  - Linear path: results/vgg16/cifar10/curve_l2_test/evaluations/linear.npz"
echo ""
echo "To download results:"
echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve_l2_test/evaluations/*.npz results/vgg16/cifar10/curve_l2_test/evaluations/"
echo ""
echo "To plot comparison (creates figure with L2 norm):"
echo "  cd results/vgg16/cifar10/curve_l2_test"
echo "  python scripts/plot/plot_connectivity.py --linear evaluations/linear.npz --curve evaluations/curve.npz --l2_evolution evaluations/middle_point_l2_norms.npz"
