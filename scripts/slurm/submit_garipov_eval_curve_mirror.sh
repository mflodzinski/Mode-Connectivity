#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_curve_mirror_%j.out
#SBATCH --error=slurm_eval_curve_mirror_%j.err
#SBATCH --job-name=eval_curve_mirror
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo ""
echo "========================================"
echo "STEP 1: Evaluating Linear Interpolation"
echo "========================================"
srun python scripts/eval/eval_linear.py \
    --dir results/vgg16/cifar10/curve_mirror/evaluations \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
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
echo "STEP 2: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_mirror

if [ $? -ne 0 ]; then
    echo "Bezier curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "EVALUATION COMPLETED SUCCESSFULLY"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Linear path: results/vgg16/cifar10/curve_mirror/evaluations/linear.npz"
echo "  - Bezier curve: results/vgg16/cifar10/curve_mirror/checkpoints/curve.npz"
echo ""
echo "To download results:"
echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve_mirror/evaluations/*.npz ."
echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve_mirror/checkpoints/curve.npz ."
echo ""
