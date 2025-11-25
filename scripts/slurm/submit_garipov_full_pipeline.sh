#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_full_pipeline_%j.out
#SBATCH --error=slurm_full_pipeline_%j.err
#SBATCH --job-name=garipov_full

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd $HOME/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=$HOME/Mode-Connectivity:$PYTHONPATH

echo "========================================"
echo "STEP 1: Training Bezier Curve"
echo "========================================"
srun python scripts/train/run_garipov_curve.py

# Check if curve training succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Curve training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py

# Check if curve evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 3: Evaluating Linear Interpolation"
echo "========================================"
srun python scripts/eval/eval_linear.py \
    --dir results/vgg16/cifar10/curve/evaluations \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed1/checkpoint-200.pt \
    --num_points 61 \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --batch_size 128 \
    --num_workers 4 \
    --use_test

# Check if linear evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Linear evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Bezier curve: results/vgg16/cifar10/curve/evaluations/curve.npz"
echo "  - Linear path: results/vgg16/cifar10/curve/evaluations/linear.npz"
echo ""
echo "To download results:"
echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve/evaluations/*.npz ."
echo ""
echo "To plot comparison (creates figure in results/vgg16/cifar10/curve/figures/):"
echo "  python scripts/plot/plot_connectivity.py --linear evaluations/linear.npz --curve evaluations/curve.npz"
