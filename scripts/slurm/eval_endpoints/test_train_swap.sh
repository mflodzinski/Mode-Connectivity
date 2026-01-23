#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_test_train_swap_%j.out
#SBATCH --error=slurm_eval_test_train_swap_%j.err
#SBATCH --job-name=eval_test_train_swap
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path so scripts can import from src/
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo ""
echo "========================================"
echo "STEP 1: Evaluating Bezier Curve"
echo "========================================"
srun python scripts/eval/eval_garipov_curve.py --config-name test_train_swap

if [ $? -ne 0 ]; then
    echo "ERROR: Bezier curve evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Polygon Chain"
echo "========================================"
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/advanced_geometry/polygon/seed0-seed1_test_train_swap/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/advanced_geometry/polygon/seed0-seed1/checkpoint-150.pt \
  --num_points 61 \
  --use_test

if [ $? -ne 0 ]; then
    echo "ERROR: Polygon chain evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETED"
echo "========================================"
