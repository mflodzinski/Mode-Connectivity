#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_symplane_seed1-mirror_%j.out
#SBATCH --error=slurm_symplane_seed1-mirror_%j.err
#SBATCH --job-name=symplane_seed1-mirror
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directories
mkdir -p results/vgg16/cifar10/advanced_geometry/symmetry_plane/seed1-mirror/checkpoints
mkdir -p results/vgg16/cifar10/advanced_geometry/symmetry_plane/seed1-mirror/evaluations
mkdir -p results/vgg16/cifar10/advanced_geometry/symmetry_plane/seed1-mirror/figures

echo ""
echo "========================================"
echo "STEP 1: Training Symmetry Plane Polygon"
echo "========================================"
# Run the symmetry plane training script
srun python scripts/train/run_garipov_polygon.py --config-name vgg16_symplane_seed1-mirror --project-symmetry-plane

if [ $? -ne 0 ]; then
    echo "Symmetry plane training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Symmetry Plane Polygon"
echo "========================================"
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/advanced_geometry/symmetry_plane/seed1-mirror/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/advanced_geometry/symmetry_plane/seed1-mirror/checkpoint-150.pt \
  --num_points 61 \
  --use_test

if [ $? -ne 0 ]; then
    echo "Symmetry plane evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "TRAINING AND EVALUATION COMPLETED"
echo "========================================"
