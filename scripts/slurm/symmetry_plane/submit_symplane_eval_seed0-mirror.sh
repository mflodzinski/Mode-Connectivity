#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_symplane_eval_seed0-mirror_%j.out
#SBATCH --error=slurm_symplane_eval_seed0-mirror_%j.err
#SBATCH --job-name=symplane_eval_seed0-mirror
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directories
mkdir -p results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations
mkdir -p results/vgg16/cifar10/symmetry_plane_seed0-mirror/figures

echo "========================================"
echo "STEP 1: Evaluating Symmetry Plane Path"
echo "========================================"
srun python scripts/eval/eval_symmetry_plane.py \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --theta_checkpoint results/vgg16/cifar10/symmetry_plane_seed0-mirror/checkpoint_optimal.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
    --model VGG16 \
    --dataset CIFAR10 \
    --data_path ./data \
    --transform VGG \
    --use_test \
    --num_points 61 \
    --batch_size 128 \
    --num_workers 4 \
    --dir results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Symmetry plane evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Running Comparison"
echo "========================================"
srun python scripts/eval/eval_symmetry_comparison.py \
    --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
    --theta_checkpoint results/vgg16/cifar10/symmetry_plane_seed0-mirror/checkpoint_optimal.pt \
    --model VGG16 \
    --dataset CIFAR10 \
    --data_path ./data \
    --transform VGG \
    --use_test \
    --num_points 61 \
    --batch_size 128 \
    --num_workers 4 \
    --dir results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations

# Check if comparison succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Comparison evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "EVALUATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Symmetry plane: results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/symmetry_plane.npz"
echo "  - Comparison: results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/comparison.npz"
echo ""
echo "To download results:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/*.npz ."
echo ""
echo "To plot comparison (run locally):"
echo "  python scripts/plot/plot_symmetry_plane_comparison.py --comparison-file results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/comparison.npz"
echo "========================================"
