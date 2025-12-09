#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_symplane_seed0-mirror_%j.out
#SBATCH --error=slurm_symplane_seed0-mirror_%j.err
#SBATCH --job-name=symplane_seed0-mirror
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "========================================"
echo "SYMMETRY PLANE OPTIMIZATION"
echo "Endpoints: seed0 - mirror"
echo "========================================"

# Create output directory
mkdir -p results/vgg16/cifar10/symmetry_plane_seed0-mirror

# Run optimization
srun python scripts/train/run_symmetry_plane.py \
    --config-name vgg16_symplane_seed0-mirror

# Check if optimization succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Optimization failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "OPTIMIZATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: results/vgg16/cifar10/symmetry_plane_seed0-mirror/"
echo "  - checkpoint_optimal.pt"
echo "  - optimization_log.json"
echo ""
echo "Next step: Run evaluation script"
echo "  sbatch scripts/slurm/symmetry_plane/submit_symplane_eval_seed0-mirror.sh"
echo "========================================"
