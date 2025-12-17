#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_calc_init_distances_%j.out
#SBATCH --error=slurm_calc_init_distances_%j.err
#SBATCH --job-name=calc_init_distances
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "CALCULATING L2 DISTANCES BETWEEN MIDDLE CONTROL POINTS"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Recreate initial middle control points using same random seeds"
echo "  2. Extract trained middle control points from checkpoints"
echo "  3. Calculate pairwise L2 distances for both"
echo "  4. Generate distance matrices, summaries, and heatmaps"
echo ""
echo "--------------------------------------------------------------------------------"

# Run the analysis script
srun python scripts/analysis/calculate_init_distances.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "SUCCESS: Distance calculation complete"
    echo "================================================================================"
    echo ""
    echo "Output files:"
    ls -lh results/vgg16/cifar10/curves/initialization/analysis/ 2>/dev/null || echo "No output directory found"
else
    echo ""
    echo "================================================================================"
    echo "ERROR: Distance calculation failed"
    echo "================================================================================"
    exit 1
fi

echo ""
echo "================================================================================"
echo "DONE"
echo "================================================================================"
