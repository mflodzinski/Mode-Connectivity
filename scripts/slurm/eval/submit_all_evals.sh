#!/bin/bash
# Submit all evaluation jobs to SLURM

echo "Submitting evaluation jobs..."

# Submit polygon chain evaluation
echo "1. Polygon chain (seed0-mirror)..."
sbatch scripts/slurm/eval/submit_eval_polygon_seed0-mirror.sh

# Submit symmetry plane evaluations
echo "2. Symmetry plane (seed0-mirror)..."
sbatch scripts/slurm/eval/submit_eval_symplane_seed0-mirror.sh

echo "3. Symmetry plane (seed0-seed1)..."
sbatch scripts/slurm/eval/submit_eval_symplane_seed0-seed1.sh

echo ""
echo "All evaluation jobs submitted!"
echo "Check status with: squeue -u $USER"
