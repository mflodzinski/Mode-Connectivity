#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_polygon_seed0-mirror_%j.out
#SBATCH --error=slurm_eval_polygon_seed0-mirror_%j.err
#SBATCH --job-name=eval_polygon_seed0-mirror
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directory
mkdir -p results/vgg16/cifar10/polygon_seed0-mirror/evaluations

# Run evaluation
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/polygon_seed0-mirror/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/polygon_seed0-mirror/checkpoint-150.pt \
  --num_points 61 \
  --use_test
