# Mode Connectivity Experiments

This document describes the organization and naming conventions for mode connectivity experiments.

## Directory Structure

```
results/vgg16/cifar10/
├── endpoints/                          # Trained endpoint models
│   └── checkpoints/
│       ├── seed0/
│       ├── seed1/
│       └── seed0_mirrored/
│
├── curve_seed0-seed1_reg/             # Curve with regularization (wd=5e-4)
│   ├── checkpoints/
│   ├── evaluations/
│   └── figures/
│
├── curve_seed0-seed1_noreg/           # Curve WITHOUT regularization (wd=0.0)
│   ├── checkpoints/
│   ├── evaluations/
│   └── figures/
│
├── curve_seed0-mirror_reg/            # Mirror curve with regularization
│   ├── checkpoints/
│   ├── evaluations/
│   └── figures/
│
└── curve_seed0-mirror_noreg/          # Mirror curve WITHOUT regularization
    ├── checkpoints/
    ├── evaluations/
    └── figures/
```

## Naming Convention

Experiment directories follow the pattern: `curve_{endpoint1}-{endpoint2}_{variant}`

### Endpoint Identifiers
- `seed0` - Model trained with random seed 0
- `seed1` - Model trained with random seed 1
- `mirror` - Permuted/mirrored version of seed0

### Variants
- `reg` - With L2 weight decay regularization (wd=5e-4)
- `noreg` - Without regularization (wd=0.0)

### Examples
- `curve_seed0-seed1_reg` - Bezier curve between seed0 and seed1 WITH regularization
- `curve_seed0-seed1_noreg` - Bezier curve between seed0 and seed1 WITHOUT regularization
- `curve_seed0-mirror_reg` - Bezier curve between seed0 and mirrored seed0 WITH regularization

## Experiment Files

### Checkpoints Directory
Contains trained model checkpoints:
- `checkpoint-{epoch}.pt` - Model checkpoint at specific epoch
- Final checkpoint typically at epoch 200

### Evaluations Directory
Contains evaluation results (`.npz` files):
- `linear.npz` - Linear interpolation evaluation (61 points from t=0 to t=1)
- `curve.npz` - Bezier curve evaluation (61 points along trained curve)
- `middle_point_l2_norms.npz` - L2 norms tracked during training

### Figures Directory
Contains generated plots:
- `connectivity_comparison.png` - 6-panel comparison plot showing:
  - Test error along path
  - Test loss along path
  - Train error along path
  - Train loss along path
  - L2 norm along path
  - L2 norm evolution during training

## Configuration Files

Located in `configs/garipov/`:

- `vgg16_curve_seed0-seed1_reg.yaml` - Config for seed0-seed1 with regularization
- `vgg16_curve_seed0-seed1_noreg.yaml` - Config for seed0-seed1 without regularization
- `vgg16_curve_seed0-mirror_reg.yaml` - Config for seed0-mirror with regularization
- `vgg16_curve_seed0-mirror_noreg.yaml` - Config for seed0-mirror without regularization

Key parameters:
- `wd: 5e-4` for regularized variants
- `wd: 0.0` for no-regularization variants
- `lr: 0.015` - Learning rate for curve training
- `epochs: 200` - Training epochs
- `curve: "Bezier"` - Curve type
- `num_bends: 3` - Number of control points (start, middle, end)

## SLURM Scripts

Located in `scripts/slurm/`:

### Training Scripts
- `submit_garipov_curve_seed0-seed1_reg.sh` - Train seed0-seed1 with regularization
- `submit_garipov_curve_seed0-seed1_noreg.sh` - Train seed0-seed1 without regularization
- `submit_garipov_curve_seed0-mirror_reg.sh` - Train seed0-mirror with regularization
- `submit_garipov_curve_seed0-mirror_noreg.sh` - Train seed0-mirror without regularization

### Evaluation Scripts
- `submit_garipov_eval_seed0-seed1_reg.sh` - Evaluate seed0-seed1 with regularization
- `submit_garipov_eval_seed0-seed1_noreg.sh` - Evaluate seed0-seed1 without regularization
- `submit_garipov_eval_seed0-mirror_reg.sh` - Evaluate seed0-mirror with regularization
- `submit_garipov_eval_seed0-mirror_noreg.sh` - Evaluate seed0-mirror without regularization

## Running Experiments

### On Cluster

```bash
# Train curve (generates checkpoints and middle_point_l2_norms.npz)
sbatch scripts/slurm/submit_garipov_curve_seed0-seed1_noreg.sh

# Evaluate curve (generates linear.npz and curve.npz)
sbatch scripts/slurm/submit_garipov_eval_seed0-seed1_noreg.sh
```

### Download Results

```bash
# Download evaluation files
scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/*.npz results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/
```

### Generate Plots Locally

```bash
python scripts/plot/plot_connectivity.py \
    --linear results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/linear.npz \
    --curve results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/curve.npz \
    --l2_evolution results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/middle_point_l2_norms.npz \
    --output results/vgg16/cifar10/curve_seed0-seed1_noreg/figures/connectivity_comparison.png \
    --title "Mode Connectivity: seed0-seed1 (No Regularization)"
```

## Adding New Experiments

To add a new experimental variant:

1. **Create config file**: `configs/garipov/vgg16_curve_{endpoints}_{variant}.yaml`
   - Set `experiment_name`, `output_root`, and variant-specific parameters

2. **Create SLURM scripts**:
   - Training: `scripts/slurm/submit_garipov_curve_{endpoints}_{variant}.sh`
   - Evaluation: `scripts/slurm/submit_garipov_eval_{endpoints}_{variant}.sh`

3. **Create results directory**: `results/vgg16/cifar10/curve_{endpoints}_{variant}/`
   - Subdirectories: `checkpoints/`, `evaluations/`, `figures/`

4. **Update this README** with the new experiment details

## Experimental Comparisons

Common comparisons:
- **Regularization effect**: Compare `_reg` vs `_noreg` variants
- **Endpoint distance**: Compare `seed0-seed1` (different initializations) vs `seed0-mirror` (permutation)
- **Training dynamics**: Analyze L2 norm evolution in `middle_point_l2_norms.npz`
- **Loss landscape**: Compare linear vs Bezier curve paths in connectivity plots
