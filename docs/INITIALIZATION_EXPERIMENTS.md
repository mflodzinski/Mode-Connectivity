# Initialization Experiments Guide

This guide explains how to run and analyze Bezier curve training experiments with different initialization methods.

## Overview

We compare 10 different initialization strategies for training Bezier curves between two fixed endpoints (seed0 and seed1):

### 1. Biased Linear Initialization (5 variants)
- **alpha = 0.1**: Initialize very close to start endpoint
- **alpha = 0.25**: Initialize closer to start endpoint
- **alpha = 0.5**: Standard midpoint initialization (baseline)
- **alpha = 0.75**: Initialize closer to end endpoint
- **alpha = 0.9**: Initialize very close to end endpoint

### 2. Perturbed Linear Initialization (3 variants)
- **noise = 0.01**: Small Gaussian perturbation off the linear path
- **noise = 0.05**: Medium Gaussian perturbation
- **noise = 0.1**: Large Gaussian perturbation

### 3. Sphere-Constrained Initialization (2 variants)
- **inside**: Project initialization inside the sphere (0.9x radius)
- **outside**: Project initialization outside the sphere (1.1x radius)

All experiments use:
- **Endpoints**: VGG16 trained with seed0 and seed1 (checkpoint-200.pt)
- **Dataset**: CIFAR-10
- **Architecture**: VGG16
- **Curve type**: Bezier with 3 bends
- **Training epochs**: 200
- **Learning rate**: 0.015
- **Weight decay**: 0.0 (no regularization)
- **Seed**: 1

## Running Experiments

### Step 1: Train all initialization variants on the cluster

```bash
# Submit training job (runs all 10 experiments sequentially)
sbatch scripts/slurm/curves/submit_initialization_sweep.sh
```

This will train curves with all 10 initialization methods and save checkpoints to:
```
results/vgg16/cifar10/curve_init_*/checkpoints/checkpoint-200.pt
```

Expected runtime: ~48 hours (all experiments)

### Step 2: Evaluate trained curves on the cluster

```bash
# Submit evaluation job (evaluates all 10 experiments)
sbatch scripts/slurm/eval/submit_eval_initialization_sweep.sh
```

This will evaluate each curve at 61 points and save results to:
```
results/vgg16/cifar10/curve_init_*/evaluations/curve.npz
```

Expected runtime: ~2 hours

### Step 3: Copy results to local machine

```bash
# From your local machine
scp -r mlodzinski@login.delftblue.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/vgg16/cifar10/curve_init_* results/vgg16/cifar10/
```

### Step 4: Analyze and plot results locally

```bash
# Run comparison analysis
poetry run python scripts/analysis/compare_initializations.py \
    --results-dir results/vgg16/cifar10 \
    --output-dir results/vgg16/cifar10/figures
```

This generates:
- `initialization_comparison.png`: 4-panel comparison showing:
  - Panel 1: All biased linear variants
  - Panel 2: All perturbed variants
  - Panel 3: All sphere-constrained variants
  - Panel 4: Barrier height comparison bar chart
- `initialization_comparison_overlay.png`: All methods overlaid on one plot
- Console output with detailed metrics table and summary statistics

## Results Interpretation

### Key Metrics

- **Barrier**: Difference between max test error along curve and average endpoint error
  - Lower is better (flatter curve)
  - Indicates how well the curve avoids high-loss regions

- **Max Test Error**: Worst test error encountered along the entire curve
  - Lower is better
  - Critical for understanding worst-case performance

- **Midpoint Error**: Test error at t=0.5 (middle of the curve)
  - Useful for comparing to linear interpolation baseline

### Expected Results

1. **Biased initialization**: Should show that initialization position (alpha) affects training dynamics but all converge to similarly low barriers
2. **Perturbed initialization**: Small noise should help escape local minima; too much noise may hurt convergence
3. **Sphere-constrained**: Tests whether staying on/near the L2 norm sphere of the endpoints matters for finding low-loss paths

## Configuration Files

All configs are in: `configs/garipov/curves_init/`

Key parameters for each initialization method:

```yaml
# Biased linear example (alpha = 0.75)
init_method: "biased"
init_alpha: 0.75

# Perturbed linear example (noise = 0.05)
init_method: "perturbed"
init_alpha: 0.5
init_noise: 0.05

# Sphere-constrained example (outside)
init_method: "sphere"
init_alpha: 0.5
init_noise: 0.01
init_inside_sphere: false
```

## Implementation Details

### Code Modifications

1. **external/dnn-mode-connectivity/curves.py** (lines 290-386)
   - Added `init_linear_custom(alpha)` method
   - Added `init_perturbed_linear(alpha, noise_scale)` method
   - Added `init_sphere_constrained(alpha, noise_scale, inside)` method

2. **external/dnn-mode-connectivity/train.py** (lines 55-70, 172-188)
   - Added CLI arguments: `--init_method`, `--init_alpha`, `--init_noise`, `--init_inside_sphere`
   - Updated initialization logic to call appropriate method

### Initialization Methods Explained

**Biased Linear (`init_linear_custom`)**:
```python
w_middle = alpha * w_end + (1 - alpha) * w_start
```
- alpha = 0.5 gives standard midpoint
- alpha < 0.5 biases toward start
- alpha > 0.5 biases toward end

**Perturbed Linear (`init_perturbed_linear`)**:
```python
w_middle = alpha * w_end + (1 - alpha) * w_start + noise
noise ~ N(0, noise_scale * ||w_end - w_start||)
```
- Adds Gaussian noise scaled by endpoint distance
- Moves initialization off the direct linear path

**Sphere-Constrained (`init_sphere_constrained`)**:
```python
w_middle = perturbed_linear_point
w_middle = w_middle * (target_radius / ||w_middle||)
target_radius = 0.9 * avg_radius (inside) or 1.1 * avg_radius (outside)
avg_radius = (||w_start|| + ||w_end||) / 2
```
- First applies perturbed linear initialization
- Then projects to sphere with controlled radius
- Tests hypothesis that staying near endpoint L2 norms helps

## Quick Reference Commands

```bash
# Train all variants
sbatch scripts/slurm/curves/submit_initialization_sweep.sh

# Evaluate all variants
sbatch scripts/slurm/eval/submit_eval_initialization_sweep.sh

# Copy results from cluster
scp -r mlodzinski@login.delftblue.tudelft.nl:/tudelft.net/staff-bulk/.../curve_init_* results/vgg16/cifar10/

# Analyze results
poetry run python scripts/analysis/compare_initializations.py

# Check job status
squeue -u mlodzinski

# View job output
tail -f slurm_init_sweep_*.out
```

## Troubleshooting

**Issue**: Some experiments fail during training
- Check `slurm_init_sweep_*.err` for error messages
- Verify endpoints exist: `results/vgg16/cifar10/endpoints/standard/seed{0,1}/checkpoint-200.pt`

**Issue**: Evaluation script can't find checkpoints
- Verify training completed: check for `checkpoint-200.pt` in each `curve_init_*/checkpoints/` directory
- Re-run failed experiments individually using their config files

**Issue**: Analysis script reports missing results
- Ensure evaluations have been copied from cluster
- Check that `curve.npz` exists in each `curve_init_*/evaluations/` directory
