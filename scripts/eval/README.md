# Evaluation Scripts

Scripts for evaluating mode connectivity along different interpolation paths.

## Overview

This directory contains scripts for evaluating neural networks along various connectivity paths:

1. **evaluate.py** - Unified evaluation script (4 modes)
2. **eval_curve_detailed.py** - Detailed curve evaluation with feature extraction
3. **eval_garipov_curve.py** - Hydra wrapper for external Garipov curve training

All scripts use the unified library at `scripts/lib/`.

---

## Scripts Reference

### 1. evaluate.py (Main Evaluation Script)

Unified script for evaluating mode connectivity along different paths.

**Modes:**
- `linear` - Linear interpolation between two endpoints
- `curve` - Bezier/PolyChain curve evaluation
- `symmetry` - Symmetry plane path (w1 → θ* → w2)
- `comparison` - Compare linear vs symmetry plane paths

---

#### Mode: linear

Evaluate linear interpolation between two independently trained models.

**Purpose:**
- Measure loss/accuracy along straight line in weight space
- Compute L2 norm along path
- Detect loss barriers between models

**Basic Usage:**
```bash
python evaluate.py \
    --mode linear \
    --init-start path/to/model1.pt \
    --init-end path/to/model2.pt \
    --dir results/linear \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data
```

**Complete Example:**
```bash
python evaluate.py \
    --mode linear \
    --init-start checkpoints/seed0/model.pt \
    --init-end checkpoints/seed1/model.pt \
    --dir results/linear_seed0_seed1 \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --batch-size 128 \
    --num-workers 4 \
    --num-points 25
```

**Arguments:**
```bash
--init-start PATH         # First endpoint checkpoint
--init-end PATH           # Second endpoint checkpoint
--dir DIR                 # Output directory
--num-points N            # Number of evaluation points (default: 25)
```

**Output:**
- `linear.npz` - Results with fields:
  - `ts` - t values (0 to 1)
  - `l2_norm` - L2 norm at each t
  - `tr_loss`, `tr_acc`, `tr_err` - Training metrics
  - `te_loss`, `te_acc`, `te_err` - Test metrics

---

#### Mode: curve

Evaluate Bezier or PolyChain curve connecting two endpoints.

**Purpose:**
- Evaluate trained curve models
- Measure loss/accuracy along curve
- Compare with linear interpolation

**Basic Usage:**
```bash
python evaluate.py \
    --mode curve \
    --ckpt path/to/curve.pt \
    --dir results/curve \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --curve Bezier \
    --num-bends 3
```

**Complete Example:**
```bash
python evaluate.py \
    --mode curve \
    --ckpt checkpoints/curve_seed0_seed1.pt \
    --dir results/curve_evaluation \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --curve Bezier \
    --num-bends 3 \
    --num-points 61 \
    --batch-size 128
```

**Arguments:**
```bash
--ckpt PATH               # Curve checkpoint
--curve TYPE              # Curve type (Bezier, PolyChain)
--num-bends N             # Number of bend points (default: 3)
--num-points N            # Evaluation points (default: 61)
```

**Output:**
- `curve.npz` - Results with same format as linear mode

---

#### Mode: symmetry

Evaluate symmetry plane path: w1 → θ* → w2.

**Purpose:**
- Evaluate path through symmetry point
- Compare with direct linear path
- Verify symmetry plane properties

**Basic Usage:**
```bash
python evaluate.py \
    --mode symmetry \
    --init-start path/to/w1.pt \
    --init-end path/to/w2.pt \
    --init-theta path/to/theta.pt \
    --dir results/symmetry \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data
```

**Complete Example:**
```bash
python evaluate.py \
    --mode symmetry \
    --init-start checkpoints/seed0.pt \
    --init-end checkpoints/seed1.pt \
    --init-theta checkpoints/theta_star.pt \
    --dir results/symmetry_plane \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --num-points 25
```

**Arguments:**
```bash
--init-start PATH         # First endpoint (w1)
--init-end PATH           # Second endpoint (w2)
--init-theta PATH         # Symmetry point (θ*)
--num-points N            # Points per segment (default: 25)
```

**Output:**
- `symmetry_plane.npz` - Results for path w1 → θ* → w2

---

#### Mode: comparison

Compare linear interpolation vs symmetry plane path.

**Purpose:**
- Direct comparison of two paths
- Measure difference in loss barriers
- Visualize path comparison

**Basic Usage:**
```bash
python evaluate.py \
    --mode comparison \
    --init-start path/to/w1.pt \
    --init-end path/to/w2.pt \
    --init-theta path/to/theta.pt \
    --dir results/comparison \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data
```

**Complete Example:**
```bash
python evaluate.py \
    --mode comparison \
    --init-start checkpoints/seed0.pt \
    --init-end checkpoints/seed1.pt \
    --init-theta checkpoints/theta_star.pt \
    --dir results/linear_vs_symmetry \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --num-points 25 \
    --batch-size 256
```

**Output:**
- `comparison.npz` - Combined results with fields:
  - `ts` - t values
  - `linear_*` - Linear path metrics
  - `symplane_*` - Symmetry plane metrics

---

### 2. eval_curve_detailed.py

Evaluate curve with per-sample predictions and feature extraction.

**Purpose:**
- Collect predictions at each point along curve
- Extract features from endpoint models
- Enable detailed prediction analysis

**Basic Usage:**
```bash
python eval_curve_detailed.py \
    --curve path/to/curve.pt \
    --endpoint-0 path/to/w1.pt \
    --endpoint-1 path/to/w2.pt \
    --output results/detailed_predictions.npz \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data
```

**Complete Example:**
```bash
python eval_curve_detailed.py \
    --curve checkpoints/curve_bezier.pt \
    --endpoint-0 checkpoints/seed0.pt \
    --endpoint-1 checkpoints/seed1.pt \
    --output results/detailed/curve_predictions.npz \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --curve-type Bezier \
    --num-bends 3 \
    --num-points 61 \
    --batch-size 128 \
    --num-samples 1000
```

**Arguments:**
```bash
--curve PATH              # Curve checkpoint
--endpoint-0 PATH         # First endpoint checkpoint
--endpoint-1 PATH         # Second endpoint checkpoint
--output PATH             # Output NPZ file
--curve-type TYPE         # Curve type (Bezier, PolyChain)
--num-bends N             # Number of bend points
--num-points N            # Evaluation points
--num-samples N           # Samples to analyze (default: all)
```

**Output:**
- `predictions.npz` - Detailed data with fields:
  - `predictions` - Predictions at each t (num_samples, num_t_values, num_classes)
  - `targets` - Ground truth labels (num_samples,)
  - `features_t0` - Features from endpoint 0 model
  - `features_t1` - Features from endpoint 1 model
  - `images` - Sample images
  - `ts` - t values

**Use Case:**
After generating detailed predictions, analyze them:
```bash
# Generate detailed predictions
python eval_curve_detailed.py \
    --curve curve.pt \
    --endpoint-0 w1.pt \
    --endpoint-1 w2.pt \
    --output predictions.npz \
    --model VGG16

# Analyze predictions
cd ../analysis
python analyze_predictions.py \
    --mode all \
    --predictions ../eval/predictions.npz \
    --output-dir results/prediction_analysis
```

---

### 3. eval_garipov_curve.py

Hydra-based wrapper for external Garipov curve training script.

**Purpose:**
- Train curves using original Garipov et al. method
- Wrapper for `external/dnn-mode-connectivity/train.py`

**Note:** This is primarily for training, not evaluation. See external repository documentation for usage.

---

## Common Arguments

All evaluation scripts share these arguments:

```bash
# Model arguments
--model VGG16              # Architecture (VGG16, VGG19, ResNet18, etc.)
--dataset CIFAR10          # Dataset (default: CIFAR10)
--data-path ./data         # Path to dataset directory

# Evaluation arguments
--batch-size 128           # Batch size (default: 128)
--num-workers 4            # DataLoader workers (default: 4)
--num-points N             # Number of evaluation points
--transform VGG            # Data transform (default: VGG)

# Device (auto-detected)
# Priority: CUDA > MPS (Apple Silicon) > CPU
```

---

## Typical Workflows

### Workflow 1: Evaluate Linear Path

```bash
# Evaluate linear interpolation
python evaluate.py \
    --mode linear \
    --init-start checkpoints/model_seed0.pt \
    --init-end checkpoints/model_seed1.pt \
    --dir results/linear_path \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --num-points 25
```

### Workflow 2: Evaluate Trained Curve

```bash
# 1. Train curve (using training scripts)

# 2. Evaluate curve
python evaluate.py \
    --mode curve \
    --ckpt checkpoints/curve_trained.pt \
    --dir results/curve_eval \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --curve Bezier \
    --num-bends 3 \
    --num-points 61
```

### Workflow 3: Compare Paths

```bash
# Compare linear vs symmetry plane
python evaluate.py \
    --mode comparison \
    --init-start checkpoints/w1.pt \
    --init-end checkpoints/w2.pt \
    --init-theta checkpoints/theta_star.pt \
    --dir results/path_comparison \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data \
    --num-points 25
```

### Workflow 4: Detailed Analysis

```bash
# 1. Generate detailed predictions
python eval_curve_detailed.py \
    --curve checkpoints/curve.pt \
    --endpoint-0 checkpoints/seed0.pt \
    --endpoint-1 checkpoints/seed1.pt \
    --output results/predictions.npz \
    --model VGG16 \
    --dataset CIFAR10 \
    --data-path ./data

# 2. Analyze predictions (in analysis directory)
cd ../analysis
python analyze_predictions.py \
    --mode all \
    --predictions ../eval/results/predictions.npz \
    --output-dir results/prediction_analysis \
    --dataset-path ./data
```

---

## Output Format

### Standard Evaluation Output (.npz)

All evaluation modes produce NPZ files with these fields:

```python
import numpy as np

# Load results
data = np.load('results/linear.npz')

# Access data
ts = data['ts']              # t values (shape: num_points)
tr_loss = data['tr_loss']    # Training loss
tr_acc = data['tr_acc']      # Training accuracy (%)
tr_err = data['tr_err']      # Training error (%)
te_loss = data['te_loss']    # Test loss
te_acc = data['te_acc']      # Test accuracy (%)
te_err = data['te_err']      # Test error (%)

# Linear mode also includes:
l2_norm = data['l2_norm']    # L2 norm at each t
```

### Comparison Output

```python
data = np.load('results/comparison.npz')

# Linear path metrics
linear_tr_loss = data['linear_tr_loss']
linear_te_acc = data['linear_te_acc']

# Symmetry plane metrics
symplane_tr_loss = data['symplane_tr_loss']
symplane_te_acc = data['symplane_te_acc']
```

### Detailed Predictions Output

```python
data = np.load('results/predictions.npz')

predictions = data['predictions']      # (num_samples, num_points, num_classes)
targets = data['targets']              # (num_samples,)
features_t0 = data['features_t0']      # (num_samples, feature_dim)
features_t1 = data['features_t1']      # (num_samples, feature_dim)
images = data['images']                # (num_samples, C, H, W)
ts = data['ts']                        # (num_points,)
```

---

## Plotting Results

After evaluation, use plotting scripts to visualize:

```bash
cd ../plot

# Plot single curve
python plot_curve.py \
    --data ../eval/results/curve.npz \
    --output plots/curve_plot.png

# Plot comparison
python plot_polygon_symmetry_comparison.py \
    --linear ../eval/results/linear.npz \
    --symmetry ../eval/results/symmetry.npz \
    --output plots/comparison.png
```

---

## Performance Tips

### Speed Up Evaluation

1. **Reduce evaluation points** for quick testing:
   ```bash
   --num-points 13  # Quick evaluation
   --num-points 25  # Standard
   --num-points 61  # Detailed (default for curves)
   ```

2. **Increase batch size** on GPU:
   ```bash
   --batch-size 256  # Faster on GPU
   --batch-size 512  # If memory allows
   ```

3. **Adjust workers**:
   ```bash
   --num-workers 8   # More workers for faster data loading
   --num-workers 0   # If debugging or memory constrained
   ```

### Memory Management

If running out of memory:
```bash
--batch-size 64       # Reduce batch size
--num-workers 2       # Reduce workers
--num-samples 1000    # For detailed eval, limit samples
```

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'lib'
```

**Solution:** Run from `scripts/eval/` directory, not from subdirectories.

### Checkpoint Mismatch

```
RuntimeError: Error loading checkpoint
```

**Solution:**
- Verify `--model` matches checkpoint architecture
- Check `--num-classes` is correct (default: 10)
- Ensure checkpoint file exists and isn't corrupted

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `--batch-size` to 64 or 32
- Reduce `--num-workers` to 2
- Use CPU: `CUDA_VISIBLE_DEVICES="" python evaluate.py ...`

### Curve Checkpoint Issues

```
KeyError: 'coefficients_0'
```

**Solution:**
- Ensure `--curve` type matches checkpoint (Bezier vs PolyChain)
- Verify `--num-bends` matches trained curve

---

## Batch Evaluation

Evaluate multiple paths in a loop:

```bash
#!/bin/bash

# Evaluate multiple seed pairs
for seed1 in 0 1 2; do
    for seed2 in 0 1 2; do
        if [ $seed1 -lt $seed2 ]; then
            python evaluate.py \
                --mode linear \
                --init-start checkpoints/seed${seed1}.pt \
                --init-end checkpoints/seed${seed2}.pt \
                --dir results/linear_seed${seed1}_${seed2} \
                --model VGG16 \
                --dataset CIFAR10 \
                --data-path ./data
        fi
    done
done
```

---

## Library Dependency

All evaluation scripts depend on the unified library at `scripts/lib/`.

Import pattern:
```python
import sys
sys.path.insert(0, '../lib')

from lib.core import setup, checkpoint, models, data, output
from lib.evaluation import interpolation, evaluate
```

The `EvalSetup` wrapper provides backward-compatible interface:
```python
EvalSetup = setup.EvalSetup

# Use unified interface
device = EvalSetup.get_device()
loaders, num_classes = EvalSetup.load_data(...)
model = EvalSetup.create_standard_model(...)
```

See `scripts/lib/README.md` for complete library documentation.

---

## References

- **Training Scripts:** See external/dnn-mode-connectivity for curve training
- **Analysis Scripts:** See `scripts/analysis/README.md` for post-evaluation analysis
- **Plotting Scripts:** See `scripts/plot/` for visualization
- **Library Documentation:** See `scripts/lib/README.md`

---

**Last Updated:** 2025-12-14
