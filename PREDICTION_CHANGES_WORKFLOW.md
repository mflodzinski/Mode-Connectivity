# Prediction Changes Analysis Workflow

This document describes the workflow for analyzing how predictions change along the Bezier curve between two trained models.

## Overview

The analysis consists of 4 steps:
1. **Collect detailed predictions** - Extract predictions at each point along the curve, plus features from endpoints
2. **Analyze changes** - Identify which samples change predictions and compute statistics
3. **Visualize in 2D** - Use UMAP to create 2D scatter plots of changing samples
4. **Create animation** - Generate GIF showing prediction evolution along the curve

## Prerequisites

Dependencies have been added to `pyproject.toml`:
- `umap-learn` - for dimensionality reduction
- `imageio` - for GIF creation

Install with:
```bash
poetry install
```

## Step 1: Collect Detailed Predictions

Run the detailed evaluation script to collect predictions and features at all points along the curve.

**Script:** `scripts/eval/eval_curve_detailed.py`

**Example command (local):**
```bash
poetry run python scripts/eval/eval_curve_detailed.py \
  --curve_ckpt results/vgg16/cifar10/curve/checkpoints/checkpoint-150.pt \
  --output results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --dataset CIFAR10 \
  --data_path ./data \
  --model VGG16 \
  --transform VGG \
  --curve Bezier \
  --num_bends 3 \
  --num_points 61 \
  --batch_size 128 \
  --num_workers 4 \
  --use_test
```

**Example command (cluster):**
```bash
# On cluster
srun --partition=general --qos=short --time=02:00:00 \
  --ntasks=1 --cpus-per-task=4 --gpus=1 --mem=16GB \
  python scripts/eval/eval_curve_detailed.py \
  --curve_ckpt results/vgg16/cifar10/curve/checkpoints/checkpoint-150.pt \
  --output results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --dataset CIFAR10 \
  --data_path ./data \
  --model VGG16 \
  --transform VGG \
  --curve Bezier \
  --num_bends 3 \
  --num_points 61 \
  --batch_size 128 \
  --num_workers 4 \
  --use_test
```

**Output:**
- `predictions_detailed.npz` containing:
  - `predictions`: [61, 10000] - predicted class at each t value
  - `targets`: [10000] - ground truth labels
  - `features_t0`: [10000, 512] - penultimate layer features at t=0
  - `features_t1`: [10000, 512] - penultimate layer features at t=1
  - `images`: [10000, 3, 32, 32] - CIFAR-10 images
  - `ts`: [61] - t values from 0.0 to 1.0

**Download from cluster:**
```bash
scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  results/vgg16/cifar10/curve/evaluations/
```

## Step 2: Analyze Prediction Changes

Identify samples that change predictions and compute statistics.

**Script:** `scripts/analysis/analyze_prediction_changes.py`

**Command:**
```bash
poetry run python scripts/analysis/analyze_prediction_changes.py \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/analysis/
```

**Output:**
- `changing_samples_info.npz` containing:
  - `indices`: indices of samples that change predictions
  - `change_counts`: number of changes per sample (all samples)
  - `changing_counts`: number of changes (only changing samples)
  - `predictions_changing`: prediction trajectories for changing samples
  - `targets_changing`: true labels for changing samples
  - `features_avg`: averaged endpoint features (all samples)
  - `features_avg_changing`: averaged features (changing samples only)
  - `features_t0_changing`: t=0 features for changing samples
  - `features_t1_changing`: t=1 features for changing samples
  - `images_changing`: images of changing samples
- `analysis_summary.txt`: text summary with statistics

## Step 3: Create 2D Visualization

Use UMAP to visualize changing samples in 2D space.

**Script:** `scripts/plot/plot_prediction_changes.py`

**Command:**
```bash
poetry run python scripts/plot/plot_prediction_changes.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --output results/vgg16/cifar10/curve/figures/ \
  --method umap \
  --random_state 42
```

**Options:**
- `--method`: `umap` (default) or `pca`
- `--random_state`: random seed for reproducibility
- `--plot_all`: also create comparison plot with stable samples

**Output:**
- `prediction_changes_2d.png`: scatter plot of changing samples
  - Color = true class label
  - Size = number of prediction changes
  - Larger points = more unstable samples

## Step 4: Create Animation

Generate animated GIF showing prediction evolution.

**Script:** `scripts/plot/create_prediction_animation.py`

**Command:**
```bash
poetry run python scripts/plot/create_prediction_animation.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/figures/prediction_evolution.gif \
  --method umap \
  --fps 5 \
  --random_state 42 \
  --skip_frames 1
```

**Options:**
- `--method`: `umap` (default) or `pca`
- `--fps`: frames per second (default: 5)
- `--skip_frames`: render every Nth frame (default: 1, use higher values for faster generation)
- `--random_state`: random seed for reproducibility

**Output:**
- `prediction_evolution.gif`: animated visualization showing:
  - Fixed 2D positions (UMAP embedding)
  - Color = current prediction at each t
  - Edge color: green = correct, red = incorrect
  - Large points = just changed prediction
  - Progress bar in title

**Note:** The script will ask if you want to delete individual frames after GIF creation. Individual frames are saved in `animation_frames/` subdirectory.

## Complete Workflow Example

```bash
# Step 1: Collect predictions (run on cluster or locally with GPU)
poetry run python scripts/eval/eval_curve_detailed.py \
  --curve_ckpt results/vgg16/cifar10/curve/checkpoints/checkpoint-150.pt \
  --output results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --dataset CIFAR10 \
  --data_path ./data \
  --model VGG16 \
  --use_test

# Step 2: Analyze changes (local)
poetry run python scripts/analysis/analyze_prediction_changes.py \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/analysis/

# Step 3: Create 2D visualization (local)
poetry run python scripts/plot/plot_prediction_changes.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --output results/vgg16/cifar10/curve/figures/

# Step 4: Create animation (local)
poetry run python scripts/plot/create_prediction_animation.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/figures/prediction_evolution.gif
```

## Feature Representation Choice

**The scripts use endpoint 0 (t=0) features for dimensionality reduction.**

**Why endpoint 0 features?**
- **No averaging artifacts**: Uses a real learned representation, not an artificial average
- **Consistent interpretation**: "Similarity according to the model at t=0"
- **Safe and simple**: Avoids potential issues if the two endpoints learned very different internal representations
- **Well-tested**: Endpoint 0 generalizes well (93% test accuracy)

**Alternative considered but rejected:**
- ❌ Averaged features `(features_t0 + features_t1) / 2`: Could create meaningless "frankenstein" features if endpoints use different representations
- ❌ Concatenated features: Doubles dimensionality unnecessarily
- ❌ Features from middle of curve: That model is overtrained (99.5% train, worse test)

## Expected Results

Based on the mode connectivity analysis:
- **~90% of samples** should remain stable (same prediction throughout)
- **~10% of samples** will change predictions somewhere along the curve
- Most changes should occur near **t=0.5** (middle of curve) where test error peaks
- Certain classes may be more prone to changes (e.g., cat/dog confusion)

## Key Insights to Look For

1. **Which classes are most unstable?** - Do certain CIFAR-10 classes change predictions more often?
2. **Where do changes occur?** - At what t values do most prediction changes happen?
3. **Spatial clustering** - Do changing samples cluster in feature space (according to endpoint 0)? Are they near decision boundaries?
4. **Prediction trajectories** - Do samples flip back and forth multiple times, or change once?
5. **Correctness patterns** - Are changing samples mostly incorrectly classified, or do correct predictions flip to incorrect?
6. **Feature space position** - Are unstable samples at the periphery or center of class clusters in endpoint 0's feature space?

## Files Created

### Scripts:
1. `scripts/eval/eval_curve_detailed.py` - Detailed curve evaluation
2. `scripts/analysis/analyze_prediction_changes.py` - Change analysis
3. `scripts/plot/plot_prediction_changes.py` - 2D visualization
4. `scripts/plot/create_prediction_animation.py` - GIF animation

### Output Structure:
```
results/vgg16/cifar10/curve/
├── evaluations/
│   ├── predictions_detailed.npz         # Raw predictions & features
│   ├── curve.npz                         # Standard evaluation metrics
│   └── linear.npz                        # Linear interpolation metrics
├── analysis/
│   ├── changing_samples_info.npz        # Analysis results
│   └── analysis_summary.txt             # Statistics summary
└── figures/
    ├── prediction_changes_2d.png        # 2D scatter plot
    ├── prediction_evolution.gif         # Animated visualization
    └── animation_frames/                # Individual frames (optional)
```

## Troubleshooting

**UMAP not available:**
- Falls back to PCA automatically
- Install UMAP: `poetry add umap-learn`

**Out of memory:**
- Reduce `--batch_size` in eval_curve_detailed.py
- Use `--skip_frames` in animation to reduce number of frames

**Slow UMAP:**
- Normal for first run (computing neighbors)
- Consider using PCA with `--method pca` for faster results

**Animation too slow/fast:**
- Adjust `--fps` parameter (higher = faster)
- Use `--skip_frames` to reduce total frames
