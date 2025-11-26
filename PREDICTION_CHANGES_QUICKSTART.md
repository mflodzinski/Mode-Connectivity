# Prediction Changes Analysis - Quick Start Guide

This guide shows you how to run the complete prediction changes analysis pipeline.

## Overview

The analysis has 4 steps:
1. **[CLUSTER]** Collect detailed predictions and features (requires GPU)
2. **[LOCAL]** Analyze which samples change predictions
3. **[LOCAL]** Create 2D UMAP visualization
4. **[LOCAL]** Generate animated GIF

## Quick Start (Recommended)

### Step 1: Run Evaluation on Cluster

**Upload scripts to cluster** (if not already there):
```bash
# From local machine
scp scripts/eval/eval_curve_detailed.py mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/scripts/eval/
scp scripts/slurm/submit_prediction_changes_eval.sh mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/scripts/slurm/
```

**Submit SLURM job:**
```bash
# SSH to cluster
ssh mlodzinski@login.daic.tudelft.nl

# Navigate to project
cd ~/Mode-Connectivity

# Submit job
sbatch scripts/slurm/submit_prediction_changes_eval.sh
```

**Monitor job:**
```bash
# Check job status
squeue -u mlodzinski

# Watch output in real-time (replace JOBID)
tail -f slurm_pred_changes_eval_JOBID.out
```

**Expected runtime:** ~30-45 minutes

### Step 2: Download Results

Once the job completes:
```bash
# From local machine
scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  results/vgg16/cifar10/curve/evaluations/
```

**File size:** ~500MB (compressed)

### Step 3: Run Local Analysis

Run all remaining steps locally:
```bash
# From local machine, in project root
./scripts/run_prediction_changes_analysis.sh
```

This single script will:
- ✅ Verify predictions_detailed.npz exists
- ✅ Run prediction change analysis
- ✅ Create 2D UMAP visualization
- ✅ Generate animated GIF

**Expected runtime:** ~15-30 minutes (mostly for UMAP computation and GIF generation)

### Step 4: View Results

```bash
# Open 2D visualization
open results/vgg16/cifar10/curve/figures/prediction_changes_2d.png

# Open animation
open results/vgg16/cifar10/curve/figures/prediction_evolution.gif

# Read summary statistics
cat results/vgg16/cifar10/curve/analysis/analysis_summary.txt
```

## Manual Step-by-Step (Alternative)

If you prefer to run each step manually:

### Step 1: Cluster Evaluation

```bash
# On cluster
cd ~/Mode-Connectivity

python scripts/eval/eval_curve_detailed.py \
  --curve_ckpt results/vgg16/cifar10/curve/checkpoints/checkpoint-150.pt \
  --output results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --dataset CIFAR10 \
  --data_path ./data \
  --model VGG16 \
  --use_test
```

### Step 2: Local Analysis

```bash
# On local machine
poetry run python scripts/analysis/analyze_prediction_changes.py \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/analysis/
```

### Step 3: Local Visualization

```bash
poetry run python scripts/plot/plot_prediction_changes.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --output results/vgg16/cifar10/curve/figures/ \
  --method umap
```

### Step 4: Local Animation

```bash
poetry run python scripts/plot/create_prediction_animation.py \
  --analysis results/vgg16/cifar10/curve/analysis/changing_samples_info.npz \
  --predictions results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  --output results/vgg16/cifar10/curve/figures/prediction_evolution.gif \
  --fps 5
```

## Troubleshooting

### Cluster Issues

**Job fails immediately:**
- Check if checkpoint exists: `ls results/vgg16/cifar10/curve/checkpoints/checkpoint-150.pt`
- Check SLURM output: `cat slurm_pred_changes_eval_*.err`

**Out of memory:**
- Reduce `--batch_size` in the script (default: 128)

**Timeout:**
- Increase time limit in SLURM script: `#SBATCH --time=04:00:00`

### Local Issues

**UMAP slow or hanging:**
- Normal on first run (computing neighbors)
- Consider using `--method pca` for faster results
- Reduce `--num_points` in eval script for fewer samples

**Animation creation too slow:**
- Use `--skip_frames 2` to render every other frame
- Reduce `--fps` for faster playback (fewer total frames needed)

**Not enough memory:**
- Close other applications
- Animation uses ~2GB RAM during generation

## Output Files

After successful completion:

```
results/vgg16/cifar10/curve/
├── evaluations/
│   └── predictions_detailed.npz          # ~500MB: Raw predictions & features
├── analysis/
│   ├── changing_samples_info.npz         # Analysis results
│   └── analysis_summary.txt              # Statistics summary
└── figures/
    ├── prediction_changes_2d.png         # Static 2D visualization
    ├── prediction_evolution.gif          # Animated prediction changes
    └── animation_frames/                 # Individual frames (optional)
```

## What to Look For

Once analysis is complete, examine:

1. **analysis_summary.txt** - Key statistics:
   - What % of samples change predictions?
   - Which classes are most unstable?
   - How many prediction changes per sample?

2. **prediction_changes_2d.png** - Spatial patterns:
   - Do changing samples cluster by class?
   - Are they at class boundaries?
   - Size indicates instability (larger = more changes)

3. **prediction_evolution.gif** - Temporal dynamics:
   - When do most changes occur (which t values)?
   - Do predictions flip back and forth?
   - Which classes get confused?

## Commands Cheat Sheet

```bash
# === CLUSTER ===
# Submit evaluation job
sbatch scripts/slurm/submit_prediction_changes_eval.sh

# Check job status
squeue -u mlodzinski

# Cancel job
scancel JOBID

# === LOCAL ===
# Download results
scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
  results/vgg16/cifar10/curve/evaluations/

# Run complete local pipeline
./scripts/run_prediction_changes_analysis.sh

# View results
open results/vgg16/cifar10/curve/figures/prediction_changes_2d.png
open results/vgg16/cifar10/curve/figures/prediction_evolution.gif
cat results/vgg16/cifar10/curve/analysis/analysis_summary.txt
```

## Need Help?

- **Full documentation:** See [PREDICTION_CHANGES_WORKFLOW.md](PREDICTION_CHANGES_WORKFLOW.md)
- **Scripts location:**
  - Cluster: `scripts/slurm/submit_prediction_changes_eval.sh`
  - Local: `scripts/run_prediction_changes_analysis.sh`
- **Individual scripts:**
  - `scripts/eval/eval_curve_detailed.py`
  - `scripts/analysis/analyze_prediction_changes.py`
  - `scripts/plot/plot_prediction_changes.py`
  - `scripts/plot/create_prediction_animation.py`
