# Analysis Scripts

Scripts for analyzing mode connectivity experiments, neural network transformations, and prediction patterns.

## Overview

This directory contains 4 consolidated analysis scripts that cover various aspects of mode connectivity research:

1. **compare_checkpoints.py** - Compare two independently trained models
2. **analyze_predictions.py** - Analyze prediction changes along curves
3. **network_transform.py** - Transform networks via mirroring or neuron swapping
4. **analyze_curve.py** - Analyze Bezier curve properties and comparisons

All scripts use the unified library at `scripts/lib/`.

---

## Scripts Reference

### 1. compare_checkpoints.py

Compare predictions and per-class performance between two independently trained models.

**Purpose:**
- Analyze whether two models make different predictions
- Compare per-class accuracies
- Measure prediction agreement/disagreement

**Basic Usage:**
```bash
python compare_checkpoints.py \
    --checkpoint1 path/to/checkpoint1.pt \
    --checkpoint2 path/to/checkpoint2.pt \
    --output-dir results/comparison
```

**Common Arguments:**
```bash
--checkpoint1 PATH         # First checkpoint to compare
--checkpoint2 PATH         # Second checkpoint to compare
--output-dir DIR          # Output directory for results
--model VGG16             # Model architecture (default: VGG16)
--dataset CIFAR10         # Dataset (default: CIFAR10)
--data-path ./data        # Path to dataset
--batch-size 128          # Batch size (default: 128)
--use-bn                  # Use batch normalization variant
```

**Example:**
```bash
python compare_checkpoints.py \
    --checkpoint1 checkpoints/seed0/model.pt \
    --checkpoint2 checkpoints/seed1/model.pt \
    --output-dir results/seed_comparison \
    --model VGG16 \
    --use-bn
```

**Outputs:**
- `comparison_results.json` - Accuracy, agreement statistics
- `per_class_comparison.png` - Per-class accuracy comparison plot
- `confusion_matrices.png` - Side-by-side confusion matrices
- `agreement_analysis.png` - Agreement breakdown visualization

---

### 2. analyze_predictions.py

Analyze prediction changes and patterns along mode connectivity curves.

**Modes:**
- `analyze` - Analyze prediction changes along curve
- `compare-endpoints` - Compare endpoint accuracies
- `save-images` - Save images of unstable samples
- `all` - Run all analyses

**Basic Usage:**
```bash
python analyze_predictions.py \
    --mode analyze \
    --predictions path/to/predictions.npz \
    --output-dir results/predictions
```

**Mode: analyze**
Identify samples that change predictions along the curve.

```bash
python analyze_predictions.py \
    --mode analyze \
    --predictions curve_predictions.npz \
    --output-dir results/analysis
```

**Mode: compare-endpoints**
Compare per-class accuracy between curve endpoints.

```bash
python analyze_predictions.py \
    --mode compare-endpoints \
    --predictions curve_predictions.npz \
    --output-dir results/endpoints
```

**Mode: save-images**
Save images of the most unstable samples (samples that change predictions most often).

```bash
python analyze_predictions.py \
    --mode save-images \
    --predictions curve_predictions.npz \
    --output-dir results/unstable_images \
    --dataset-path ./data \
    --top-k 20
```

**Mode: all**
Run complete analysis and generate comprehensive report.

```bash
python analyze_predictions.py \
    --mode all \
    --predictions curve_predictions.npz \
    --output-dir results/full_analysis \
    --dataset-path ./data
```

**Outputs:**
- `prediction_analysis.json` - Prediction change statistics
- `prediction_analysis_detailed.npz` - Detailed data arrays
- `unstable_samples/` - Images of unstable samples (if mode includes save-images)

---

### 3. network_transform.py

Transform neural networks via mirroring or neuron swapping.

**Modes:**
- `mirror` - Create mirrored network via reverse permutation
- `swap` - Swap specific neurons for minimal perturbation

**Mode: mirror**

Create a functionally equivalent network by reversing neuron order.

```bash
python network_transform.py \
    --mode mirror \
    --checkpoint path/to/model.pt \
    --output path/to/mirrored_model.pt \
    --model VGG16 \
    --use-bn
```

**With verification:**
```bash
python network_transform.py \
    --mode mirror \
    --checkpoint checkpoints/seed0.pt \
    --output checkpoints/seed0_mirror.pt \
    --model VGG16 \
    --use-bn \
    --verify \
    --full-dataset-verify \
    --dataset CIFAR10 \
    --data-path ./data
```

**Mode: swap**

Swap specific neurons to create minimally perturbed network.

```bash
python network_transform.py \
    --mode swap \
    --checkpoint path/to/model.pt \
    --output path/to/swapped_model.pt \
    --depth 2 \
    --max-swaps 10 \
    --model VGG16
```

**Arguments:**
```bash
# Common
--checkpoint PATH         # Input checkpoint
--output PATH            # Output checkpoint path
--model VGG16            # Model architecture
--use-bn                 # Use batch normalization

# Mirror mode
--verify                 # Verify functional equivalence
--full-dataset-verify    # Verify on full dataset (slower but thorough)

# Swap mode
--depth 0-4              # Layer depth to swap (0=early, 4=late)
--max-swaps N            # Maximum number of neuron swaps
--reference PATH         # Reference checkpoint for distance calculation
--verify                 # Verify equivalence after swapping
```

**Example - Create mirror and verify:**
```bash
python network_transform.py \
    --mode mirror \
    --checkpoint checkpoints/trained_model.pt \
    --output checkpoints/mirrored_model.pt \
    --model VGG16 \
    --use-bn \
    --verify \
    --full-dataset-verify \
    --dataset CIFAR10 \
    --data-path ./data
```

**Example - Swap neurons at mid-depth:**
```bash
python network_transform.py \
    --mode swap \
    --checkpoint checkpoints/seed0.pt \
    --output checkpoints/seed0_swapped.pt \
    --depth 2 \
    --max-swaps 5 \
    --reference checkpoints/seed1.pt \
    --model VGG16 \
    --verify
```

**Outputs:**
- Transformed checkpoint at specified output path
- Console output with verification results (if --verify used)

---

### 4. analyze_curve.py

Analyze Bezier curve properties, symmetry, and comparisons.

**Modes:**
- `layer-distances` - Analyze layer-wise distances along curve
- `symmetry` - Verify symmetry plane constraints
- `compare-seeds` - Compare curves from different random seeds
- `compare-inits` - Compare different initialization methods
- `checkpoint-distance` - Calculate L2 distance between two checkpoints

**Mode: layer-distances**

Analyze how distance changes across layers along the curve.

```bash
python analyze_curve.py \
    --mode layer-distances \
    --curve path/to/curve.pt \
    --endpoint0 path/to/endpoint0.pt \
    --endpoint1 path/to/endpoint1.pt \
    --output results/layer_distances \
    --num-points 61
```

**With neuron swap metadata:**
```bash
python analyze_curve.py \
    --mode layer-distances \
    --curve curve_swapped.pt \
    --endpoint0 seed0.pt \
    --endpoint1 seed0_swapped.pt \
    --output results/swap_distances \
    --swap-metadata swap_metadata.json \
    --permutation-invariant
```

**Mode: symmetry**

Verify that a curve satisfies symmetry plane constraints.

```bash
python analyze_curve.py \
    --mode symmetry \
    --curve path/to/curve.pt \
    --endpoint0 path/to/w1.pt \
    --endpoint1 path/to/w2.pt \
    --theta path/to/theta.pt \
    --output results/symmetry_verification
```

**Mode: compare-seeds**

Compare curves trained with different random seeds.

```bash
python analyze_curve.py \
    --mode compare-seeds \
    --curve1 curve_seed0.pt \
    --curve2 curve_seed1.pt \
    --output results/seed_comparison
```

**Mode: compare-inits**

Compare curves with different initialization methods.

```bash
python analyze_curve.py \
    --mode compare-inits \
    --curve path/to/curve.pt \
    --endpoint0 path/to/w1.pt \
    --endpoint1 path/to/w2.pt \
    --output results/init_comparison
```

**Mode: checkpoint-distance**

Calculate simple L2 distance between two checkpoints.

```bash
python analyze_curve.py \
    --mode checkpoint-distance \
    --checkpoint1 path/to/model1.pt \
    --checkpoint2 path/to/model2.pt
```

**Common Arguments:**
```bash
--curve PATH              # Curve checkpoint
--endpoint0 PATH          # First endpoint checkpoint
--endpoint1 PATH          # Second endpoint checkpoint
--output DIR              # Output directory
--num-points N            # Number of points along curve (default: 61)
--permutation-invariant   # Use permutation-invariant distance
```

**Outputs:**
- `layer_distances.json` - Layer-wise distance statistics
- `layer_distances_plot.png` - Visualization of distances
- `symmetry_verification.json` - Symmetry constraint verification
- `curve_comparison.json` - Comparison statistics

---

## Common Patterns

### Typical Workflow

1. **Train models** (using training scripts, not in this directory)

2. **Compare trained models:**
   ```bash
   python compare_checkpoints.py \
       --checkpoint1 seed0/model.pt \
       --checkpoint2 seed1/model.pt \
       --output-dir results/comparison
   ```

3. **Create mirrored network:**
   ```bash
   python network_transform.py \
       --mode mirror \
       --checkpoint seed0/model.pt \
       --output seed0_mirror.pt \
       --verify
   ```

4. **Train curve** (using training scripts)

5. **Analyze curve properties:**
   ```bash
   python analyze_curve.py \
       --mode layer-distances \
       --curve curve.pt \
       --endpoint0 seed0.pt \
       --endpoint1 seed1.pt \
       --output results/curve_analysis
   ```

6. **Analyze predictions along curve:**
   ```bash
   python analyze_predictions.py \
       --mode all \
       --predictions curve_predictions.npz \
       --output-dir results/predictions
   ```

---

## Common Arguments

Most scripts share these common arguments:

```bash
# Model arguments
--model VGG16              # Architecture (VGG16, VGG19, ResNet18, etc.)
--num-classes 10           # Number of output classes (default: 10)
--use-bn                   # Use batch normalization variant

# Data arguments
--dataset CIFAR10          # Dataset (default: CIFAR10)
--data-path ./data         # Path to dataset directory
--batch-size 128           # Batch size (default: 128)
--num-workers 4            # DataLoader workers (default: 4)

# Device arguments
# (Automatically detected - CUDA > MPS > CPU)
```

---

## Output Formats

### JSON Files
Analysis results saved as JSON with statistics and metrics.

**Example:**
```json
{
  "accuracy_model1": 92.5,
  "accuracy_model2": 91.8,
  "agreement_rate": 95.2,
  "per_class_accuracies": [...]
}
```

### NPZ Files
Detailed numerical data saved as NumPy arrays.

**Example:**
```python
import numpy as np
data = np.load('results.npz')
predictions = data['predictions']  # Shape: (num_points, num_samples)
targets = data['targets']          # Shape: (num_samples,)
```

### Plots
Visualizations saved as PNG images with 300 DPI.

---

## Tips

### Performance
- Use `--batch-size 256` for faster evaluation on GPU
- Reduce `--num-workers` if running out of memory
- Use `--num-points 25` for quick curve analysis (default: 61)

### Verification
- Always use `--verify` when creating transformed networks
- Use `--full-dataset-verify` for thorough equivalence checking
- Compare checkpoints before and after transformations

### Output Organization
Create organized output directory structure:
```
results/
├── comparison/
│   ├── seed0_vs_seed1/
│   └── seed0_vs_mirror/
├── curves/
│   ├── layer_distances/
│   └── symmetry/
└── predictions/
    ├── analysis/
    └── unstable_samples/
```

---

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'lib'`:
- Ensure you're running from the `scripts/analysis/` directory
- Check that `scripts/lib/` exists

### CUDA Out of Memory
- Reduce `--batch-size` (try 64 or 32)
- Reduce `--num-workers`
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

### Checkpoint Loading Errors
- Verify checkpoint paths are correct
- Ensure checkpoints match the specified `--model` architecture
- Check that `--num-classes` matches checkpoint

---

## Library Dependency

All analysis scripts depend on the unified library at `scripts/lib/`.

Import pattern:
```python
import sys
sys.path.insert(0, '../lib')

from lib.core import checkpoint, data, models
from lib.evaluation import evaluate, metrics
from lib.analysis import plotting, prediction_analyzer
from lib.transform import mirror, neuron_swap
from lib.curves import analyzer
```

See `scripts/lib/README.md` for complete library documentation.

---

## References

- **Consolidation Details:** See `scripts/lib/CONSOLIDATION_SUMMARY.md`
- **Import Fixes:** See `scripts/lib/IMPORT_FIXES.md`
- **Validation Report:** See `scripts/lib/VALIDATION_REPORT.md`
- **Library Documentation:** See `scripts/lib/README.md`

---

**Last Updated:** 2025-12-14
