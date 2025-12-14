# Mode Connectivity Plotting Scripts

This directory contains visualization scripts for analyzing mode connectivity experiments. All scripts use shared utilities from `scripts/lib/` for consistent behavior and reduced code duplication.

## Available Scripts

### 1. plot_prediction_changes.py

Visualize how predictions change along the connectivity curve in 2D feature space.

**Usage:**
```bash
python plot_prediction_changes.py \
  --data path/to/predictions.npz \
  --output path/to/output.png \
  --method umap \
  --random-state 42 \
  --show
```

**Arguments:**
- `--data`: Path to predictions NPZ file (required)
- `--output`: Output path for plot (required)
- `--method`: Dimensionality reduction method: `umap` or `pca` (default: umap)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--show`: Display plot interactively

**Output:**
- PNG file with 2D visualization of prediction changes
- Shows how model predictions evolve along the connectivity path

---

### 2. create_prediction_animation.py

Create animated visualization of prediction changes along the connectivity curve.

**Usage:**
```bash
python create_prediction_animation.py \
  --data path/to/predictions.npz \
  --output path/to/animation.gif \
  --method umap \
  --fps 5 \
  --skip-frames 1 \
  --show
```

**Arguments:**
- `--data`: Path to predictions NPZ file (required)
- `--output`: Output path for GIF animation (required)
- `--method`: Dimensionality reduction method: `umap` or `pca` (default: umap)
- `--random-state`: Random seed (default: 42)
- `--fps`: Frames per second (default: 5)
- `--skip-frames`: Only render every Nth frame (default: 1)
- `--show`: Display animation interactively

**Output:**
- GIF animation showing prediction evolution over time
- Progress bar indicating position along curve

---

### 3. plot_class_accuracy_vs_t.py

Plot per-class accuracy evolution along the connectivity curve.

**Usage:**
```bash
python plot_class_accuracy_vs_t.py \
  --data path/to/class_accuracies.npz \
  --output path/to/output_dir \
  --show
```

**Arguments:**
- `--data`: Path to class accuracies NPZ file (required)
- `--output`: Output directory for plots (required)
- `--show`: Display plots interactively

**Output:**
- Individual plots for each class showing accuracy vs t
- All plots saved in specified output directory

---

### 4. plot_connectivity.py

Plot connectivity curves showing error/loss metrics along the path.

**Usage:**
```bash
python plot_connectivity.py \
  --data path/to/curve.npz \
  --output path/to/connectivity.png \
  --show
```

**Arguments:**
- `--data`: Path to curve evaluation NPZ file (required)
- `--output`: Output path for plot (optional, inferred from data path if not provided)
- `--show`: Display plot interactively

**Output:**
- 2x2 grid showing test error, test loss, train error, train loss along curve
- Summary text file with barrier statistics
- Vertical line at t=0.5 marking middle bend location

---

### 5. plot_symmetry_plane_comparison.py

Compare symmetry plane constraint vs linear interpolation.

**Usage:**
```bash
python plot_symmetry_plane_comparison.py \
  --symplane-file path/to/symplane_curve.npz \
  --linear-file path/to/linear.npz \
  --output path/to/comparison.png \
  --show
```

**Arguments:**
- `--symplane-file`: Path to symmetry plane curve NPZ file (required)
- `--linear-file`: Path to linear interpolation NPZ file (required)
- `--output`: Output path for plot (optional)
- `--show`: Display plot interactively

**Output:**
- 2x2 comparison plot of symmetry plane vs linear interpolation
- Summary statistics showing barrier reduction
- Text summary file with detailed metrics

---

### 6. plot_polygon_symmetry_comparison.py

Compare polygon chain, symmetry plane, and linear interpolation (all optional).

**Usage:**
```bash
python plot_polygon_symmetry_comparison.py \
  --polygon-file path/to/polygon_curve.npz \
  --symplane-file path/to/symplane_curve.npz \
  --linear-file path/to/linear.npz \
  --output path/to/comparison.png \
  --show
```

**Arguments:**
- `--polygon-file`: Path to polygon chain curve NPZ file (optional)
- `--symplane-file`: Path to symmetry plane curve NPZ file (optional)
- `--linear-file`: Path to linear interpolation NPZ file (optional)
- `--output`: Output path for plot (optional)
- `--show`: Display plot interactively

**Notes:**
- At least one curve file must be provided
- Plots any combination of the three curve types
- Compares constrained vs unconstrained approaches

**Output:**
- 2x2 grid comparing all provided curves
- Barrier reduction statistics
- Summary text file

---

### 7. plot_middle_point_l2_evolution.py

Plot L2 norm evolution of the middle point during curve training.

**Usage:**
```bash
python plot_middle_point_l2_evolution.py \
  --data path/to/middle_point_l2_norms.npz \
  --output path/to/l2_evolution.png \
  --title "Custom Title" \
  --dpi 300 \
  --show
```

**Arguments:**
- `--data`: Path to middle_point_l2_norms.npz file (required)
- `--output`: Output path for plot (optional, inferred from data path if not provided)
- `--title`: Custom title for the plot (optional)
- `--dpi`: DPI for saved figure (default: 300)
- `--show`: Display plot interactively

**Output:**
- Line plot showing L2 norm evolution over training epochs
- Shows both raw middle point and interpolated L2 norms (if available)
- Statistics box with initial, final, min, max values

---

### 8. plot_layer_distance_animation.py

Create animated visualization of layer-wise distance evolution along the curve.

**Usage:**
```bash
python plot_layer_distance_animation.py \
  --data path/to/layer_distances.npz \
  --output path/to/animation.gif \
  --fps 10 \
  --metric normalized_l2 \
  --heatmap \
  --show
```

**Arguments:**
- `--data`: Path to layer distances NPZ file (required)
- `--output`: Path to save animated GIF (required)
- `--metric`: Distance metric to plot: `normalized_l2`, `relative`, or `raw_l2` (default: normalized_l2)
- `--fps`: Frames per second (default: 10)
- `--skip-frames`: Only render every Nth frame (default: 1)
- `--heatmap`: Also create static heatmap view
- `--show`: Display animation interactively

**Output:**
- GIF animation showing layer-wise distances at each t value
- Optional: Static heatmap showing all layers across all t values
- Progress bar in animation
- Color-coded layers (red = swapped layer)

---

## Common Arguments

Most scripts support these standard arguments via `ArgumentParserBuilder`:

- `--output`: Output path for plot/figure
- `--show`: Display plot interactively (without blocking execution)
- `--method`: Dimensionality reduction method (`umap` or `pca`) - for prediction scripts
- `--random-state`: Random seed for reproducibility (default: 42)
- `--fps`: Animation frames per second (default: 5)
- `--skip-frames`: Skip frames in animation (default: 1)

## Input File Formats

All scripts expect NPZ files with specific structures:

- **predictions.npz**: Contains `ts`, `predictions`, `targets`, `features`
- **curve.npz**: Contains `ts`, `te_err`, `te_loss`, `tr_err`, `tr_loss`
- **class_accuracies.npz**: Contains `ts`, `class_accuracies`, `class_names`
- **layer_distances.npz**: Contains `layer_names`, `layer_types`, `t_values`, distance metrics
- **middle_point_l2_norms.npz**: Contains `epochs`, `l2_norms`, `interpolated_l2_norms`

## Shared Library Modules

These scripts use shared utilities from `scripts/lib/`:

- **lib/analysis/plotting.py**: Plotting utilities (save_figure, create_output_dir, etc.)
- **lib/analysis/dim_reduction.py**: UMAP/PCA dimensionality reduction with fallback
- **lib/utils/args.py**: Reusable argument parser components

## Example Workflow

```bash
# 1. Plot connectivity curve
python plot_connectivity.py \
  --data results/curve_evaluation.npz \
  --output figures/connectivity.png

# 2. Compare different approaches
python plot_polygon_symmetry_comparison.py \
  --polygon-file results/polygon_curve.npz \
  --symplane-file results/symplane_curve.npz \
  --linear-file results/linear.npz \
  --output figures/comparison.png

# 3. Visualize prediction changes
python plot_prediction_changes.py \
  --data results/predictions.npz \
  --output figures/predictions_2d.png \
  --method umap

# 4. Create animation
python create_prediction_animation.py \
  --data results/predictions.npz \
  --output figures/predictions.gif \
  --fps 5 \
  --method umap

# 5. Plot L2 norm evolution
python plot_middle_point_l2_evolution.py \
  --data results/middle_point_l2_norms.npz \
  --output figures/l2_evolution.png

# 6. Create layer distance animation
python plot_layer_distance_animation.py \
  --data results/layer_distances.npz \
  --output figures/layer_animation.gif \
  --fps 10 \
  --heatmap
```

## Output Format

All plots are saved with:
- **DPI**: 300 (high resolution)
- **Format**: PNG for static plots, GIF for animations
- **Bbox**: 'tight' (minimal whitespace)
- **Directory creation**: Automatic (parent directories created as needed)
