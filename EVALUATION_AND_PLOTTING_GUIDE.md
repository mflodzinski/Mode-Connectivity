# Evaluation and Plotting Guide

## Overview
This guide provides commands to evaluate and plot the low-loss paths (curves, symmetry plane, polygon chain) versus linear interpolation for VGG16 on CIFAR10.

---

## 1. EVALUATION

### A. Linear Interpolation (Baseline)

Evaluate the direct linear path between two endpoints:

```bash
# seed0 ↔ seed0_mirrored
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/linear_seed0-mirror/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
  --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
  --num_points 61 \
  --use_test

# seed0 ↔ seed1
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/linear_seed0-seed1/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
  --init_end results/vgg16/cifar10/endpoints/checkpoints/seed1/checkpoint-200.pt \
  --num_points 61 \
  --use_test
```

### B. Bezier Curve (Garipov et al.)

Evaluate the optimized Bezier curve:

```bash
# seed0 ↔ seed0_mirrored (with regularization)
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/curve_seed0-mirror_reg/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve Bezier \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/curve_seed0-mirror_reg/checkpoints/checkpoint-200.pt \
  --num_points 61 \
  --use_test

# seed0 ↔ seed0_mirrored (no regularization)
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/curve_seed0-mirror_noreg/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve Bezier \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/curve_seed0-mirror_noreg/checkpoints/checkpoint-200.pt \
  --num_points 61 \
  --use_test

# seed0 ↔ seed1 (no regularization)
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve Bezier \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/curve_seed0-seed1_noreg/checkpoints/checkpoint-200.pt \
  --num_points 61 \
  --use_test
```

### C. Polygon Chain (PolyChain)

Evaluate the polygon chain (3-bend piecewise linear path):

```bash
# seed0 ↔ seed0_mirrored
poetry run python external/dnn-mode-connectivity/eval_curve.py \
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
```

### D. Symmetry Plane

Evaluate the symmetry plane constrained path:

```bash
# seed0 ↔ seed0_mirrored
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/symmetry_plane_seed0-mirror/checkpoint-150.pt \
  --num_points 61 \
  --use_test

# seed0 ↔ seed1
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/symmetry_plane_seed0-seed1/evaluations \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt results/vgg16/cifar10/symmetry_plane_seed0-seed1/checkpoint-150.pt \
  --num_points 61 \
  --use_test
```

---

## 2. PLOTTING

### A. Single Comparison: Linear vs Curve

Compare linear interpolation with a single curve:

```bash
# Bezier curve vs linear (seed0-mirror with regularization)
poetry run python scripts/plot/plot_connectivity.py \
  --linear results/vgg16/cifar10/linear_seed0-mirror/evaluations/curve.npz \
  --curve results/vgg16/cifar10/curve_seed0-mirror_reg/evaluations/curve.npz \
  --output results/vgg16/cifar10/figures/connectivity_seed0-mirror_reg.png \
  --title "Mode Connectivity: Linear vs Bezier Curve (seed0-mirror, regularized)"

# Bezier curve vs linear (seed0-mirror no regularization)
poetry run python scripts/plot/plot_connectivity.py \
  --linear results/vgg16/cifar10/linear_seed0-mirror/evaluations/curve.npz \
  --curve results/vgg16/cifar10/curve_seed0-mirror_noreg/evaluations/curve.npz \
  --output results/vgg16/cifar10/figures/connectivity_seed0-mirror_noreg.png \
  --title "Mode Connectivity: Linear vs Bezier Curve (seed0-mirror, no regularization)"
```

### B. Multi-Method Comparison

Compare linear, curve, polygon, and symmetry plane:

```bash
# Create multi-method comparison plot (seed0-mirror)
poetry run python scripts/plot/plot_symmetry_plane_comparison.py \
  --linear results/vgg16/cifar10/linear_seed0-mirror/evaluations/curve.npz \
  --curve results/vgg16/cifar10/curve_seed0-mirror_noreg/evaluations/curve.npz \
  --polygon results/vgg16/cifar10/polygon_seed0-mirror/evaluations/curve.npz \
  --symmetry results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/curve.npz \
  --output results/vgg16/cifar10/figures/all_methods_seed0-mirror.png \
  --title "Mode Connectivity: All Methods (seed0-mirror)"

# seed0-seed1 comparison
poetry run python scripts/plot/plot_symmetry_plane_comparison.py \
  --linear results/vgg16/cifar10/linear_seed0-seed1/evaluations/curve.npz \
  --curve results/vgg16/cifar10/curve_seed0-seed1_noreg/evaluations/curve.npz \
  --symmetry results/vgg16/cifar10/symmetry_plane_seed0-seed1/evaluations/curve.npz \
  --output results/vgg16/cifar10/figures/all_methods_seed0-seed1.png \
  --title "Mode Connectivity: All Methods (seed0-seed1)"
```

---

## 3. BATCH EVALUATION AND PLOTTING

### Run All Evaluations

```bash
# Create script to run all evaluations
cat > run_all_evaluations.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "Starting all evaluations..."

# Linear interpolations
echo "1/8: Linear seed0-mirror..."
poetry run python scripts/eval/eval_linear.py --config-name vgg16_linear_seed0-mirror

echo "2/8: Linear seed0-seed1..."
poetry run python scripts/eval/eval_linear.py --config-name vgg16_linear_seed0-seed1

# Bezier curves
echo "3/8: Bezier curve seed0-mirror (reg)..."
poetry run python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_seed0-mirror_reg

echo "4/8: Bezier curve seed0-mirror (noreg)..."
poetry run python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_seed0-mirror_noreg

echo "5/8: Bezier curve seed0-seed1 (noreg)..."
poetry run python scripts/eval/eval_garipov_curve.py --config-name vgg16_curve_seed0-seed1_noreg

# Polygon chains
echo "6/8: Polygon chain seed0-mirror..."
[manual command - needs wrapper script]

# Symmetry planes
echo "7/8: Symmetry plane seed0-mirror..."
[manual command - needs wrapper script]

echo "8/8: Symmetry plane seed0-seed1..."
[manual command - needs wrapper script]

echo "All evaluations complete!"
SCRIPT

chmod +x run_all_evaluations.sh
```

---

## 4. OUTPUT FILES

After running evaluations, you'll get:

### Evaluation Files (.npz)
Each evaluation creates a `curve.npz` file with:
- `ts`: t values (0 to 1)
- `tr_loss`, `tr_acc`, `tr_err`: Training metrics
- `te_loss`, `te_acc`, `te_err`: Test metrics
- `l2_norm`: L2 distance from previous point

Location: `results/vgg16/cifar10/{method}/evaluations/curve.npz`

### Plot Files (.png)
Plots showing:
1. Test Error vs t
2. Test Loss vs t
3. Train Error vs t
4. Train Loss vs t
5. L2 norm along path
6. L2 norm training evolution (if available)

Location: `results/vgg16/cifar10/figures/{method}_comparison.png`

---

## 5. EXPECTED RESULTS

### Typical Loss/Error Patterns:

**Linear Interpolation:**
- High loss barrier in the middle (~0.5)
- Test error peaks around 50-90%
- Forms a "mountain" shape

**Bezier Curve (Optimized):**
- Much lower loss throughout
- Test error stays near endpoint levels (~6-8%)
- Nearly flat profile

**Polygon Chain:**
- Similar to Bezier (unconstrained optimization)
- Slight kink at middle bend
- Low loss throughout

**Symmetry Plane:**
- Constrained to hyperplane
- May have slightly higher loss than unconstrained polygon
- Still much better than linear interpolation

---

## 6. QUICK START

For a quick comparison of seed0-mirror experiments:

```bash
# 1. Evaluate linear baseline (if not done)
mkdir -p results/vgg16/cifar10/linear_seed0-mirror/evaluations
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/linear_seed0-mirror/evaluations \
  --dataset CIFAR10 --data_path ./data --transform VGG --model VGG16 \
  --init_start results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
  --init_end results/vgg16/cifar10/endpoints/checkpoints/seed0_mirrored/checkpoint-200.pt \
  --num_points 61 --use_test

# 2. Evaluate polygon chain
mkdir -p results/vgg16/cifar10/polygon_seed0-mirror/evaluations
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/polygon_seed0-mirror/evaluations \
  --dataset CIFAR10 --data_path ./data --transform VGG --model VGG16 \
  --curve PolyChain --num_bends 3 \
  --ckpt results/vgg16/cifar10/polygon_seed0-mirror/checkpoint-150.pt \
  --num_points 61 --use_test

# 3. Evaluate symmetry plane
mkdir -p results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations
poetry run python external/dnn-mode-connectivity/eval_curve.py \
  --dir results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations \
  --dataset CIFAR10 --data_path ./data --transform VGG --model VGG16 \
  --curve PolyChain --num_bends 3 \
  --ckpt results/vgg16/cifar10/symmetry_plane_seed0-mirror/checkpoint-150.pt \
  --num_points 61 --use_test

# 4. Plot comparison (if plot script exists)
mkdir -p results/vgg16/cifar10/figures
poetry run python scripts/plot/plot_connectivity.py \
  --linear results/vgg16/cifar10/linear_seed0-mirror/evaluations/curve.npz \
  --curve results/vgg16/cifar10/polygon_seed0-mirror/evaluations/curve.npz \
  --output results/vgg16/cifar10/figures/polygon_vs_linear_seed0-mirror.png \
  --title "Polygon Chain vs Linear Interpolation"
```

---

## 7. TROUBLESHOOTING

### Issue: "No module named 'data'"
**Solution:** Make sure you're running from the project root and PYTHONPATH includes `external/dnn-mode-connectivity`

### Issue: "Checkpoint not found"
**Solution:** Check that training completed and checkpoint exists at the specified path

### Issue: "CUDA out of memory"
**Solution:** Reduce `--batch_size` (default is 128, try 64 or 32)

### Issue: "Plot script not found"
**Solution:** Some plot scripts may need to be created. The base `plot_connectivity.py` should exist.

---

## 8. NOTES

- Evaluation is done on the **test set** (use `--use_test` flag)
- Default: 61 evaluation points along the path (t ∈ [0, 1])
- Results are saved as `.npz` files (NumPy compressed arrays)
- Plots are generated as `.png` files
- All commands use `poetry run` to ensure correct virtual environment

