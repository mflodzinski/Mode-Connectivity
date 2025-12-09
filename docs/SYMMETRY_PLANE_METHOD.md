# Symmetry Plane Optimization Method

## Overview

This method finds an optimal point θ* on the symmetry plane between two trained neural networks, creating a two-segment linear path **w₁ → θ* → w₂** that minimizes the maximum loss along the path.

## Mathematical Formulation

### Symmetry Plane Definition

The symmetry plane between two weight configurations w₁ and w₂ is the hyperplane equidistant from both:

```
n · (θ - m) = 0
```

Where:
- **n = w₂ - w₁**: Normal vector (perpendicular to the plane)
- **m = (w₁ + w₂) / 2**: Midpoint (a point on the plane)
- **θ**: Any point on the symmetry plane

### Path Parametrization

The two-segment path consists of:
1. **Segment 1**: Linear interpolation from w₁ to θ*
   ```
   w(t) = (1 - t)w₁ + tθ*    for t ∈ [0, 0.5]
   ```

2. **Segment 2**: Linear interpolation from θ* to w₂
   ```
   w(t) = (1 - t)θ* + tw₂    for t ∈ [0.5, 1]
   ```

### Optimization Objective

Find θ* that minimizes the maximum loss along both segments:

```
θ* = argmin max(L(w₁ → θ), L(θ → w₂))
     θ on plane
```

Subject to the constraint: **n · (θ - m) = 0**

## Method: Projected Gradient Descent

### Algorithm

1. **Initialize** θ₀ on the symmetry plane (at midpoint or random location)

2. **For each optimization step**:

   a. **Evaluate path**: Compute loss at multiple points along both segments

   b. **Find worst point**: Identify the point with maximum loss

   c. **Compute gradient**: Backpropagate through the worst point
      ```
      ∇J(θ) = ∂L(w_worst)/∂θ
      ```

   d. **Unconstrained step**: Take gradient descent step
      ```
      θ' = θ - η∇J(θ)
      ```

   e. **Project to plane**: Remove component parallel to normal
      ```
      d_parallel = ((θ' - m)·n / ||n||²) × n
      θ_new = θ' - d_parallel
      ```

3. **Return** θ* with lowest maximum loss encountered

### Projection Formula

The projection ensures θ stays on the symmetry plane by removing the displacement component parallel to the normal vector:

```python
# Displacement from midpoint
displacement = θ' - m

# Dot product with normal
dot = displacement · n

# Parallel component magnitude
scale = dot / ||n||²

# Remove parallel component
θ_projected = θ' - scale × n
```

## Implementation

### File Structure

```
scripts/
├── train/
│   ├── train_symmetry_plane.py     # Core optimization algorithm
│   └── run_symmetry_plane.py       # Hydra wrapper
├── eval/
│   ├── eval_symmetry_plane.py      # Evaluate two-segment path
│   └── eval_symmetry_comparison.py # Compare with linear interpolation
├── plot/
│   └── plot_symmetry_plane_comparison.py  # Visualization
└── slurm/
    └── symmetry_plane/
        ├── submit_symplane_seed0-seed1.sh
        ├── submit_symplane_seed0-mirror.sh
        ├── submit_symplane_eval_seed0-seed1.sh
        └── submit_symplane_eval_seed0-mirror.sh

configs/garipov/symmetry_plane/
├── vgg16_symplane_seed0-seed1.yaml
└── vgg16_symplane_seed0-mirror.yaml
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization_steps` | 200 | Number of optimization iterations |
| `lr` | 0.015 | Learning rate (based on VGG16 curve training) |
| `momentum` | 0.9 | SGD momentum |
| `init_mode` | `"midpoint"` | Initialization: `"midpoint"` or `"random"` |
| `eval_points_per_segment` | 30 | Evaluation points per segment (60 total) |
| `batch_size` | 128 | Training batch size |

### Learning Rate Schedule

Same schedule as Bezier curve training (from Garipov et al.):
- **0-50% of training**: Full learning rate
- **50-90%**: Linear decay to 1% of initial LR
- **90-100%**: Constant at 1% of initial LR

## Usage

### 1. Train Symmetry Plane Optimization

**On cluster:**
```bash
sbatch scripts/slurm/symmetry_plane/submit_symplane_seed0-seed1.sh
```

**Locally:**
```bash
python scripts/train/run_symmetry_plane.py \
    --config-name vgg16_symplane_seed0-seed1
```

**Output:**
- `results/vgg16/cifar10/symmetry_plane_seed0-seed1/checkpoint_optimal.pt`
- `results/vgg16/cifar10/symmetry_plane_seed0-seed1/optimization_log.json`

### 2. Evaluate Results

**On cluster:**
```bash
sbatch scripts/slurm/symmetry_plane/submit_symplane_eval_seed0-seed1.sh
```

This runs:
1. `eval_symmetry_plane.py` - Evaluates two-segment path at 61 points
2. `eval_symmetry_comparison.py` - Compares with linear interpolation

**Output:**
- `evaluations/symmetry_plane.npz` - Path evaluation
- `evaluations/comparison.npz` - Comparison data

### 3. Visualize Results

**Locally (after downloading results):**
```bash
python scripts/plot/plot_symmetry_plane_comparison.py \
    --comparison-file results/vgg16/cifar10/symmetry_plane_seed0-seed1/evaluations/comparison.npz \
    --show
```

**Output:**
- 4-panel plot comparing linear interpolation vs. symmetry plane path
- Metrics: train loss, test loss, train error, test error

## Experiments

### Experiment 1: Independent Models (seed0 ↔ seed1)

**Research Question:** Can we find a better path than linear interpolation between independently trained networks?

**Setup:**
- Endpoint 1: VGG16 trained with seed 0
- Endpoint 2: VGG16 trained with seed 1
- L2 distance: ~1.5-3.0 (large)

**Expected Outcome:**
- Symmetry plane path should have lower barrier than linear interpolation
- Demonstrates non-trivial geometry in weight space

### Experiment 2: Permuted Network (seed0 ↔ mirror)

**Research Question:** When networks differ only by permutation, is the optimal path on the symmetry plane close to the midpoint?

**Setup:**
- Endpoint 1: VGG16 trained with seed 0
- Endpoint 2: Full neuron permutation of seed 0
- L2 distance: ~1.0-2.0 (large but functionally equivalent)

**Expected Outcome:**
- Optimal θ* may be very close to midpoint
- Lower barrier than linear interpolation
- Tests if permutation symmetry affects optimal path

## Comparison with Other Methods

| Method | Path Type | Parameters | Barrier Height | Training Cost |
|--------|-----------|------------|----------------|---------------|
| **Linear Interpolation** | Straight line | None | High | None |
| **Bezier Curve** | Smooth curve | 3 control points | Low | High (600 epochs) |
| **Symmetry Plane** | Two segments | 1 bend point | Medium | Medium (200 steps) |

### Advantages of Symmetry Plane Method

1. **Interpretability**: The bend point has geometric meaning (on symmetry plane)
2. **Efficiency**: Faster than full curve training (~200 steps vs 600 epochs)
3. **Simplicity**: Only 1 parameter to optimize (θ*), not full curve
4. **Theoretical grounding**: Motivated by geometric symmetry in weight space

### When to Use

- **Use Symmetry Plane** when:
  - You want geometric interpretation
  - Training time is limited
  - You're studying symmetry properties

- **Use Bezier Curve** when:
  - You need the smoothest, lowest-loss path
  - Training time is not a concern
  - You want ensemble predictions along curve

- **Use Linear Interpolation** when:
  - You only need a baseline
  - No GPU resources available

## Key Insights

### 1. Constraint Enforcement
The projection step is critical - without it, the optimization becomes unconstrained and θ drifts away from the symmetry plane.

### 2. Initialization Matters
- **Midpoint**: Stable, reproducible, natural starting point
- **Random**: Explores different regions of the plane, may find different local optima

### 3. Evaluation Points
Using 30 points per segment (60 total) provides good resolution for finding the maximum loss while being computationally feasible.

### 4. Learning Rate Schedule
Using the same schedule as curve training ensures comparable convergence properties.

## Troubleshooting

### Issue: Projection fails (assertion error)
**Cause**: Numerical precision issues with plane constraint
**Solution**: Increase tolerance in `verify_on_plane()` function

### Issue: Optimization doesn't improve over linear
**Cause**: Learning rate too high or initialization poor
**Solution**: Reduce learning rate or try different initialization

### Issue: Out of memory
**Cause**: Too many evaluation points or large batch size
**Solution**: Reduce `eval_points_per_segment` or `batch_size`

## References

1. Garipov, T., et al. (2018). "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs." *NeurIPS 2018.*
2. Draxler, F., et al. (2018). "Essentially No Barriers in Neural Network Energy Landscape." *ICML 2018.*
3. Fort, S., & Jastrzębski, S. (2019). "Large Scale Structure of Neural Network Loss Landscapes." *NeurIPS 2019.*

## Future Directions

1. **Multi-segment paths**: Optimize multiple points on the plane
2. **Adaptive evaluation**: Dynamically adjust evaluation points based on loss landscape
3. **Regularization**: Add regularization term to encourage θ* near midpoint
4. **Comparison with alignment methods**: Test against weight matching/permutation algorithms
