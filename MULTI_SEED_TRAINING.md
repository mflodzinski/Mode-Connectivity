# Multi-Seed Bezier Curve Training

This guide explains how to train multiple Bezier curves with different random seeds and verify that stochasticity is working correctly.

## Sources of Stochasticity

When training with different seeds, the following will vary:
1. **Batch shuffling** - Different order of samples each epoch
2. **SGD mini-batch variance** - Different gradient estimates
3. **Data augmentations** - Different random crops and horizontal flips

## Training Multiple Curves

### Option 1: Using the Multi-Seed Script (Recommended)

Train multiple curves with a single command:

```bash
python scripts/train/run_multiple_curves.py \
    --config-name vgg16_curve_seed0-seed1_noreg \
    --seeds 0 42 123 \
    --base-output-dir results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed
```

**Arguments:**
- `--config-name`: Base config to use (without .yaml extension)
- `--seeds`: List of seeds to train with (default: 0 42 123)
- `--base-output-dir`: Base directory for outputs (will append `_seed0`, `_seed42`, etc.)

**Examples:**

```bash
# Train 3 curves with default seeds (0, 42, 123)
python scripts/train/run_multiple_curves.py \
    --config-name vgg16_curve_seed0-seed1_noreg

# Train 5 curves with custom seeds
python scripts/train/run_multiple_curves.py \
    --config-name vgg16_curve_seed0-seed1_noreg \
    --seeds 1 2 3 4 5 \
    --base-output-dir results/vgg16/cifar10/curve_multiseed_5runs

# Train curves between mirrored endpoints
python scripts/train/run_multiple_curves.py \
    --config-name vgg16_curve_seed0-mirror_noreg \
    --seeds 10 20 30
```

### Option 2: Manual Training

Train curves individually with different seeds:

```bash
# Run 1 with seed=0
python scripts/train/run_garipov_curve.py \
    --config-name=vgg16_curve_seed0-seed1_noreg \
    seed=0 \
    output_root=results/vgg16/cifar10/curve_seed0-seed1_noreg_seed0/checkpoints

# Run 2 with seed=42
python scripts/train/run_garipov_curve.py \
    --config-name=vgg16_curve_seed0-seed1_noreg \
    seed=42 \
    output_root=results/vgg16/cifar10/curve_seed0-seed1_noreg_seed42/checkpoints

# Run 3 with seed=123
python scripts/train/run_garipov_curve.py \
    --config-name=vgg16_curve_seed0-seed1_noreg \
    seed=123 \
    output_root=results/vgg16/cifar10/curve_seed0-seed1_noreg_seed123/checkpoints
```

## Verifying Results are Different

After training, verify that the curves are actually different:

```bash
python scripts/analysis/compare_curves.py \
    --checkpoint-dirs \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed0/checkpoints \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed42/checkpoints \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed123/checkpoints \
    --checkpoint-name checkpoint-200.pt
```

**Expected Output:**

```
✅ DIFFERENT: curve_seed0-seed1_noreg_multiseed_seed0 vs curve_seed0-seed1_noreg_multiseed_seed42
  Normalized L2: 1.234567e-03

✅ DIFFERENT: curve_seed0-seed1_noreg_multiseed_seed0 vs curve_seed0-seed1_noreg_multiseed_seed123
  Normalized L2: 2.345678e-03

✅ DIFFERENT: curve_seed0-seed1_noreg_multiseed_seed42 vs curve_seed0-seed1_noreg_multiseed_seed123
  Normalized L2: 1.987654e-03

✅ SUCCESS: All curves are different!
   This confirms that stochasticity is working correctly.
```

**Interpreting Results:**

- **Normalized L2 > 1e-6**: Curves are different ✅
- **Normalized L2 < 1e-6**: Curves are identical (seed not applied correctly) ⚠️

## Complete Workflow Example

```bash
# 1. Train multiple curves with different seeds
python scripts/train/run_multiple_curves.py \
    --config-name vgg16_curve_seed0-seed1_noreg \
    --seeds 0 42 123 \
    --base-output-dir results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed

# 2. Wait for training to complete...

# 3. Compare the resulting curves
python scripts/analysis/compare_curves.py \
    --checkpoint-dirs \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed0/checkpoints \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed42/checkpoints \
        results/vgg16/cifar10/curve_seed0-seed1_noreg_multiseed_seed123/checkpoints \
    --checkpoint-name checkpoint-200.pt
```

## Configuration Files

All config files now support the `seed` parameter:

- `configs/garipov/curves/vgg16_curve_seed0-seed1_reg.yaml` (seed: 0)
- `configs/garipov/curves/vgg16_curve_seed0-seed1_noreg.yaml` (seed: 0)
- `configs/garipov/curves/vgg16_curve_seed0-mirror_noreg.yaml` (seed: 0)
- `configs/garipov/curves/vgg16_curve_seed0-mirror_reg.yaml` (seed: 0)

You can override the seed using Hydra syntax: `seed=42`

## Technical Details

### Seed Control Implementation

The seed controls all sources of randomness:

1. **Python random**: `random.seed(seed)`
2. **NumPy**: `np.random.seed(seed)`
3. **PyTorch CPU**: `torch.manual_seed(seed)`
4. **PyTorch CUDA**: `torch.cuda.manual_seed_all(seed)`
5. **CUDNN**: `torch.backends.cudnn.deterministic = True`

This is implemented in:
- `src/utils.py::set_global_seed()`
- `scripts/train/run_garipov_curve.py` (reads seed from config)
- `external/dnn-mode-connectivity/train.py` (receives seed argument)

### Reproducibility

- **Same seed** → Identical results (bit-for-bit reproducibility)
- **Different seed** → Different results (stochastic variation)

All randomness is controlled by the seed parameter, ensuring reproducible experiments.
