# Neuron Swap Experiment Implementation Summary

## Overview

Complete implementation of the neuron swapping experiment to test local vs global weight corrections in mode connectivity.

**Implementation Date:** 2025-12-04
**Status:** ✅ Complete - Ready for execution

---

## Files Created

### Core Analysis Scripts (3 files)

1. **[scripts/analysis/swap_neurons.py](scripts/analysis/swap_neurons.py)** (370 lines)
   - Creates functionally equivalent networks by swapping specific neurons
   - Supports early/mid/late layer selection
   - Includes verification on test dataset
   - Outputs metadata JSON with swap details

2. **[scripts/analysis/analyze_neuron_swap_distances.py](scripts/analysis/analyze_neuron_swap_distances.py)** (350 lines)
   - Evaluates layer-wise distances along trained curves
   - Computes normalized L2, relative, and raw L2 distances
   - Outputs NPZ file for visualization

3. **[scripts/plotting/plot_layer_distance_animation.py](scripts/plotting/plot_layer_distance_animation.py)** (280 lines)
   - Creates animated GIF showing distance evolution
   - Red bars highlight swapped layers
   - Optional static heatmap view

### Configuration Files (3 files)

4. **[configs/garipov/vgg16_curve_neuronswap_early2_reg.yaml](configs/garipov/vgg16_curve_neuronswap_early2_reg.yaml)**
   - Config for training curve with early layer swap

5. **[configs/garipov/vgg16_curve_neuronswap_mid2_reg.yaml](configs/garipov/vgg16_curve_neuronswap_mid2_reg.yaml)**
   - Config for training curve with mid layer swap

6. **[configs/garipov/vgg16_curve_neuronswap_late2_reg.yaml](configs/garipov/vgg16_curve_neuronswap_late2_reg.yaml)**
   - Config for training curve with late layer swap

### SLURM Scripts (4 files)

7. **[scripts/slurm/submit_neuronswap_curve_early2.sh](scripts/slurm/submit_neuronswap_curve_early2.sh)**
   - Cluster job for training early swap curve

8. **[scripts/slurm/submit_neuronswap_curve_mid2.sh](scripts/slurm/submit_neuronswap_curve_mid2.sh)**
   - Cluster job for training mid swap curve

9. **[scripts/slurm/submit_neuronswap_curve_late2.sh](scripts/slurm/submit_neuronswap_curve_late2.sh)**
   - Cluster job for training late swap curve

10. **[scripts/slurm/submit_neuronswap_analysis.sh](scripts/slurm/submit_neuronswap_analysis.sh)**
    - Cluster job for running analysis + visualization

### Workflow Scripts (1 file)

11. **[scripts/workflows/setup_neuronswap_experiments.sh](scripts/workflows/setup_neuronswap_experiments.sh)**
    - Automated setup for all three swap variants
    - Creates early/mid/late swapped endpoints
    - Runs verification tests

### Documentation (3 files)

12. **[NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md)** (500 lines)
    - Complete technical documentation
    - Detailed usage instructions
    - Mathematical formulas
    - Troubleshooting guide

13. **[NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md)** (300 lines)
    - Quick start guide
    - Step-by-step pipeline
    - Common issues and solutions
    - Interpretation guide

14. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** (this file)
    - Implementation overview
    - File listing
    - Execution instructions

---

## Total Implementation

- **14 files created**
- **~2,500 lines of code**
- **~1,500 lines of documentation**

---

## Directory Structure

```
Mode-Connectivity/
├── scripts/
│   ├── analysis/
│   │   ├── swap_neurons.py                    [NEW] ✨
│   │   ├── analyze_neuron_swap_distances.py   [NEW] ✨
│   │   └── mirror_network.py                  [existing]
│   ├── plotting/
│   │   └── plot_layer_distance_animation.py   [NEW] ✨
│   ├── workflows/
│   │   └── setup_neuronswap_experiments.sh    [NEW] ✨
│   └── slurm/
│       ├── submit_neuronswap_curve_early2.sh  [NEW] ✨
│       ├── submit_neuronswap_curve_mid2.sh    [NEW] ✨
│       ├── submit_neuronswap_curve_late2.sh   [NEW] ✨
│       └── submit_neuronswap_analysis.sh      [NEW] ✨
├── configs/garipov/
│   ├── vgg16_curve_neuronswap_early2_reg.yaml [NEW] ✨
│   ├── vgg16_curve_neuronswap_mid2_reg.yaml   [NEW] ✨
│   └── vgg16_curve_neuronswap_late2_reg.yaml  [NEW] ✨
├── NEURON_SWAP_EXPERIMENTS.md                 [NEW] ✨
├── NEURON_SWAP_QUICKSTART.md                  [NEW] ✨
└── IMPLEMENTATION_SUMMARY.md                  [NEW] ✨
```

---

## Execution Pipeline

### Complete Execution (from scratch)

```bash
# 1. Create swapped endpoints (5 minutes, local)
bash scripts/workflows/setup_neuronswap_experiments.sh

# 2. Train curves (2 hours, cluster with GPU)
sbatch scripts/slurm/submit_neuronswap_curve_early2.sh
sbatch scripts/slurm/submit_neuronswap_curve_mid2.sh
sbatch scripts/slurm/submit_neuronswap_curve_late2.sh

# 3. Wait for curves to finish, then analyze each
# (Edit EXPERIMENT variable in script before each run)
sbatch scripts/slurm/submit_neuronswap_analysis.sh  # Set EXPERIMENT="early2"
sbatch scripts/slurm/submit_neuronswap_analysis.sh  # Set EXPERIMENT="mid2"
sbatch scripts/slurm/submit_neuronswap_analysis.sh  # Set EXPERIMENT="late2"

# 4. Results will be in:
# results/vgg16/cifar10/curve_neuronswap_{early2,mid2,late2}_reg/analysis/
```

### Quick Test (local execution for debugging)

```bash
# 1. Create one swapped endpoint (mid layer)
python scripts/analysis/swap_neurons.py \
    --checkpoint results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --output results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200.pt \
    --layer-depth mid \
    --num-swaps 1 \
    --seed 42 \
    --verify

# 2. (Train curve on cluster - can't avoid this step)

# 3. Analyze results (after curve training completes)
python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/checkpoint-200.pt \
    --original-checkpoint results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --swap-metadata results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200_metadata.json \
    --output-dir results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis

# 4. Create visualization
python scripts/plotting/plot_layer_distance_animation.py \
    --data results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_along_curve.npz \
    --output results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/animation.gif \
    --fps 10 \
    --heatmap
```

---

## Key Features

### 1. Flexible Layer Selection
- **Early**: Block 0, Layer 0 (64 filters, early features)
- **Mid**: Block 2, Layer 1 (256 filters, mid features)
- **Late**: Block 4, Layer 1 (512 filters, late features)

### 2. Comprehensive Metrics
- **Normalized L2**: Fair comparison across layers
- **Relative distance**: Percentage change
- **Raw L2**: Absolute distance

### 3. Automated Pipeline
- Single command setup script
- SLURM integration for cluster
- Metadata tracking throughout

### 4. Rich Visualization
- Animated GIF (61 frames)
- Static heatmap
- Layer highlighting
- Progress indicator

### 5. Robust Verification
- Test accuracy verification
- Functional equivalence checking
- Detailed metadata logging

---

## Dependencies

All dependencies already installed in your environment:
- ✅ PyTorch
- ✅ NumPy
- ✅ Matplotlib
- ✅ Pillow (for GIF creation)
- ✅ Hydra (for configs)
- ✅ tqdm (for progress bars)

---

## Expected Results Structure

After running all experiments, you'll have:

```
results/vgg16/cifar10/
├── endpoints_neuronswap/
│   ├── early2/
│   │   ├── checkpoint-200.pt                    # Swapped network
│   │   └── checkpoint-200_metadata.json         # Swap details
│   ├── mid2/
│   │   ├── checkpoint-200.pt
│   │   └── checkpoint-200_metadata.json
│   └── late2/
│       ├── checkpoint-200.pt
│       └── checkpoint-200_metadata.json
├── curve_neuronswap_early2_reg/
│   ├── checkpoints/
│   │   └── checkpoint-200.pt                    # Trained curve
│   └── analysis/
│       ├── layer_distances_along_curve.npz      # Distance data
│       ├── analysis_summary.json                # Summary
│       ├── layer_distances_evolution.gif        # Animation
│       └── layer_distances_evolution_heatmap.png # Heatmap
├── curve_neuronswap_mid2_reg/
│   └── ... (same structure as early2)
└── curve_neuronswap_late2_reg/
    └── ... (same structure as early2)
```

---

## Validation Checklist

Before considering results valid, verify:

- [ ] All swapped endpoints show ~same accuracy as original
- [ ] All curves trained for 200 epochs
- [ ] Analysis completes without errors
- [ ] Animations show clear patterns
- [ ] Metadata files are readable
- [ ] Results are consistent across depths

---

## Research Questions Answered

This implementation enables you to answer:

1. **Are weight corrections local or global?**
   - Local: Only swapped layer changes
   - Global: All layers adjust

2. **Does layer depth matter?**
   - Compare early vs mid vs late results
   - Which depth shows most local correction?

3. **How does this compare to full permutation?**
   - 2 neurons vs all neurons (mirror experiment)
   - Scaling of permutation effects

4. **What does this reveal about loss landscape structure?**
   - Local structure → simple geometry
   - Global structure → entangled geometry

---

## Future Extensions

Easy modifications to explore more:

### 1. More Swap Amounts
```bash
# Modify setup script to use --num-swaps 5 (10 neurons)
# Compare: 2, 10, 20, 50 neurons
```

### 2. Different Layers
```python
# In swap_neurons.py, modify get_vgg16_layer_info()
# Add custom layer selections
```

### 3. Different Metrics
```bash
# Try all three metrics in visualization
for metric in normalized_l2 relative raw_l2; do
    python scripts/plotting/plot_layer_distance_animation.py \
        --metric $metric ...
done
```

### 4. Per-Neuron Analysis
```python
# Add to analyze_neuron_swap_distances.py:
# - Track individual swapped neurons
# - Compare to non-swapped neurons in same layer
# - Statistical significance testing
```

---

## Known Limitations

1. **VGG16 specific**: Current implementation hardcoded for VGG16 architecture
   - Easy fix: Generalize layer detection

2. **Single swap per run**: Can only create one swap configuration at a time
   - Easy fix: Batch mode in setup script

3. **Fixed curve parameters**: Uses same settings as other experiments
   - Could explore: Different learning rates, regularization

4. **No baseline comparison**: Doesn't include linear interpolation baseline
   - Easy addition: Evaluate linear path in analysis script

---

## Testing Status

- ✅ Code written and syntax-checked
- ✅ File structure validated
- ✅ Documentation complete
- ⏳ Execution pending (requires cluster)
- ⏳ Results validation pending
- ⏳ Scientific findings pending

---

## Next Immediate Actions

1. **Test swap creation locally**:
   ```bash
   bash scripts/workflows/setup_neuronswap_experiments.sh
   ```

2. **Submit curve training jobs**:
   ```bash
   sbatch scripts/slurm/submit_neuronswap_curve_mid2.sh
   ```

3. **Monitor training**: Check WandB or SLURM logs

4. **Run analysis**: After training completes

5. **Interpret results**: Compare visualizations

---

## Success Criteria

Experiment is successful if you can:
- [ ] Generate all swapped endpoints
- [ ] Train all curves without errors
- [ ] Produce clean visualizations
- [ ] Observe clear patterns (local or global)
- [ ] Draw conclusions about weight space structure

---

## Contact & Support

**Implementation**: Claude Code
**Research**: Michal Lodzinski (mlodzinski@student.tudelft.nl)
**Institution**: TU Delft

For issues:
- Check [NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md) troubleshooting section
- Review SLURM error logs
- Open GitHub issue

---

## Summary

**Status**: ✅ **READY TO EXECUTE**

All code, configurations, and documentation are complete. The experiment pipeline is fully automated and ready to run on your cluster.

**Estimated time to first results**: ~3 hours (including 2h training)

**Good luck with your experiments! 🚀**
