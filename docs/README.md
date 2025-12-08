# Documentation

This folder contains all documentation for the Mode Connectivity research project.

## Quick Navigation

### Getting Started
- **[ORGANIZATION.md](ORGANIZATION.md)** - Complete project structure, file organization, and workflow
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guide for updated file paths after reorganization

### Neuron Swap Experiments
- **[NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md)** - Quick start guide for neuron swap experiments
- **[NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md)** - Detailed technical documentation
- **[NEURON_SWAP_RESULTS.md](NEURON_SWAP_RESULTS.md)** - How to view and interpret results

### Analysis Tools
- **[L2_DISTANCE_TRACKING.md](L2_DISTANCE_TRACKING.md)** - L2 distance calculation and interpretation
- **[PREDICTION_CHANGES_WORKFLOW.md](PREDICTION_CHANGES_WORKFLOW.md)** - Prediction change analysis workflow
- **[PREDICTION_CHANGES_QUICKSTART.md](PREDICTION_CHANGES_QUICKSTART.md)** - Quick start for prediction analysis

### Reference
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Experiment tracking and results log
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation overview

## Documentation by Topic

### Setup and Configuration
1. [ORGANIZATION.md](ORGANIZATION.md) - Start here for project structure
2. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Updated paths reference

### Running Experiments
1. [NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md) - Neuron swap quick start
2. [PREDICTION_CHANGES_QUICKSTART.md](PREDICTION_CHANGES_QUICKSTART.md) - Prediction analysis quick start
3. [EXPERIMENTS.md](EXPERIMENTS.md) - Experiment tracking

### Understanding Results
1. [NEURON_SWAP_RESULTS.md](NEURON_SWAP_RESULTS.md) - View neuron swap results
2. [L2_DISTANCE_TRACKING.md](L2_DISTANCE_TRACKING.md) - Understand L2 distances
3. [PREDICTION_CHANGES_WORKFLOW.md](PREDICTION_CHANGES_WORKFLOW.md) - Analyze prediction changes

### Technical Details
1. [NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md) - Neuron swap implementation
2. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Overall implementation

## Quick Reference

### Common Workflows

**Basic Mode Connectivity:**
```bash
# 1. Train endpoints
sbatch scripts/slurm/endpoints/submit_garipov.sh

# 2. Train curve
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_noreg.sh

# 3. Evaluate
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-seed1_noreg.sh
```

**Neuron Swap Experiments:**
See [NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md)

**Prediction Analysis:**
See [PREDICTION_CHANGES_QUICKSTART.md](PREDICTION_CHANGES_QUICKSTART.md)

### File Organization

All scripts and configs are organized by function:
- `scripts/slurm/endpoints/` - Endpoint training scripts
- `scripts/slurm/curves/` - Curve training scripts
- `scripts/slurm/evaluation/` - Evaluation scripts
- `scripts/slurm/neuronswap/` - Neuron swap scripts
- `configs/garipov/endpoints/` - Endpoint configs
- `configs/garipov/curves/` - Curve configs
- `configs/garipov/neuronswap/` - Neuron swap configs

See [ORGANIZATION.md](ORGANIZATION.md) for complete structure.

---

**For the main project README, see [../README.md](../README.md)**
