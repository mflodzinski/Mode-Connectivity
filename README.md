# Mode Connectivity Research

Implementation of mode connectivity experiments for neural networks, exploring how different networks in weight space can be connected via low-loss paths. Based on [DNN Mode Connectivity](https://github.com/timgaripov/dnn-mode-connectivity) (Garipov et al., 2018).

## Overview

This project investigates the loss landscape geometry of neural networks by training Bezier curves that connect different network configurations in weight space. Key research questions include:

- Can independently trained networks be connected via low-loss paths?
- How does weight space distance relate to functional similarity?
- Are connectivity corrections local or global when networks differ by permutations?
- How do predictions change along connectivity paths?

## Project Structure

```
Mode-Connectivity/
├── scripts/
│   ├── slurm/
│   │   ├── endpoints/       # Train initial models
│   │   ├── curves/          # Train connectivity curves
│   │   ├── evaluation/      # Evaluate trained curves
│   │   ├── neuronswap/      # Neuron swap experiments
│   │   └── pipelines/       # Complete workflows
│   ├── train/              # Training scripts
│   ├── eval/               # Evaluation scripts
│   ├── analysis/           # Analysis tools
│   └── plot/               # Visualization scripts
├── configs/garipov/
│   ├── endpoints/          # Endpoint training configs
│   ├── curves/             # Curve training configs
│   └── neuronswap/         # Neuron swap configs
└── results/vgg16/cifar10/
    ├── endpoints/          # Trained endpoint models
    ├── curve_*/            # Trained curves and evaluations
    └── endpoint_l2_distances/  # Distance analysis
```

## Setup

### Local Development

```bash
# Install dependencies
poetry install

# Download CIFAR-10 dataset
poetry run python scripts/utils/download_cifar10.py
```

### DAIC Cluster

```bash
# Clone repository
git clone --recurse-submodules git@github.com:yourusername/Mode-Connectivity.git
cd Mode-Connectivity

# Create virtual environment (on compute node)
srun --partition=general --gpus=1 --mem=16GB --time=1:00:00 --pty bash
source ~/.bashrc
python -m venv ~/venvs/mode-connectivity
source ~/venvs/mode-connectivity/bin/activate
pip install torch torchvision hydra-core omegaconf wandb tabulate

# Download CIFAR-10
python scripts/utils/download_cifar10.py
```

## Quick Start

### Basic Mode Connectivity Workflow

```bash
# 1. Train endpoint models (seed0, seed1)
sbatch scripts/slurm/endpoints/submit_garipov.sh

# 2. Train Bezier curve connecting endpoints
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_noreg.sh

# 3. Evaluate curve
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-seed1_noreg.sh
```

### Complete Pipeline (Single Command)

```bash
# Run entire workflow: train endpoints → train curve → evaluate
sbatch scripts/slurm/pipelines/submit_garipov_full_pipeline.sh
```

## Key Experiments

### 1. Basic Mode Connectivity

**Purpose:** Connect independently trained networks (different random seeds)

**Endpoints:**
- `seed0`: VGG16 trained with random seed 0
- `seed1`: VGG16 trained with random seed 1

**Variants:**
- `curve_seed0-seed1_reg`: With L2 regularization (wd=5e-4)
- `curve_seed0-seed1_noreg`: Without regularization (wd=0.0)

**Commands:**
```bash
# Train curve
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_noreg.sh

# Evaluate
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-seed1_noreg.sh
```

### 2. Permutation Experiments

**Purpose:** Connect original network to its permuted version (all neurons swapped)

**Endpoints:**
- `seed0`: Original network
- `mirror`: Full neuron permutation of seed0

**Commands:**
```bash
# Train curve
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-mirror_noreg.sh

# Evaluate
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-mirror_noreg.sh
```

### 3. Neuron Swap Experiments

**Purpose:** Test whether connectivity corrections are local or global

**Research Question:** When swapping just 2 neurons, does the connectivity path adjust:
- **Locally**: Only the swapped layer changes?
- **Globally**: All layers readjust throughout the network?

**Setup:**
```bash
# Create swapped endpoints (early, mid, late layers)
bash scripts/workflows/setup_neuronswap_experiments.sh
```

**Train curves:**
```bash
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_early2.sh
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_mid2.sh
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_late2.sh
```

**Analyze:**
```bash
# Edit EXPERIMENT variable (early2, mid2, or late2)
sbatch scripts/slurm/neuronswap/submit_neuronswap_analysis.sh
```

**Visualize:**
```bash
python scripts/plotting/plot_layer_distance_animation.py \
    --data results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_along_curve.npz \
    --output results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_evolution.gif \
    --fps 10 \
    --metric normalized_l2 \
    --heatmap
```

## Essential Analysis Tools

### 1. L2 Distance Tracking

**Purpose:** Measure Euclidean distance between networks in weight space

**Automatic calculation:**
- During neuron swapping (in metadata files)
- Before curve training (in curve directory)

**Manual calculation:**
```bash
# Single pair
python scripts/analysis/calculate_endpoint_l2.py \
    --checkpoint1 PATH_TO_FIRST.pt \
    --checkpoint2 PATH_TO_SECOND.pt \
    --output results.json

# All pairs
bash scripts/workflows/calculate_all_endpoint_l2.sh
```

**Output:** `endpoint_l2_distances/` with per-pair JSON files and summary

**Key metrics:**
- **Normalized L2**: Distance per parameter (recommended for comparisons)
- **Relative Distance**: Percentage change from original
- **Raw L2**: Absolute Euclidean distance

### 2. Prediction Changes Analysis

**Purpose:** Analyze which samples change predictions along the curve

**Workflow:**
```bash
# 1. Collect detailed predictions (on cluster/GPU)
python scripts/eval/eval_curve_detailed.py \
    --curve_ckpt CURVE_CHECKPOINT.pt \
    --output predictions_detailed.npz \
    --dataset CIFAR10 \
    --model VGG16 \
    --num_points 61

# 2. Analyze changes
python scripts/analysis/analyze_prediction_changes.py \
    --predictions predictions_detailed.npz \
    --output analysis/

# 3. Visualize in 2D (UMAP)
python scripts/plot/plot_prediction_changes.py \
    --analysis analysis/changing_samples_info.npz \
    --output figures/

# 4. Create animation
python scripts/plot/create_prediction_animation.py \
    --analysis analysis/changing_samples_info.npz \
    --predictions predictions_detailed.npz \
    --output figures/prediction_evolution.gif \
    --fps 5
```

**Insights:**
- Which classes are most unstable?
- Where do prediction changes occur (what t values)?
- Are changing samples near decision boundaries?

### 3. Layer-wise Distance Analysis

**Purpose:** Measure per-layer changes along connectivity curves

**Usage:**
```bash
python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint CURVE_CHECKPOINT.pt \
    --original-checkpoint ORIGINAL.pt \
    --swap-metadata METADATA.json \
    --output-dir analysis/ \
    --num-points 61
```

**Output:**
- `layer_distances_along_curve.npz`: Distance data for all layers
- `analysis_summary.json`: Human-readable summary

## Configuration

### Model Settings
- **Architecture:** VGG16 (no batch normalization)
- **Dataset:** CIFAR-10
- **Training:** 200 epochs, SGD with momentum (0.9)
- **Learning rate:** 0.05 (endpoints), 0.015 (curves)

### Curve Settings
- **Type:** Bezier curve with 3 control points
- **Training:** 200 epochs (curves), 600 epochs (full pipeline)
- **Evaluation:** 61 points from t=0 to t=1

### Experiment Variants
- `_reg`: L2 weight decay = 5e-4
- `_noreg`: L2 weight decay = 0.0

## Interpreting Results

### Expected Outcomes

**Linear Interpolation (Baseline):**
- High loss barrier between endpoints
- Test error spikes in middle (~30-50%)
- Indicates endpoints are in different "modes"

**Bezier Curve (Mode Connected):**
- Low loss path connecting endpoints
- Test error remains low (~7-10%)
- Demonstrates non-linear connectivity in weight space

### L2 Distance Expectations

| Experiment | Normalized L2 | Interpretation |
|------------|---------------|----------------|
| seed0 ↔ seed1 | ~1.5 - 3.0 | Large (different initializations) |
| seed0 ↔ mirror | ~1.0 - 2.0 | Large (full permutation) |
| seed0 ↔ early2 | ~0.001 - 0.01 | Tiny (2 neurons swapped) |
| seed0 ↔ mid2 | ~0.001 - 0.01 | Tiny (2 neurons swapped) |
| seed0 ↔ late2 | ~0.001 - 0.01 | Tiny (2 neurons swapped) |

**Key insight:** Functionally equivalent networks (same accuracy) can be far apart in weight space!

### Neuron Swap Results

**Hypothesis A: Local Correction**
- Swapped layer shows high distance
- Other layers show minimal distance
- Interpretation: Weight space has local structure

**Hypothesis B: Global Adjustment**
- Many layers show significant distance
- Swapped layer may not be most changed
- Interpretation: Weight space has entangled structure

## Visualization

### Download Results from Cluster

```bash
# Download evaluations
scp -r user@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve_*/evaluations/ local/path/

# Download analysis
scp -r user@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve_*/analysis/ local/path/
```

### Generate Plots

```bash
# Connectivity comparison (6-panel plot)
python scripts/plot/plot_connectivity.py \
    --linear CURVE_DIR/evaluations/linear.npz \
    --curve CURVE_DIR/evaluations/curve.npz \
    --output CURVE_DIR/figures/connectivity_comparison.png

# Layer distance animation
python scripts/plotting/plot_layer_distance_animation.py \
    --data ANALYSIS_DIR/layer_distances_along_curve.npz \
    --output ANALYSIS_DIR/layer_distances_evolution.gif \
    --fps 10 \
    --metric normalized_l2 \
    --heatmap
```

## Graceful Shutdown

Training scripts support graceful shutdown on Ctrl+C or SIGTERM:

1. Press **Ctrl+C** during training
2. Current epoch completes
3. Checkpoint saved as `checkpoint-interrupted-<epoch>.pt`
4. Clean exit

**Resume training:**
```bash
python external/dnn-mode-connectivity/train.py \
    --resume checkpoint-interrupted-42.pt \
    [other arguments...]
```

## Naming Conventions

### Experiment Directories
Pattern: `curve_{endpoint1}-{endpoint2}_{variant}`

Examples:
- `curve_seed0-seed1_noreg`: Different seeds, no regularization
- `curve_seed0-mirror_reg`: Permutation, with regularization
- `curve_neuronswap_mid2_reg`: 2 neurons swapped in middle layer

### Checkpoint Files
- `checkpoint-<epoch>.pt`: Regular checkpoint
- `checkpoint-interrupted-<epoch>.pt`: Saved on Ctrl+C
- `checkpoint-best.pt`: Best validation performance

### Output Files
- `linear.npz`: Linear interpolation evaluation
- `curve.npz`: Bezier curve evaluation
- `predictions_detailed.npz`: Detailed predictions and features
- `layer_distances_along_curve.npz`: Per-layer distance analysis
- `endpoint_l2_distance.txt`: Distance between curve endpoints

## Time Estimates

| Task | Duration | Location |
|------|----------|----------|
| Train endpoints (2 models) | 6-8 hours | Cluster (GPU) |
| Train curve | 30-40 hours | Cluster (GPU) |
| Evaluate curve | 30 minutes | Cluster (GPU) |
| Create neuron swap endpoints | 5 minutes | Local |
| Train neuron swap curve | 2 hours | Cluster (GPU) |
| Analyze layer distances | 10 minutes | Cluster/Local |
| Generate visualizations | 2-5 minutes | Local |

## References

- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS 2018*.
- Entezari et al. (2021). The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.
- Original implementation: https://github.com/timgaripov/dnn-mode-connectivity

## License

MIT License
