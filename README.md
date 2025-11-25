# Mode Connectivity Research

Implementation of mode connectivity experiments for neural networks, based on [DNN Mode Connectivity](https://github.com/timgaripov/dnn-mode-connectivity) (Garipov et al., 2018).

## Project Structure

```
Mode-Connectivity/
├── configs/                    # Hydra configuration files
│   └── garipov/
│       ├── vgg16_endpoints.yaml    # Endpoint training config
│       └── vgg16_curve.yaml        # Curve training config
├── src/                        # Source code
│   └── utils.py               # Utility functions
├── external/                   # External repositories
│   └── dnn-mode-connectivity/ # Forked mode connectivity implementation
└── scripts/
    ├── slurm/                 # SLURM job submission scripts
    │   ├── submit_garipov.sh              # Submit endpoint training
    │   ├── submit_garipov_curve.sh        # Submit curve training
    │   └── submit_garipov_full_pipeline.sh # Submit full pipeline
    ├── train/                 # Training scripts
    │   ├── run_garipov_endpoints.py       # Train two endpoint models
    │   └── run_garipov_curve.py           # Train Bezier curve
    ├── eval/                  # Evaluation scripts
    │   ├── eval_garipov_curve.py          # Evaluate trained curve
    │   └── eval_linear.py                 # Evaluate linear interpolation
    ├── plot/                  # Plotting scripts
    │   └── plot_connectivity.py           # Plot connectivity comparison
    └── utils/                 # Utility scripts
        └── download_cifar10.py            # Download CIFAR-10 dataset
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

## Usage

### 1. Train Endpoint Models

Train two VGG16 models with different random seeds:

```bash
# Local
poetry run python scripts/train/run_garipov_endpoints.py

# Cluster
sbatch scripts/slurm/submit_garipov.sh
```

**Config:** [configs/garipov/vgg16_endpoints.yaml](configs/garipov/vgg16_endpoints.yaml)
- Dataset: CIFAR-10
- Model: VGG16 (no batch norm)
- Seeds: [0, 1]
- Epochs: 200
- Learning rate: 0.05
- Optimizer: SGD with momentum (0.9)

**Output:** `results/garipov/vgg16/endpoints/VGG16_seed{0,1}/checkpoint-200.pt`

### 2. Full Mode Connectivity Pipeline

Run the complete pipeline (train curve + evaluate curve + evaluate linear):

```bash
sbatch scripts/slurm/submit_garipov_full_pipeline.sh
```

This sequentially runs:
1. **Train Bezier curve** (~30-40 hours for 600 epochs)
2. **Evaluate curve** (~30 min)
3. **Evaluate linear interpolation** (~30 min)

**Outputs:**
- Bezier curve: `results/garipov/vgg16/curve/curve.npz`
- Linear path: `results/garipov/vgg16/curve/linear/linear.npz`

### 3. Visualize Results

Download results and plot:

```bash
# Download from cluster
scp -r mlodzinski@linux-bastion-ex.tudelft.nl:Mode-Connectivity/results/garipov/vgg16/curve .

# Plot comparison
python scripts/plot/plot_connectivity.py \
    --linear curve/linear/linear.npz \
    --curve curve/curve.npz \
    --output connectivity_comparison.png
```

## Expected Results

**Linear Interpolation:** High loss barrier between endpoints (poor connectivity)
- Test error spikes in the middle (~30-50%)
- Indicates endpoints are in different "modes"

**Bezier Curve:** Low loss path connecting endpoints (mode connected)
- Test error remains low throughout (~7-10%)
- Demonstrates non-linear path in weight space connecting modes

## References

- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS 2018*.
- Original implementation: https://github.com/timgaripov/dnn-mode-connectivity
