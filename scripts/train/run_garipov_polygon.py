import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
import numpy as np

from src.utils import set_global_seed


def calculate_endpoint_l2(checkpoint1_path, checkpoint2_path):
    """Calculate L2 distance between two checkpoints."""
    # Load checkpoints
    ckpt1 = torch.load(checkpoint1_path, map_location='cpu')
    ckpt2 = torch.load(checkpoint2_path, map_location='cpu')

    # Get state dicts
    state1 = ckpt1.get('model_state', ckpt1)
    state2 = ckpt2.get('model_state', ckpt2)

    # Calculate L2 distance
    total_l2_squared = 0.0
    total_params = 0

    for key in state1.keys():
        if key in state2 and isinstance(state1[key], torch.Tensor):
            diff = state1[key] - state2[key]
            total_l2_squared += torch.sum(diff ** 2).item()
            total_params += state1[key].numel()

    total_l2 = np.sqrt(total_l2_squared)
    normalized_l2 = total_l2 / np.sqrt(total_params) if total_params > 0 else 0

    return {
        'total_l2': total_l2,
        'normalized_l2': normalized_l2,
        'total_params': total_params
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/symmetry_plane",
    config_name="vgg16_symplane_seed0-seed1",
)
def main(cfg: DictConfig):
    set_global_seed(0)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    run_dir = to_absolute_path(cfg.output_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Get absolute paths to endpoints
    init_start = to_absolute_path(cfg.init_start)
    init_end = to_absolute_path(cfg.init_end)

    # Calculate and log L2 distance between endpoints
    print("\n" + "="*80)
    print("SYMMETRY PLANE OPTIMIZATION")
    print("="*80)
    print("Calculating L2 distance between endpoints...")
    l2_stats = calculate_endpoint_l2(init_start, init_end)
    print(f"Endpoint 1: {cfg.init_start}")
    print(f"Endpoint 2: {cfg.init_end}")
    print(f"\nL2 Distance Statistics:")
    print(f"  Total L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"  Normalized L2 distance: {l2_stats['normalized_l2']:.6f}")
    print(f"  Total parameters:       {l2_stats['total_params']:,}")

    # Save L2 distance to file
    l2_file = os.path.join(run_dir, "endpoint_l2_distance.txt")
    with open(l2_file, 'w') as f:
        f.write(f"Symmetry Plane: L2 Distance Between Endpoints\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Endpoint 1: {cfg.init_start}\n")
        f.write(f"Endpoint 2: {cfg.init_end}\n\n")
        f.write(f"Total L2 distance:      {l2_stats['total_l2']:.6f}\n")
        f.write(f"Normalized L2 distance: {l2_stats['normalized_l2']:.6f}\n")
        f.write(f"Total parameters:       {l2_stats['total_params']:,}\n")
    print(f"✓ L2 distance saved to: {l2_file}")
    print("="*80 + "\n")

    # Build command for symmetry plane training
    cmd = [
        "python",
        train_script,
        "--dir",
        run_dir,
        "--dataset",
        cfg.dataset,
        "--data_path",
        cfg.data_path,
        "--transform",
        cfg.transform,
        "--model",
        cfg.model,
        "--curve",
        "PolyChain",  # Always use PolyChain for symmetry plane
        "--num_bends",
        "3",  # Always 3 for symmetry plane
        "--init_start",
        init_start,
        "--fix_start",
        "--init_end",
        init_end,
        "--fix_end",
        "--epochs",
        str(cfg.epochs),
        "--lr",
        str(cfg.lr),
        "--wd",
        str(cfg.wd),
        "--momentum",
        str(cfg.momentum),
        "--batch_size",
        str(cfg.batch_size),
        "--num-workers",
        str(cfg.num_workers),
        "--save_freq",
        str(cfg.get('save_freq', 50)),
    ]

    if cfg.use_test:
        cmd.append("--use_test")

    if cfg.get('use_wandb', False):
        run_name = f"symplane_{cfg.model}_{cfg.experiment_name}"
        cmd.append("--wandb")
        cmd += ["--wandb_project", cfg.project_name]
        cmd += ["--wandb_name", run_name]

    print("Running command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)

    print("\n" + "="*80)
    print("SYMMETRY PLANE OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
