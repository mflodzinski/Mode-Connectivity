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
    config_path="../../configs/garipov/curves",
    config_name="vgg16_curve_seed0-seed1_reg",
)
def main(cfg: DictConfig):
    set_global_seed(0)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    run_dir = to_absolute_path(cfg.output_root)
    os.makedirs(run_dir, exist_ok=True)

    # Get absolute paths to endpoints
    endpoint0 = to_absolute_path(cfg.endpoint0)
    endpoint1 = to_absolute_path(cfg.endpoint1)

    # Calculate and log L2 distance between endpoints
    print("\n" + "="*70)
    print("CALCULATING L2 DISTANCE BETWEEN ENDPOINTS")
    print("="*70)
    l2_stats = calculate_endpoint_l2(endpoint0, endpoint1)
    print(f"Endpoint 0: {cfg.endpoint0}")
    print(f"Endpoint 1: {cfg.endpoint1}")
    print(f"\nL2 Distance Statistics:")
    print(f"  Total L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"  Normalized L2 distance: {l2_stats['normalized_l2']:.6f}")
    print(f"  Total parameters:       {l2_stats['total_params']:,}")

    # Save L2 distance to file
    l2_file = os.path.join(run_dir, "endpoint_l2_distance.txt")
    with open(l2_file, 'w') as f:
        f.write(f"L2 Distance Between Endpoints\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Endpoint 0: {cfg.endpoint0}\n")
        f.write(f"Endpoint 1: {cfg.endpoint1}\n\n")
        f.write(f"Total L2 distance:      {l2_stats['total_l2']:.6f}\n")
        f.write(f"Normalized L2 distance: {l2_stats['normalized_l2']:.6f}\n")
        f.write(f"Total parameters:       {l2_stats['total_params']:,}\n")
    print(f"\n✓ L2 distance saved to: {l2_file}")
    print("="*70 + "\n")

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
        cfg.curve,
        "--num_bends",
        str(cfg.num_bends),
        "--init_start",
        endpoint0,
        "--fix_start",  # Fix the starting endpoint
        "--init_end",
        endpoint1,
        "--fix_end",  # Fix the ending endpoint
        "--epochs",
        str(cfg.epochs),
        "--lr",
        str(cfg.lr),
        "--wd",
        str(cfg.wd),
    ]

    # Only save intermediate checkpoints if requested
    if cfg.get("save_intermediate", True):
        cmd += ["--save_freq", str(cfg.save_freq)]

    if cfg.use_test:
        cmd.append("--use_test")

    if cfg.use_wandb:
        run_name = f"garipov_{cfg.model}_curve_{cfg.curve}"
        cmd.append("--wandb")
        cmd += ["--wandb_project", cfg.project_name]
        cmd += ["--wandb_name", run_name]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
