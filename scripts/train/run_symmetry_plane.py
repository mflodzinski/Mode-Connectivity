"""
Hydra wrapper for symmetry plane optimization.

Loads configuration and calls train_symmetry_plane.py with appropriate arguments.
"""

import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/symmetry_plane",
    config_name="vgg16_symplane_seed0-seed1",
)
def main(cfg: DictConfig):
    # Resolve paths
    init_start = to_absolute_path(cfg.init_start)
    init_end = to_absolute_path(cfg.init_end)
    output_dir = to_absolute_path(cfg.output_dir)
    data_path = to_absolute_path(cfg.data_path)
    script_path = to_absolute_path("scripts/train/train_symmetry_plane.py")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        "python",
        script_path,
        "--init_start", init_start,
        "--init_end", init_end,
        "--model", cfg.model,
        "--dataset", cfg.dataset,
        "--data_path", data_path,
        "--transform", cfg.transform,
        "--batch_size", str(cfg.batch_size),
        "--num_workers", str(cfg.num_workers),
        "--steps", str(cfg.optimization_steps),
        "--lr", str(cfg.lr),
        "--momentum", str(cfg.momentum),
        "--init_mode", cfg.init_mode,
        "--eval_points", str(cfg.eval_points_per_segment),
        "--dir", output_dir,
        "--print_freq", str(cfg.print_freq),
    ]

    if cfg.use_test:
        cmd.append("--use_test")

    print("Running symmetry plane optimization with command:")
    print(" ".join(cmd))
    print()

    # Run the training script
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 80)
    print("SYMMETRY PLANE OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - checkpoint_optimal.pt")
    print(f"  - optimization_log.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
