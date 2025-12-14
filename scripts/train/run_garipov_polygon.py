import os
import sys
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from src.utils import set_global_seed
from lib.metrics.distances import calculate_checkpoint_l2_distance, save_l2_distance_report, print_l2_statistics
from lib.core.training_commands import (
    build_base_command, add_curve_args, add_wandb_args, add_seed_arg,
    add_training_hyperparams, add_optional_arg, print_and_format_command
)


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/polygon",
    config_name="vgg16_polygon_seed0-mirror",
)
def main(cfg: DictConfig):
    seed = cfg.get('seed', 0)
    set_global_seed(seed)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    run_dir = to_absolute_path(cfg.output_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Get absolute paths to endpoints
    init_start = to_absolute_path(cfg.init_start)
    init_end = to_absolute_path(cfg.init_end)

    # Calculate and log L2 distance between endpoints
    print("\n" + "="*80)
    print("POLYGON CHAIN OPTIMIZATION")
    print("="*80)
    print("Calculating L2 distance between endpoints...")
    l2_stats = calculate_checkpoint_l2_distance(init_start, init_end)
    print_l2_statistics(l2_stats, endpoint_names=(cfg.init_start, cfg.init_end), title="")

    # Save L2 distance to file
    l2_file = save_l2_distance_report(run_dir, l2_stats, (cfg.init_start, cfg.init_end))
    print(f"✓ L2 distance saved to: {l2_file}")
    print("="*80 + "\n")

    # Build command for polygon chain training
    cmd = build_base_command(train_script, run_dir, cfg)
    add_curve_args(cmd, cfg, init_start, init_end, fix_endpoints=True,
                  curve_type="PolyChain", num_bends=3)
    add_training_hyperparams(cmd, cfg)
    add_seed_arg(cmd, seed)
    add_optional_arg(cmd, cfg, 'save_freq', '--save_freq', default=50)
    add_optional_arg(cmd, cfg, 'use_test', '--use_test', is_flag=True)

    # Add WandB logging
    run_name = f"polygon_{cfg.model}_{cfg.experiment_name}"
    add_wandb_args(cmd, cfg, run_name)

    print_and_format_command(cmd)
    subprocess.run(cmd, check=True)

    print("\n" + "="*80)
    print("POLYGON CHAIN OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
