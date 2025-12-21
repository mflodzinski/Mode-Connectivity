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
    add_save_freq_arg, add_optional_arg, print_and_format_command
)


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/curves_init",
    config_name="vgg16_curve_seed0-seed1_reg",
)
def main(cfg: DictConfig):
    seed = cfg.get('seed', 0)
    set_global_seed(seed)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    run_dir = to_absolute_path(cfg.output_root)
    os.makedirs(run_dir, exist_ok=True)

    # Get absolute paths to endpoints
    endpoint0 = to_absolute_path(cfg.endpoint0)
    endpoint1 = to_absolute_path(cfg.endpoint1)

    # Calculate and log L2 distance between endpoints
    l2_stats = calculate_checkpoint_l2_distance(endpoint0, endpoint1)
    print_l2_statistics(l2_stats, endpoint_names=(cfg.endpoint0, cfg.endpoint1))

    # Save L2 distance to file
    l2_file = save_l2_distance_report(run_dir, l2_stats, (cfg.endpoint0, cfg.endpoint1))
    print(f"\n✓ L2 distance saved to: {l2_file}")
    print("="*70 + "\n")

    # Build training command
    cmd = build_base_command(train_script, run_dir, cfg)
    add_curve_args(cmd, cfg, endpoint0, endpoint1, fix_endpoints=True)
    add_seed_arg(cmd, seed)
    add_save_freq_arg(cmd, cfg)
    add_optional_arg(cmd, cfg, 'use_test', '--use_test', is_flag=True)

    # Add WandB logging
    run_name = f"garipov_{cfg.model}_curve_{cfg.curve}"
    add_wandb_args(cmd, cfg, run_name)

    print_and_format_command(cmd)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
