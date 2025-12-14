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
from lib.core.training_commands import (
    build_base_command, add_wandb_args, add_seed_arg,
    add_save_freq_arg, add_early_stopping_args, add_optional_arg,
    print_and_format_command
)

@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/endpoints",
    config_name="vgg16_endpoints",
)
def main(cfg: DictConfig):
    seed = cfg.get('seed', 0)
    set_global_seed(seed)
    
    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    for seed in cfg.seeds:
        run_dir = to_absolute_path(
            os.path.join(cfg.output_root, f"seed{seed}")
        )
        os.makedirs(run_dir, exist_ok=True)

        # Build training command
        cmd = build_base_command(train_script, run_dir, cfg)
        add_seed_arg(cmd, seed)
        add_save_freq_arg(cmd, cfg)
        add_optional_arg(cmd, cfg, 'use_test', '--use_test', is_flag=True)
        add_early_stopping_args(cmd, cfg)

        # Add WandB logging
        run_name = f"garipov_{cfg.model}_endpoint_seed{seed}"
        if cfg.get("early_stopping", False):
            run_name += "_early_stop"
        add_wandb_args(cmd, cfg, run_name)

        print_and_format_command(cmd)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()