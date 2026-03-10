"""Train VGG16 endpoints on MNIST using the existing dnn-mode-connectivity trainer."""

import os
import subprocess
import sys

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, "..")
sys.path.insert(0, scripts_root)

from src.utils import set_global_seed
from lib.core.training_commands import (
    add_early_stopping_args,
    add_optional_arg,
    add_save_freq_arg,
    add_seed_arg,
    add_training_hyperparams,
    add_wandb_args,
    build_base_command,
    print_and_format_command,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/endpoints",
    config_name="vgg16_mnist_endpoints",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.get("seed", 0)))

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    for seed in cfg.seeds:
        run_dir = to_absolute_path(os.path.join(cfg.output_root, f"seed{seed}"))
        os.makedirs(run_dir, exist_ok=True)

        cmd = build_base_command(train_script, run_dir, cfg)
        add_training_hyperparams(cmd, cfg)
        add_seed_arg(cmd, int(seed))
        add_save_freq_arg(cmd, cfg)
        add_optional_arg(cmd, cfg, "use_test", "--use_test", is_flag=True)
        add_early_stopping_args(cmd, cfg)

        run_name = f"garipov_{cfg.model}_{cfg.dataset}_seed{seed}"
        if cfg.get("early_stopping", False):
            run_name += "_early_stop"
        add_wandb_args(cmd, cfg, run_name)

        print_and_format_command(cmd)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
