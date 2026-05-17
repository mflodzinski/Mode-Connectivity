"""Train the endpoint models used by the curved-path experiments.

The runner iterates over configured seeds, builds one upstream endpoint command
per seed, and keeps the orchestration for those runs in one place.
"""

from omegaconf import DictConfig

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.common.utils import set_global_seed
from mode_connectivity.core.training_commands import (
    build_base_command, add_wandb_args, add_seed_arg,
    add_save_freq_arg, add_early_stopping_args, add_optional_arg,
    print_and_format_command
)
from mode_connectivity.external import train_script_path

def run(cfg: DictConfig) -> None:
    seed = cfg.get('seed', 0)
    set_global_seed(seed)

    for seed in cfg.seeds:
        run_dir = f"{cfg.output_root}/seed{seed}"

        # Build training command
        cmd = build_base_command(str(train_script_path()), run_dir, cfg)
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
        import subprocess
        subprocess.run(cmd, check=True)


def main() -> None:
    cfg = compose_experiment_config(
        default_config_name="curves/runs/endpoints_standard",
        caller_file=__file__,
    )
    run(cfg)

if __name__ == "__main__":
    main()
