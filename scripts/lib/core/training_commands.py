"""Utilities for building training command-line arguments.

Provides helpers to construct subprocess commands for training scripts.
"""

from typing import List, Optional, Any
from omegaconf import DictConfig


def build_base_command(train_script: str,
                      run_dir: str,
                      cfg: DictConfig) -> List[str]:
    """Build base training command with required arguments.

    Args:
        train_script: Path to training script
        run_dir: Output directory for training run
        cfg: Hydra configuration object

    Returns:
        List of command arguments
    """
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
        "--epochs",
        str(cfg.epochs),
        "--lr",
        str(cfg.lr),
        "--wd",
        str(cfg.wd),
    ]
    return cmd


def add_curve_args(cmd: List[str],
                  cfg: DictConfig,
                  endpoint0: str,
                  endpoint1: str,
                  fix_endpoints: bool = True,
                  curve_type: Optional[str] = None,
                  num_bends: Optional[int] = None) -> None:
    """Add curve-specific arguments to command.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
        endpoint0: Path to first endpoint checkpoint
        endpoint1: Path to second endpoint checkpoint
        fix_endpoints: Whether to fix endpoints during training
        curve_type: Override curve type (default: use cfg.curve)
        num_bends: Override number of bends (default: use cfg.num_bends)
    """
    cmd += ["--curve", curve_type if curve_type else cfg.curve]
    cmd += ["--num_bends", str(num_bends if num_bends else cfg.num_bends)]
    cmd += ["--init_start", endpoint0]
    if fix_endpoints:
        cmd.append("--fix_start")
    cmd += ["--init_end", endpoint1]
    if fix_endpoints:
        cmd.append("--fix_end")


def add_wandb_args(cmd: List[str],
                  cfg: DictConfig,
                  run_name: str) -> None:
    """Add WandB logging arguments if enabled.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
        run_name: Name for the WandB run
    """
    if cfg.use_wandb:
        cmd.append("--wandb")
        cmd += ["--wandb_project", cfg.project_name]
        cmd += ["--wandb_name", run_name]


def add_seed_arg(cmd: List[str], seed: int) -> None:
    """Add seed argument to command.

    Args:
        cmd: Command list to modify in-place
        seed: Random seed value
    """
    cmd += ["--seed", str(seed)]


def add_optional_arg(cmd: List[str],
                    cfg: DictConfig,
                    config_key: str,
                    flag: str,
                    default: Any = None,
                    is_flag: bool = False) -> None:
    """Add optional argument if present in config.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
        config_key: Key to look up in config
        flag: Command-line flag to add
        default: Default value if key not found
        is_flag: If True, treat as boolean flag (no value)
    """
    value = cfg.get(config_key, default)
    if value is not None:
        if is_flag:
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]


def add_save_freq_arg(cmd: List[str], cfg: DictConfig) -> None:
    """Add save_freq argument if intermediate checkpoints requested.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
    """
    if cfg.get("save_intermediate", True):
        cmd += ["--save_freq", str(cfg.save_freq)]


def add_early_stopping_args(cmd: List[str], cfg: DictConfig) -> None:
    """Add early stopping arguments if enabled.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
    """
    if cfg.get("early_stopping", False):
        cmd.append("--early_stopping")
        cmd += ["--patience", str(cfg.get("patience", 20))]
        cmd += ["--min_delta", str(cfg.get("min_delta", 0.0))]
        cmd.append("--split_test_from_train")


def add_training_hyperparams(cmd: List[str], cfg: DictConfig) -> None:
    """Add optional training hyperparameters.

    Args:
        cmd: Command list to modify in-place
        cfg: Hydra configuration object
    """
    # Add momentum if present
    if hasattr(cfg, 'momentum'):
        cmd += ["--momentum", str(cfg.momentum)]

    # Add batch size if present
    if hasattr(cfg, 'batch_size'):
        cmd += ["--batch_size", str(cfg.batch_size)]

    # Add num workers if present
    if hasattr(cfg, 'num_workers'):
        cmd += ["--num-workers", str(cfg.num_workers)]


def print_and_format_command(cmd: List[str]) -> None:
    """Print command in readable format.

    Args:
        cmd: Command list to print
    """
    print("Running:", " ".join(cmd))
