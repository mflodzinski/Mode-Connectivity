"""Compose experiment configs from the repo root across Hydra versions."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def _find_config_root(caller_file: str) -> Path:
    for parent in Path(caller_file).resolve().parents:
        candidate = parent / "configs" / "experiments"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate configs/experiments from {caller_file}")


def compose_experiment_config(
    *,
    default_config_name: str,
    caller_file: str,
    argv: list[str] | None = None,
) -> DictConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    config_name = default_config_name
    overrides: list[str] = []

    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config-name":
            if index + 1 >= len(args):
                raise ValueError("--config-name requires a value")
            config_name = args[index + 1]
            index += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            index += 1
            continue
        overrides.append(arg)
        index += 1

    config_root = _find_config_root(caller_file)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_root)):
        return compose(config_name=config_name, overrides=overrides)
