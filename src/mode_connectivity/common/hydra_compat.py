"""Compose experiment configs from the repo root across Hydra versions."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


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
        cfg = compose(config_name=config_name, overrides=overrides)
    return _normalize_config_shape(cfg, config_name)


def _deep_merge(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def _flatten_package_layers(node: dict) -> dict:
    result: dict = {}
    package_keys = {"curves", "sinkhorn", "lmc", "_base", "geometry", "pairs", "presets", "splits"}
    for key, value in node.items():
        if isinstance(value, dict) and key in package_keys:
            _deep_merge(result, _flatten_package_layers(value))
        else:
            result[key] = value
    return result


def _normalize_config_shape(cfg: DictConfig, config_name: str) -> DictConfig:
    container = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(container, dict):
        return cfg

    parts = [part for part in config_name.split("/") if part]
    group_parts = parts[:-1] if len(parts) > 1 else []
    node = container
    for part in group_parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            node = None
            break

    if not isinstance(node, dict):
        return cfg

    flattened = _flatten_package_layers(node)
    if group_parts:
        root_group = group_parts[0]
        for key, value in container.items():
            if key != root_group:
                flattened[key] = value
    return OmegaConf.create(flattened)
