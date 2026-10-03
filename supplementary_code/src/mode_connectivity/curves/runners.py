"""Shared orchestration helpers for the curve experiment runners.

These functions keep the repo-level curve entrypoints thin by centralizing
command construction, endpoint diagnostics, and upstream evaluation dispatch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from mode_connectivity.core.training_commands import (
    add_curve_args,
    add_optional_arg,
    add_save_freq_arg,
    add_seed_arg,
    add_training_hyperparams,
    add_wandb_args,
    build_base_command,
    print_and_format_command,
)
from mode_connectivity.evaluation.metrics import (
    calculate_checkpoint_l2_distance,
    print_l2_statistics,
    save_l2_distance_report,
)
from mode_connectivity.external import eval_curve_script_path, train_script_path


def _ensure_dir(path_like: str) -> Path:
    path = Path(to_absolute_path(path_like))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_path(path_like: str) -> str:
    return str(Path(to_absolute_path(path_like)))


def log_endpoint_distance(
    *,
    output_dir: str,
    left_label: str,
    left_path: str,
    right_label: str,
    right_path: str,
    title: str,
) -> str:
    l2_stats = calculate_checkpoint_l2_distance(left_path, right_path)
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)
        print("Calculating L2 distance between endpoints...")
    print_l2_statistics(l2_stats, endpoint_names=(left_label, right_label), title="")
    output_path = save_l2_distance_report(output_dir, l2_stats, (left_label, right_label))
    print(f"✓ L2 distance saved to: {output_path}")
    print("=" * 80 + "\n")
    return output_path


def build_curve_training_command(
    *,
    cfg: DictConfig,
    output_dir: str,
    endpoint0: str,
    endpoint1: str,
    curve_type: str | None = None,
    num_bends: int | None = None,
    include_training_hparams: bool = False,
    extra_flags: Iterable[str] = (),
    extra_kv: dict[str, object] | None = None,
) -> list[str]:
    cmd = build_base_command(str(train_script_path()), output_dir, cfg)
    add_curve_args(
        cmd,
        cfg,
        endpoint0,
        endpoint1,
        fix_endpoints=True,
        curve_type=curve_type,
        num_bends=num_bends,
    )
    if include_training_hparams:
        add_training_hyperparams(cmd, cfg)
    add_seed_arg(cmd, int(cfg.get("seed", 0)))
    add_save_freq_arg(cmd, cfg)
    add_optional_arg(cmd, cfg, "use_test", "--use_test", is_flag=True)
    add_optional_arg(cmd, cfg, "no_train_aug", "--no_train_aug", is_flag=True)
    add_optional_arg(cmd, cfg, "train_half_only", "--train_half_only", is_flag=True)
    for flag in extra_flags:
        cmd.append(flag)
    for key, value in (extra_kv or {}).items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_training_command(cmd: list[str], *, cfg: DictConfig, run_name: str) -> None:
    add_wandb_args(cmd, cfg, run_name)
    print_and_format_command(cmd)
    subprocess.run(cmd, check=True)


def run_curve_evaluation(
    *,
    cfg: DictConfig,
    output_root_key: str = "output_root",
    num_points: int = 61,
) -> None:
    output_root = str(cfg[output_root_key])
    curve_checkpoint = resolve_checkpoint_path(f"{output_root}/checkpoint-{cfg.epochs}.pt")
    eval_dir = _ensure_dir(output_root.replace("/checkpoints", "/evaluations"))
    cmd = [
        sys.executable,
        str(eval_curve_script_path()),
        "--dir",
        str(eval_dir),
        "--dataset",
        str(cfg.dataset),
        "--data_path",
        str(cfg.data_path),
        "--transform",
        str(cfg.transform),
        "--model",
        str(cfg.model),
        "--curve",
        str(cfg.curve),
        "--num_bends",
        str(cfg.num_bends),
        "--ckpt",
        curve_checkpoint,
        "--num_points",
        str(num_points),
    ]
    if cfg.get("num_workers") is not None:
        cmd.extend(["--num_workers", str(cfg.num_workers)])
    if cfg.get("use_test"):
        cmd.append("--use_test")
    print("Evaluating curve:", " ".join(cmd))
    subprocess.run(cmd, check=True)
