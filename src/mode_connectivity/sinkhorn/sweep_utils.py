"""Shared sweep helpers for the retained VGG/CIFAR Sinkhorn workflows.

The functions here enumerate hyperparameter grids, derive output tags, and load
comparison summaries for the repo-level sweep runner.
"""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

from omegaconf import DictConfig, OmegaConf

from mode_connectivity.core.output import load_json


def _listify(values: Iterable[Any]) -> list[Any]:
    return [OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value for value in values]


def enumerate_sweep_combinations(sweep_cfg: DictConfig) -> List[Dict[str, Any]]:
    keys = list(sweep_cfg.keys())
    value_lists = [_listify(sweep_cfg[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def sanitize_value(value: Any) -> str:
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "_")
    return text


def resolve_task_id(cfg: DictConfig) -> int:
    if "sweep_task_id" in cfg and cfg.sweep_task_id is not None:
        return int(cfg.sweep_task_id)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    if "SWEEP_TASK_ID" in os.environ:
        return int(os.environ["SWEEP_TASK_ID"])
    return 0


def build_output_tag(
    combo: Dict[str, Any],
    *,
    finetune_mode: str = "joint",
    starting_alignment_artifact: str | None = None,
) -> str:
    parts = [
        f"steps{combo['alignment_iterations']}",
        f"tau{sanitize_value(combo['tau'])}",
        f"lr{sanitize_value(combo['lr'])}",
        f"l{sanitize_value(combo['sinkhorn_l'])}",
        f"loss{combo['loss_name']}",
    ]
    if "lambda_scale" in combo:
        parts.append(f"lam{sanitize_value(combo['lambda_scale'])}")
    if finetune_mode != "joint" or starting_alignment_artifact not in (None, "", "null"):
        parts.append(f"ft{finetune_mode}")
    return "_".join(parts)


def load_run_comparison(output_root: str) -> Dict[str, Dict[str, Any]]:
    rows = load_json(Path(output_root) / "comparison.json")
    return {row["variant_key"]: row for row in rows}


def build_sweep_comparison_row(run_record: Dict[str, Any]) -> Dict[str, Any]:
    comparison = load_run_comparison(run_record["output_root"])
    lmc = comparison["original_sinkhorn_lmc"]
    no_alignment = comparison["no_alignment"]
    combo = run_record["combo"]
    metadata = load_json(Path(run_record["output_root"]) / "metadata.json")
    return {
        "task_id": run_record["task_id"],
        "output_tag": run_record["output_tag"],
        "output_root": run_record["output_root"],
        "alignment_iterations": combo["alignment_iterations"],
        "loss_name": combo["loss_name"],
        "tau": combo["tau"],
        "lr": combo["lr"],
        "sinkhorn_l": combo["sinkhorn_l"],
        "lambda_scale": combo.get("lambda_scale", 0.0),
        "best_alignment_score": metadata.get("best_alignment_score"),
        "best_alignment_iteration": metadata.get("best_alignment_iteration"),
        "lmc_mean_val_interp_loss": lmc["mean_val_interp_loss"],
        "lmc_val_loss_barrier_max_endpoint": lmc["val_loss_barrier_max_endpoint"],
        "lmc_min_val_acc": lmc["min_val_acc"],
        "lmc_mean_test_interp_loss": lmc["mean_test_interp_loss"],
        "lmc_test_loss_barrier_max_endpoint": lmc["test_loss_barrier_max_endpoint"],
        "lmc_min_test_acc": lmc["min_test_acc"],
        "no_align_mean_val_interp_loss": no_alignment["mean_val_interp_loss"],
        "no_align_val_loss_barrier_max_endpoint": no_alignment["val_loss_barrier_max_endpoint"],
        "no_align_min_val_acc": no_alignment["min_val_acc"],
        "no_align_mean_test_interp_loss": no_alignment["mean_test_interp_loss"],
        "no_align_test_loss_barrier_max_endpoint": no_alignment["test_loss_barrier_max_endpoint"],
        "no_align_min_test_acc": no_alignment["min_test_acc"],
        "delta_val_barrier_vs_no_align": lmc["val_loss_barrier_max_endpoint"] - no_alignment["val_loss_barrier_max_endpoint"],
        "delta_mean_test_interp_loss_vs_no_align": lmc["mean_test_interp_loss"] - no_alignment["mean_test_interp_loss"],
        "delta_test_loss_barrier_max_endpoint_vs_no_align": lmc["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"],
        "delta_min_test_acc_vs_no_align": lmc["min_test_acc"] - no_alignment["min_test_acc"],
    }


def print_top_runs(rows: list[Dict[str, Any]], *, top_k: int = 5) -> None:
    if not rows:
        print("No completed runs available for sweep comparison.")
        return

    top_barrier = sorted(rows, key=lambda row: row["lmc_val_loss_barrier_max_endpoint"])[:top_k]
    top_mean = sorted(rows, key=lambda row: float("inf") if row["best_alignment_score"] is None else row["best_alignment_score"])[:top_k]

    print("")
    print("=" * 80)
    print("TOP RUNS BY LMC VALIDATION BARRIER")
    print("=" * 80)
    for row in top_barrier:
        print(
            f"task={row['task_id']:02d} "
            f"steps={row['alignment_iterations']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"val_barrier={row['lmc_val_loss_barrier_max_endpoint']:.4f} "
            f"best_val_loss={row['best_alignment_score']:.4f} "
            f"test_barrier={row['lmc_test_loss_barrier_max_endpoint']:.4f} "
            f"lmc_min_acc={row['lmc_min_test_acc']:.2f}"
        )

    print("")
    print("=" * 80)
    print("TOP RUNS BY VALIDATION LOSS")
    print("=" * 80)
    for row in top_mean:
        print(
            f"task={row['task_id']:02d} "
            f"steps={row['alignment_iterations']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"best_val_loss={row['best_alignment_score']:.4f} "
            f"val_barrier={row['lmc_val_loss_barrier_max_endpoint']:.4f} "
            f"test_barrier={row['lmc_test_loss_barrier_max_endpoint']:.4f} "
            f"lmc_min_acc={row['lmc_min_test_acc']:.2f}"
        )
