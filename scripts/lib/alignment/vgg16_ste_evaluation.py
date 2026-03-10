"""Evaluation utilities for the VGG16 STE permutation baseline."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.lib.alignment.permutation_pipeline import compute_barrier_metrics, write_summary_files
from scripts.lib.alignment.vgg16_sinkhorn_evaluation import (
    _evaluate_endpoint_metrics,
    _evaluate_linear_interpolation,
    _per_layer_scale_means,
    _save_interpolation_npz,
)
from scripts.lib.alignment.vgg16_sinkhorn_alignment import load_alignment_artifact
from scripts.lib.analysis.alignment import load_dataset_eval_loaders
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.checkpoint import load_state_dict
from scripts.lib.core.output import ensure_dir, save_json


VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "ste_perm_soft": "STE permutation (soft)",
    "ste_perm_hard": "STE permutation (hard)",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "ste_perm_soft": {"color": "#059669", "linestyle": "-"},
    "ste_perm_hard": {"color": "#059669", "linestyle": "--"},
}


def _plot_variant_curves(output_path: str, variant_results: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metric_layout = [
        ("tr_loss", "Train Loss", axes[0, 0]),
        ("te_loss", "Test Loss", axes[0, 1]),
        ("tr_acc", "Train Accuracy", axes[1, 0]),
        ("te_acc", "Test Accuracy", axes[1, 1]),
    ]

    for variant_key, results in variant_results.items():
        style = VARIANT_STYLES[variant_key]
        for metric_name, title, axis in metric_layout:
            axis.plot(
                results["ts"],
                results[metric_name],
                label=VARIANT_DISPLAY_NAMES[variant_key],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
            axis.set_title(title)
            axis.set_xlabel("Interpolation t")
            axis.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.35), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _variant_row(
    *,
    variant_key: str,
    endpoint_a: Mapping[str, float],
    endpoint_b: Mapping[str, float],
    interpolation: Mapping[str, np.ndarray],
    scale_stats: Mapping[str, Any] | None,
    checkpoint_path: str,
    delta_vs_soft: Mapping[str, float | None] | None,
) -> Dict[str, Any]:
    barrier_metrics = compute_barrier_metrics(interpolation)
    per_layer_means = _per_layer_scale_means(scale_stats)
    overall_scale = scale_stats["overall"] if scale_stats else None

    row = {
        "variant_key": variant_key,
        "variant_name": VARIANT_DISPLAY_NAMES[variant_key],
        "checkpoint_path": checkpoint_path,
        "endpoint_a_train_loss": endpoint_a["train_loss"],
        "endpoint_a_train_acc": endpoint_a["train_acc"],
        "endpoint_a_test_loss": endpoint_a["test_loss"],
        "endpoint_a_test_acc": endpoint_a["test_acc"],
        "endpoint_b_train_loss": endpoint_b["train_loss"],
        "endpoint_b_train_acc": endpoint_b["train_acc"],
        "endpoint_b_test_loss": endpoint_b["test_loss"],
        "endpoint_b_test_acc": endpoint_b["test_acc"],
        "mean_train_interp_loss": float(np.mean(interpolation["tr_loss"])),
        "mean_test_interp_loss": float(np.mean(interpolation["te_loss"])),
        "raw_max_train_interp_loss": float(np.max(interpolation["tr_loss"])),
        "raw_max_test_interp_loss": float(np.max(interpolation["te_loss"])),
        "train_loss_barrier_avg": barrier_metrics["train_loss_barrier_avg"],
        "test_loss_barrier_avg": barrier_metrics["test_loss_barrier_avg"],
        "train_loss_barrier_max_endpoint": barrier_metrics["train_loss_barrier_max_endpoint"],
        "test_loss_barrier_max_endpoint": barrier_metrics["test_loss_barrier_max_endpoint"],
        "min_train_acc": barrier_metrics["min_train_acc"],
        "min_test_acc": barrier_metrics["min_test_acc"],
        "train_acc_drop_from_endpoint_min": barrier_metrics["train_acc_drop_from_endpoint_min"],
        "test_acc_drop_from_endpoint_min": barrier_metrics["test_acc_drop_from_endpoint_min"],
        "overall_scale_mean": None if overall_scale is None else overall_scale["mean"],
        "overall_scale_std": None if overall_scale is None else overall_scale["std"],
        "overall_scale_min": None if overall_scale is None else overall_scale["min"],
        "overall_scale_max": None if overall_scale is None else overall_scale["max"],
        "per_layer_scale_means_json": json.dumps(per_layer_means, sort_keys=True),
    }
    if delta_vs_soft:
        row.update(delta_vs_soft)
    else:
        row.update(
            {
                "delta_test_loss_barrier_avg_vs_soft": None,
                "delta_mean_test_interp_loss_vs_soft": None,
                "delta_min_test_acc_vs_soft": None,
            }
        )
    return row


def run_vgg16_ste_evaluation(
    *,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    output_root: str,
    dataset: str = "MNIST",
    data_path: str = "./data",
    num_eval_points: int = 21,
    evaluation_batch_size: int = 128,
    device: str | torch.device = "auto",
    num_workers: int = 4,
    max_eval_batches: int | None = None,
    plot_filename: str = "comparison.png",
) -> Dict[str, Any]:
    runtime_device = resolve_device(str(device))
    root = Path(output_root)
    evaluation_dir = ensure_dir(root / "evaluation")

    loaders, _ = load_dataset_eval_loaders(
        dataset=dataset,
        data_path=data_path,
        batch_size=evaluation_batch_size,
        num_workers=num_workers,
    )

    state_a = load_state_dict(model_a_checkpoint)
    raw_state_b = load_state_dict(model_b_checkpoint)
    endpoint_a_metrics = _evaluate_endpoint_metrics(state_a, loaders, device=runtime_device, max_eval_batches=max_eval_batches)
    raw_endpoint_b_metrics = _evaluate_endpoint_metrics(raw_state_b, loaders, device=runtime_device, max_eval_batches=max_eval_batches)

    method_dir = root / "ste_perm"
    artifact = load_alignment_artifact(str(method_dir / "alignment_artifacts.pt"))
    scale_stats = artifact.get("scale_statistics")

    variant_states: Dict[str, OrderedDict[str, torch.Tensor]] = {
        "no_alignment": OrderedDict((key, value.clone()) for key, value in raw_state_b.items()),
        "ste_perm_soft": load_state_dict(str(method_dir / "soft_aligned.pt")),
        "ste_perm_hard": load_state_dict(str(method_dir / "hard_aligned.pt")),
    }
    variant_scale_stats = {
        "no_alignment": None,
        "ste_perm_soft": scale_stats,
        "ste_perm_hard": scale_stats,
    }
    variant_checkpoint_paths = {
        "no_alignment": model_b_checkpoint,
        "ste_perm_soft": str(method_dir / "soft_aligned.pt"),
        "ste_perm_hard": str(method_dir / "hard_aligned.pt"),
    }

    variant_interpolations: Dict[str, Dict[str, np.ndarray]] = {}
    variant_endpoint_metrics: Dict[str, Dict[str, float]] = {}
    variant_rows: Dict[str, Dict[str, Any]] = {}

    for variant_key, variant_state in variant_states.items():
        endpoint_metrics = raw_endpoint_b_metrics if variant_key == "no_alignment" else _evaluate_endpoint_metrics(
            variant_state,
            loaders,
            device=runtime_device,
            max_eval_batches=max_eval_batches,
        )
        interpolation = _evaluate_linear_interpolation(
            state_a,
            variant_state,
            loaders,
            device=runtime_device,
            num_points=num_eval_points,
            max_eval_batches=max_eval_batches,
        )

        variant_endpoint_metrics[variant_key] = endpoint_metrics
        variant_interpolations[variant_key] = interpolation
        _save_interpolation_npz(str(evaluation_dir / f"{variant_key}.npz"), interpolation)

    soft_row = _variant_row(
        variant_key="ste_perm_soft",
        endpoint_a=endpoint_a_metrics,
        endpoint_b=variant_endpoint_metrics["ste_perm_soft"],
        interpolation=variant_interpolations["ste_perm_soft"],
        scale_stats=variant_scale_stats["ste_perm_soft"],
        checkpoint_path=variant_checkpoint_paths["ste_perm_soft"],
        delta_vs_soft=None,
    )
    variant_rows["no_alignment"] = _variant_row(
        variant_key="no_alignment",
        endpoint_a=endpoint_a_metrics,
        endpoint_b=variant_endpoint_metrics["no_alignment"],
        interpolation=variant_interpolations["no_alignment"],
        scale_stats=None,
        checkpoint_path=variant_checkpoint_paths["no_alignment"],
        delta_vs_soft=None,
    )
    variant_rows["ste_perm_soft"] = soft_row
    variant_rows["ste_perm_hard"] = _variant_row(
        variant_key="ste_perm_hard",
        endpoint_a=endpoint_a_metrics,
        endpoint_b=variant_endpoint_metrics["ste_perm_hard"],
        interpolation=variant_interpolations["ste_perm_hard"],
        scale_stats=variant_scale_stats["ste_perm_hard"],
        checkpoint_path=variant_checkpoint_paths["ste_perm_hard"],
        delta_vs_soft={
            "delta_test_loss_barrier_avg_vs_soft": (
                compute_barrier_metrics(variant_interpolations["ste_perm_hard"])["test_loss_barrier_avg"]
                - compute_barrier_metrics(variant_interpolations["ste_perm_soft"])["test_loss_barrier_avg"]
            ),
            "delta_mean_test_interp_loss_vs_soft": (
                float(np.mean(variant_interpolations["ste_perm_hard"]["te_loss"]))
                - float(np.mean(variant_interpolations["ste_perm_soft"]["te_loss"]))
            ),
            "delta_min_test_acc_vs_soft": (
                compute_barrier_metrics(variant_interpolations["ste_perm_hard"])["min_test_acc"]
                - compute_barrier_metrics(variant_interpolations["ste_perm_soft"])["min_test_acc"]
            ),
        },
    )

    ordered_rows = [variant_rows["no_alignment"], variant_rows["ste_perm_soft"], variant_rows["ste_perm_hard"]]
    write_summary_files(str(evaluation_dir), ordered_rows)

    plot_path = str(evaluation_dir / plot_filename)
    _plot_variant_curves(plot_path, variant_interpolations)

    summary = {
        "evaluation_dir": str(evaluation_dir),
        "rows": ordered_rows,
        "comparison_json": str(evaluation_dir / "comparison.json"),
        "comparison_csv": str(evaluation_dir / "comparison.csv"),
        "comparison_md": str(evaluation_dir / "comparison.md"),
        "plot_path": plot_path,
    }
    save_json(summary, Path(evaluation_dir) / "summary.json")
    return summary
