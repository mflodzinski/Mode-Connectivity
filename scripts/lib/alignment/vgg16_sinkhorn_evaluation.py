"""Evaluation utilities for the VGG16 Sinkhorn+scale alignment prototype."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

project_root = Path(__file__).resolve().parents[3]
os.environ["MPLCONFIGDIR"] = str(project_root / ".mplcache")
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.lib.analysis.alignment import create_vgg16_model, load_dataset_eval_loaders
from scripts.lib.alignment.permutation_pipeline import compute_barrier_metrics, write_summary_files
from scripts.lib.alignment.vgg16_sinkhorn_alignment import load_alignment_artifact
from scripts.lib.core.output import ResultSaver, ensure_dir, save_json
from scripts.lib.core.checkpoint import load_state_dict


VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "sinkhorn_perm_soft": "Sinkhorn permutation-only (soft)",
    "sinkhorn_perm_hard": "Sinkhorn permutation-only (hard)",
    "sinkhorn_scale_soft": "Sinkhorn + scale (soft)",
    "sinkhorn_scale_hard": "Sinkhorn + scale (hard)",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "sinkhorn_perm_soft": {"color": "#2563eb", "linestyle": "-"},
    "sinkhorn_perm_hard": {"color": "#2563eb", "linestyle": "--"},
    "sinkhorn_scale_soft": {"color": "#dc2626", "linestyle": "-"},
    "sinkhorn_scale_hard": {"color": "#dc2626", "linestyle": "--"},
}


def _evaluate_loaded_model(model, loader, *, device: torch.device, max_batches: int | None = None) -> Dict[str, float]:
    """Evaluate a VGG16 model already loaded with the desired state dict."""

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

    if total == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def _evaluate_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    loader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Evaluate one VGG16 state dict on one loader."""

    model = create_vgg16_model(num_classes=10, device=device)
    model.load_state_dict(OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items()))
    model.eval()
    return _evaluate_loaded_model(model, loader, device=device, max_batches=max_batches)


def _evaluate_endpoint_metrics(
    state_dict: Mapping[str, torch.Tensor],
    loaders,
    *,
    device: torch.device,
    max_eval_batches: int | None,
) -> Dict[str, float]:
    train_metrics = _evaluate_state_dict(state_dict, loaders["train"], device=device, max_batches=max_eval_batches)
    test_metrics = _evaluate_state_dict(state_dict, loaders["test"], device=device, max_batches=max_eval_batches)
    return {
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
    }


def _evaluate_linear_interpolation(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    loaders,
    *,
    device: torch.device,
    num_points: int,
    max_eval_batches: int | None,
) -> Dict[str, np.ndarray]:
    """Evaluate full train/test interpolation curves between two endpoint states."""

    model = create_vgg16_model(num_classes=10, device=device)
    ts = np.linspace(0.0, 1.0, num_points)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for t in ts:
        interpolated_state = OrderedDict(
            (key, ((1.0 - t) * state_a[key] + t * state_b[key]).detach().cpu()) for key in state_a
        )
        model.load_state_dict(interpolated_state)
        model.eval()

        train_metrics = _evaluate_loaded_model(model, loaders["train"], device=device, max_batches=max_eval_batches)
        test_metrics = _evaluate_loaded_model(model, loaders["test"], device=device, max_batches=max_eval_batches)

        train_loss.append(train_metrics["loss"])
        train_acc.append(train_metrics["accuracy"])
        test_loss.append(test_metrics["loss"])
        test_acc.append(test_metrics["accuracy"])

    return {
        "ts": ts,
        "tr_loss": np.asarray(train_loss, dtype=np.float64),
        "tr_acc": np.asarray(train_acc, dtype=np.float64),
        "tr_err": 100.0 - np.asarray(train_acc, dtype=np.float64),
        "te_loss": np.asarray(test_loss, dtype=np.float64),
        "te_acc": np.asarray(test_acc, dtype=np.float64),
        "te_err": 100.0 - np.asarray(test_acc, dtype=np.float64),
    }


def _save_interpolation_npz(path: str, results: Mapping[str, np.ndarray]) -> None:
    ResultSaver.save_standard(
        path,
        results["ts"],
        {"loss": results["tr_loss"], "acc": results["tr_acc"], "err": results["tr_err"]},
        {"loss": results["te_loss"], "acc": results["te_acc"], "err": results["te_err"]},
    )


def _per_layer_scale_means(scale_stats: Mapping[str, Any] | None) -> Dict[str, float]:
    if not scale_stats:
        return {}
    return {name: layer_stats["mean"] for name, layer_stats in scale_stats["per_layer"].items()}


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


def run_vgg16_alignment_evaluation(
    *,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    output_root: str,
    methods: Sequence[str],
    dataset: str = "CIFAR10",
    data_path: str = "./data",
    num_eval_points: int = 21,
    evaluation_batch_size: int = 128,
    device: str | torch.device = "auto",
    num_workers: int = 4,
    max_eval_batches: int | None = None,
    plot_filename: str = "comparison.png",
) -> Dict[str, Any]:
    """Evaluate interpolation curves for the trained alignment outputs."""

    runtime_device = torch.device(device) if isinstance(device, str) and device != "auto" else None
    if runtime_device is None:
        from scripts.lib.alignment.permutation_pipeline import resolve_device

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

    variant_states: Dict[str, OrderedDict[str, torch.Tensor]] = {
        "no_alignment": OrderedDict((key, value.clone()) for key, value in raw_state_b.items())
    }
    variant_scale_stats: Dict[str, Mapping[str, Any] | None] = {"no_alignment": None}
    variant_checkpoint_paths: Dict[str, str] = {"no_alignment": model_b_checkpoint}

    for method in methods:
        method_dir = root / method
        artifact = load_alignment_artifact(str(method_dir / "alignment_artifacts.pt"))
        scale_stats = artifact.get("scale_statistics")

        if method == "perm_only":
            soft_key = "sinkhorn_perm_soft"
            hard_key = "sinkhorn_perm_hard"
        elif method == "perm_scale":
            soft_key = "sinkhorn_scale_soft"
            hard_key = "sinkhorn_scale_hard"
        else:
            raise ValueError(f"Unexpected method '{method}' while building evaluation variants.")

        soft_path = str(method_dir / "soft_aligned.pt")
        hard_path = str(method_dir / "hard_aligned.pt")
        variant_states[soft_key] = load_state_dict(soft_path)
        variant_states[hard_key] = load_state_dict(hard_path)
        variant_scale_stats[soft_key] = scale_stats
        variant_scale_stats[hard_key] = scale_stats
        variant_checkpoint_paths[soft_key] = soft_path
        variant_checkpoint_paths[hard_key] = hard_path

    variant_interpolations: Dict[str, Dict[str, np.ndarray]] = {}
    variant_endpoint_metrics: Dict[str, Dict[str, float]] = {}
    variant_rows: Dict[str, Dict[str, Any]] = {}

    for variant_key, state_b in variant_states.items():
        variant_dir = ensure_dir(evaluation_dir / variant_key)
        interpolation = _evaluate_linear_interpolation(
            state_a,
            state_b,
            loaders,
            device=runtime_device,
            num_points=num_eval_points,
            max_eval_batches=max_eval_batches,
        )
        interpolation_path = str(Path(variant_dir) / "interpolation.npz")
        _save_interpolation_npz(interpolation_path, interpolation)

        endpoint_b_metrics = (
            raw_endpoint_b_metrics
            if variant_key == "no_alignment"
            else _evaluate_endpoint_metrics(state_b, loaders, device=runtime_device, max_eval_batches=max_eval_batches)
        )

        variant_interpolations[variant_key] = interpolation
        variant_endpoint_metrics[variant_key] = endpoint_b_metrics
        variant_rows[variant_key] = _variant_row(
            variant_key=variant_key,
            endpoint_a=endpoint_a_metrics,
            endpoint_b=endpoint_b_metrics,
            interpolation=interpolation,
            scale_stats=variant_scale_stats[variant_key],
            checkpoint_path=variant_checkpoint_paths[variant_key],
            delta_vs_soft=None,
        )

    soft_hard_pairs = [
        ("sinkhorn_perm_soft", "sinkhorn_perm_hard"),
        ("sinkhorn_scale_soft", "sinkhorn_scale_hard"),
    ]
    for soft_key, hard_key in soft_hard_pairs:
        if soft_key in variant_rows and hard_key in variant_rows:
            variant_rows[hard_key]["delta_test_loss_barrier_avg_vs_soft"] = (
                variant_rows[hard_key]["test_loss_barrier_avg"] - variant_rows[soft_key]["test_loss_barrier_avg"]
            )
            variant_rows[hard_key]["delta_mean_test_interp_loss_vs_soft"] = (
                variant_rows[hard_key]["mean_test_interp_loss"] - variant_rows[soft_key]["mean_test_interp_loss"]
            )
            variant_rows[hard_key]["delta_min_test_acc_vs_soft"] = (
                variant_rows[hard_key]["min_test_acc"] - variant_rows[soft_key]["min_test_acc"]
            )

    ordered_variant_keys = [
        "no_alignment",
        "sinkhorn_perm_soft",
        "sinkhorn_perm_hard",
        "sinkhorn_scale_soft",
        "sinkhorn_scale_hard",
    ]
    rows = [variant_rows[key] for key in ordered_variant_keys if key in variant_rows]
    write_summary_files(str(evaluation_dir), rows)

    full_summary = {
        "endpoint_metrics": {
            "model_a": endpoint_a_metrics,
            "model_b_raw": raw_endpoint_b_metrics,
        },
        "variants": {
            key: {
                "display_name": VARIANT_DISPLAY_NAMES[key],
                "checkpoint_path": variant_checkpoint_paths[key],
                "endpoint_b_metrics": variant_endpoint_metrics[key],
                "interpolation_metrics": variant_rows[key],
                "scale_statistics": variant_scale_stats[key],
                "interpolation_path": str(evaluation_dir / key / "interpolation.npz"),
            }
            for key in variant_rows
        },
    }
    save_json(full_summary, evaluation_dir / "full_summary.json")

    plot_path = str(Path(evaluation_dir) / plot_filename)
    _plot_variant_curves(plot_path, variant_interpolations)

    return {
        "evaluation_dir": str(evaluation_dir),
        "summary_path": str(evaluation_dir / "full_summary.json"),
        "plot_path": plot_path,
        "variant_rows": rows,
    }
