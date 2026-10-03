"""Shared helpers for the retained VGG/CIFAR Sinkhorn workflows.

This module owns the reusable model-loading, interpolation, scale-handling, and
alignment-helper logic used by the sweep, comparison, and verification runners.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, Dict

import numpy as np
import torch

from mode_connectivity.core.checkpoint import build_model_from_state_dict, load_state_dict
from mode_connectivity.evaluation.interpolation import evaluate_classifier
from mode_connectivity.alignment.permutation_pipeline import compute_paper_loss_barrier
from mode_connectivity.core.output import ensure_dir


def build_vgg_model(VGGClass, vgg_name: str, num_classes: int, image_size: int) -> torch.nn.Module:
    return VGGClass(vgg_name, in_channels=3, out_features=num_classes, h_in=image_size, w_in=image_size)


def clone_module_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_model_from_checkpoint(
    model_path: Path,
    *,
    model_factory,
    device: torch.device,
) -> torch.nn.Module:
    state_dict = load_state_dict(model_path, normalize_keys=True)
    model = build_model_from_state_dict(state_dict, model_factory=model_factory)
    model.to(device)
    model.eval()
    return model


def load_vgg_checkpoint_model(
    model_path: Path,
    VGGClass,
    *,
    vgg_name: str,
    image_size: int,
    device: torch.device,
) -> torch.nn.Module:
    return load_model_from_checkpoint(
        model_path,
        model_factory=lambda: build_vgg_model(VGGClass, vgg_name, num_classes=10, image_size=image_size),
        device=device,
    )


def evaluate_model(model: torch.nn.Module, dataset, criterion, device: torch.device) -> tuple[float, float]:
    results = evaluate_classifier(model, dataset, device, criterion=criterion)
    return float(results["loss"]), float(results["accuracy"]) / 100.0


def format_scale_stats(scale_stats: Dict[str, Any] | None) -> str:
    if not scale_stats:
        return "scale_stats=<disabled>"
    return (
        "scale[min={scale_min:.4f}, mean={scale_mean:.4f}, max={scale_max:.4f}] "
        "inv_scale[min={inv_scale_min:.4f}, mean={inv_scale_mean:.4f}, max={inv_scale_max:.4f}]"
    ).format(**scale_stats)


def extract_scale_artifacts(pi_model_a: torch.nn.Module) -> Dict[str, Any]:
    if not hasattr(pi_model_a, "u"):
        return {
            "raw_log_scales": [],
            "scales": [],
            "inv_scales": [],
            "layer_scale_stats": [],
        }

    raw_log_scales: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    inv_scales: list[torch.Tensor] = []
    layer_scale_stats: list[dict[str, float | int]] = []
    permutation_to_parameter_names: dict[int, list[str]] = {}
    if hasattr(pi_model_a, "reparamnet"):
        for parameter_name, permutation_graph_index in pi_model_a.reparamnet.map_param_index.items():
            permutation_index = pi_model_a.reparamnet.perm_dict[permutation_graph_index]
            if permutation_index is None:
                continue
            permutation_to_parameter_names.setdefault(int(permutation_index), []).append(parameter_name)

    for layer_index, log_scale in enumerate(pi_model_a.u):
        if log_scale is None:
            continue
        log_scale_cpu = log_scale.detach().cpu().clone()
        scale_cpu = torch.exp(log_scale_cpu)
        inv_scale_cpu = torch.exp(-log_scale_cpu)
        raw_log_scales.append(log_scale_cpu)
        scales.append(scale_cpu)
        inv_scales.append(inv_scale_cpu)
        layer_scale_stats.append(
            {
                "layer_index": int(layer_index),
                "parameter_names": sorted(permutation_to_parameter_names.get(int(layer_index), [])),
                "num_channels": int(scale_cpu.numel()),
                "log_scale_min": float(log_scale_cpu.min().item()),
                "log_scale_max": float(log_scale_cpu.max().item()),
                "log_scale_mean": float(log_scale_cpu.mean().item()),
                "scale_min": float(scale_cpu.min().item()),
                "scale_max": float(scale_cpu.max().item()),
                "scale_mean": float(scale_cpu.mean().item()),
                "inv_scale_min": float(inv_scale_cpu.min().item()),
                "inv_scale_max": float(inv_scale_cpu.max().item()),
                "inv_scale_mean": float(inv_scale_cpu.mean().item()),
            }
        )

    return {
        "raw_log_scales": raw_log_scales,
        "scales": scales,
        "inv_scales": inv_scales,
        "layer_scale_stats": layer_scale_stats,
    }


def maybe_load_starting_alignment(
    pi_model_a: torch.nn.Module,
    *,
    artifact_path: str | Path | None,
    permutation_kind: str,
    scale_invariant: bool,
) -> None:
    if artifact_path in (None, "", "null"):
        return

    resolved_path = Path(artifact_path)
    payload = torch.load(resolved_path, map_location="cpu")
    if permutation_kind == "hard":
        source = payload.get("hard_permutations")
    elif permutation_kind == "raw":
        source = payload.get("raw_parameters")
    else:
        raise ValueError(f"Unsupported starting_permutation_kind={permutation_kind!r}. Expected 'hard' or 'raw'.")

    if source is None:
        raise ValueError(f"Starting alignment artifact {resolved_path} does not contain {permutation_kind!r} permutation data.")

    target_params = [parameter for parameter in pi_model_a.p if parameter is not None]
    if len(source) != len(target_params):
        raise ValueError(
            f"Starting alignment artifact {resolved_path} has {len(source)} permutations but the current model expects {len(target_params)}."
        )

    with torch.no_grad():
        for target, source_value in zip(target_params, source):
            source_tensor = source_value if isinstance(source_value, torch.Tensor) else torch.as_tensor(source_value)
            target.data.copy_(source_tensor.to(device=target.device, dtype=target.dtype))

        if scale_invariant and hasattr(pi_model_a, "u"):
            raw_log_scales = payload.get("raw_log_scales")
            if raw_log_scales is not None:
                target_scale_params = [parameter for parameter in pi_model_a.u if parameter is not None]
                if len(raw_log_scales) != len(target_scale_params):
                    raise ValueError(
                        f"Starting alignment artifact {resolved_path} has {len(raw_log_scales)} raw_log_scales but "
                        f"the current model expects {len(target_scale_params)}."
                    )
                for target, source_value in zip(target_scale_params, raw_log_scales):
                    source_tensor = source_value if isinstance(source_value, torch.Tensor) else torch.as_tensor(source_value)
                    target.data.copy_(source_tensor.to(device=target.device, dtype=target.dtype))


def enable_fixed_hard_permutation_scale_only_mode(pi_model_a: torch.nn.Module, *, matching) -> None:
    fixed_hard_permutations: list[torch.Tensor | None] = []
    for parameter in pi_model_a.p:
        if parameter is None:
            fixed_hard_permutations.append(None)
            continue
        hard_perm = matching(parameter.detach().cpu().numpy()).to(pi_model_a.param_precision).to(parameter.device)
        fixed_hard_permutations.append(hard_perm)

    pi_model_a._fixed_hard_permutations = fixed_hard_permutations

    def forward_with_fixed_hard_permutations(self, x=None):
        gk = []
        for hard_perm, log_scale in zip(self._fixed_hard_permutations, self.u):
            if hard_perm is None:
                continue
            if self.scale_invariant:
                scale = torch.exp(log_scale)
                inv_scale = torch.exp(-log_scale)
                gk.append({"perm": hard_perm, "scale": scale, "inv_scale": inv_scale})
            else:
                gk.append(hard_perm)

        model = self.reparamnet(gk)
        if x is not None and x.ndim == 1:
            x.unsqueeze_(0)
        if x is not None:
            return model(x)
        return model

    pi_model_a.forward = MethodType(forward_with_fixed_hard_permutations, pi_model_a)


def configure_trainable_alignment_params(pi_model_a: torch.nn.Module, *, finetune_mode: str) -> None:
    if finetune_mode == "joint":
        for parameter in pi_model_a.p:
            if parameter is not None:
                parameter.requires_grad_(True)
        for parameter in getattr(pi_model_a, "u", []):
            if parameter is not None:
                parameter.requires_grad_(True)
        return

    if finetune_mode == "scale_only":
        if not hasattr(pi_model_a, "u") or not any(parameter is not None for parameter in pi_model_a.u):
            raise ValueError("finetune_mode='scale_only' requires scale_invariant=true so scale parameters exist.")
        for parameter in pi_model_a.p:
            if parameter is not None:
                parameter.requires_grad_(False)
        for parameter in pi_model_a.u:
            if parameter is not None:
                parameter.requires_grad_(True)
        return

    if finetune_mode == "scale_only_fixed_hard":
        if not hasattr(pi_model_a, "u") or not any(parameter is not None for parameter in pi_model_a.u):
            raise ValueError("finetune_mode='scale_only_fixed_hard' requires scale_invariant=true so scale parameters exist.")
        for parameter in pi_model_a.p:
            if parameter is not None:
                parameter.requires_grad_(False)
        for parameter in pi_model_a.u:
            if parameter is not None:
                parameter.requires_grad_(True)
        return

    raise ValueError(
        f"Unsupported finetune_mode={finetune_mode!r}. Expected 'joint', 'scale_only', or 'scale_only_fixed_hard'."
    )


def build_optimizer(pi_model_a: torch.nn.Module, *, learning_rate: float) -> torch.optim.Optimizer:
    trainable_params = [parameter for parameter in pi_model_a.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable alignment parameters were selected for optimization.")
    return torch.optim.AdamW(trainable_params, lr=float(learning_rate))


def build_alignment_criterion(loss_name: str, model_b: torch.nn.Module, MidLoss, RndLoss, DistL1Loss, DistL2Loss):
    if loss_name == "random":
        return RndLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    if loss_name == "midpoint":
        return MidLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    if loss_name == "dist_l1":
        return DistL1Loss(model_b)
    if loss_name == "dist_l2":
        return DistL2Loss(model_b)
    raise ValueError(f"Unsupported loss_name={loss_name!r}. Expected 'random', 'midpoint', 'dist_l1', or 'dist_l2'.")


def compute_curve_metrics(costs: list[float], accs: list[float], endpoint_a_loss: float, endpoint_b_loss: float) -> dict[str, float]:
    ts = torch.linspace(0.0, 1.0, len(costs)).tolist()
    paper_barrier = compute_paper_loss_barrier(np.asarray(costs, dtype=np.float64), np.asarray(ts, dtype=np.float64))
    return {
        "mean_test_interp_loss": float(sum(costs) / len(costs)),
        "test_loss_barrier_avg": paper_barrier,
        "test_loss_barrier_max_endpoint": paper_barrier,
        "min_test_acc": float(min(accs)),
        "endpoint_a_test_loss": float(endpoint_a_loss),
        "endpoint_b_test_loss": float(endpoint_b_loss),
    }


def build_variant_row(
    *,
    variant_key: str,
    display_name: str,
    val_costs: list[float],
    val_accs: list[float],
    test_costs: list[float],
    test_accs: list[float],
    endpoint_a_val_loss: float,
    endpoint_b_val_loss: float,
    endpoint_a_test_loss: float,
    endpoint_b_test_loss: float,
    endpoint_a_val_acc: float,
    endpoint_b_val_acc: float,
    endpoint_a_test_acc: float,
    endpoint_b_test_acc: float,
) -> dict[str, Any]:
    val_metrics = compute_curve_metrics(val_costs, val_accs, endpoint_a_val_loss, endpoint_b_val_loss)
    test_metrics = compute_curve_metrics(test_costs, test_accs, endpoint_a_test_loss, endpoint_b_test_loss)
    return {
        "variant_key": variant_key,
        "display_name": display_name,
        "endpoint_a_val_acc": float(endpoint_a_val_acc),
        "endpoint_b_val_acc": float(endpoint_b_val_acc),
        "endpoint_a_test_acc": float(endpoint_a_test_acc),
        "endpoint_b_test_acc": float(endpoint_b_test_acc),
        "mean_val_interp_loss": val_metrics["mean_test_interp_loss"],
        "val_loss_barrier_avg": val_metrics["test_loss_barrier_avg"],
        "val_loss_barrier_max_endpoint": val_metrics["test_loss_barrier_max_endpoint"],
        "min_val_acc": val_metrics["min_test_acc"],
        "endpoint_a_val_loss": val_metrics["endpoint_a_test_loss"],
        "endpoint_b_val_loss": val_metrics["endpoint_b_test_loss"],
        "mean_test_interp_loss": test_metrics["mean_test_interp_loss"],
        "test_loss_barrier_avg": test_metrics["test_loss_barrier_avg"],
        "test_loss_barrier_max_endpoint": test_metrics["test_loss_barrier_max_endpoint"],
        "min_test_acc": test_metrics["min_test_acc"],
        "endpoint_a_test_loss": test_metrics["endpoint_a_test_loss"],
        "endpoint_b_test_loss": test_metrics["endpoint_b_test_loss"],
    }


def evaluate_rebased_curve(
    *,
    model_left: torch.nn.Module,
    model_right: torch.nn.Module,
    loader,
    num_eval_points: int,
    eval_loss_acc,
    lerp,
    device: torch.device,
) -> dict[str, list[float]]:
    lambdas = torch.linspace(0, 1, int(num_eval_points))
    losses: list[float] = []
    accuracies: list[float] = []
    errors: list[float] = []
    for lam in lambdas.tolist():
        temporal_model = lerp(model_left, model_right, lam)
        loss_value, acc_value = eval_loss_acc(temporal_model, loader, torch.nn.CrossEntropyLoss(), device)
        acc_percent = float(acc_value) * 100.0
        losses.append(float(loss_value))
        accuracies.append(acc_percent)
        errors.append(100.0 - acc_percent)
    return {
        "lambdas": lambdas.tolist(),
        "losses": losses,
        "accuracies": accuracies,
        "errors": errors,
    }


def plot_three_way_curves(
    *,
    x: list[float],
    y_naive: list[float],
    y_perm: list[float],
    y_scale: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    show_legend: bool,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y_naive, label="No Alignment", color="tab:gray", linewidth=2.0)
    plt.plot(x, y_perm, label="Sinkhorn Permutation Only (From Scratch)", color="tab:orange", linewidth=2.0)
    plt.plot(x, y_scale, label="Sinkhorn Permutation + Scale (From Scratch)", color="tab:purple", linewidth=2.0)
    plt.xlabel("t (interpolation parameter)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    if show_legend:
        plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_three_way_curve_arrays(output_dir: Path, curves: dict[str, dict[str, list[float]]]) -> None:
    arrays_dir = ensure_dir(output_dir / "arrays")
    perm_bundle = {
        "lambdas": np.asarray(curves["test_naive"]["lambdas"], dtype=np.float64),
        "train_naive_losses": np.asarray(curves["train_naive"]["losses"], dtype=np.float64),
        "train_naive_accuracies": np.asarray(curves["train_naive"]["accuracies"], dtype=np.float64),
        "train_naive_errors": np.asarray(curves["train_naive"]["errors"], dtype=np.float64),
        "test_naive_losses": np.asarray(curves["test_naive"]["losses"], dtype=np.float64),
        "test_naive_accuracies": np.asarray(curves["test_naive"]["accuracies"], dtype=np.float64),
        "test_naive_errors": np.asarray(curves["test_naive"]["errors"], dtype=np.float64),
        "train_perm_losses": np.asarray(curves["train_perm"]["losses"], dtype=np.float64),
        "train_perm_accuracies": np.asarray(curves["train_perm"]["accuracies"], dtype=np.float64),
        "train_perm_errors": np.asarray(curves["train_perm"]["errors"], dtype=np.float64),
        "test_perm_losses": np.asarray(curves["test_perm"]["losses"], dtype=np.float64),
        "test_perm_accuracies": np.asarray(curves["test_perm"]["accuracies"], dtype=np.float64),
        "test_perm_errors": np.asarray(curves["test_perm"]["errors"], dtype=np.float64),
    }
    scale_bundle = {
        "lambdas": np.asarray(curves["test_naive"]["lambdas"], dtype=np.float64),
        "train_naive_losses": np.asarray(curves["train_naive"]["losses"], dtype=np.float64),
        "train_naive_accuracies": np.asarray(curves["train_naive"]["accuracies"], dtype=np.float64),
        "train_naive_errors": np.asarray(curves["train_naive"]["errors"], dtype=np.float64),
        "test_naive_losses": np.asarray(curves["test_naive"]["losses"], dtype=np.float64),
        "test_naive_accuracies": np.asarray(curves["test_naive"]["accuracies"], dtype=np.float64),
        "test_naive_errors": np.asarray(curves["test_naive"]["errors"], dtype=np.float64),
        "train_scale_losses": np.asarray(curves["train_scale"]["losses"], dtype=np.float64),
        "train_scale_accuracies": np.asarray(curves["train_scale"]["accuracies"], dtype=np.float64),
        "train_scale_errors": np.asarray(curves["train_scale"]["errors"], dtype=np.float64),
        "test_scale_losses": np.asarray(curves["test_scale"]["losses"], dtype=np.float64),
        "test_scale_accuracies": np.asarray(curves["test_scale"]["accuracies"], dtype=np.float64),
        "test_scale_errors": np.asarray(curves["test_scale"]["errors"], dtype=np.float64),
    }
    np.save(arrays_dir / "perm_bundle.npy", perm_bundle, allow_pickle=True)
    np.save(arrays_dir / "scale_bundle.npy", scale_bundle, allow_pickle=True)


def evaluate_interpolation_results(
    model_left: torch.nn.Module,
    model_right: torch.nn.Module,
    *,
    train_loader,
    test_loader,
    lerp,
    eval_loss_acc,
    device: torch.device,
    num_eval_points: int,
) -> Dict[str, np.ndarray]:
    lambdas = np.linspace(0.0, 1.0, int(num_eval_points), dtype=np.float64)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for lam in lambdas.tolist():
        temporal_model = lerp(model_left, model_right, float(lam))
        loss_train, acc_train = eval_loss_acc(temporal_model, train_loader, torch.nn.CrossEntropyLoss(), device)
        loss_test, acc_test = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        train_loss.append(float(loss_train))
        train_acc.append(float(acc_train) * 100.0)
        test_loss.append(float(loss_test))
        test_acc.append(float(acc_test) * 100.0)

    return {
        "ts": lambdas,
        "tr_loss": np.asarray(train_loss, dtype=np.float64),
        "tr_acc": np.asarray(train_acc, dtype=np.float64),
        "tr_err": 100.0 - np.asarray(train_acc, dtype=np.float64),
        "te_loss": np.asarray(test_loss, dtype=np.float64),
        "te_acc": np.asarray(test_acc, dtype=np.float64),
        "te_err": 100.0 - np.asarray(test_acc, dtype=np.float64),
    }


def save_interpolation_npz(output_path: Path, results: Dict[str, np.ndarray]) -> None:
    train_loss_barrier = compute_paper_loss_barrier(results["tr_loss"], results["ts"])
    test_loss_barrier = compute_paper_loss_barrier(results["te_loss"], results["ts"])
    np.savez(
        output_path,
        ts=results["ts"],
        tr_loss=results["tr_loss"],
        tr_acc=results["tr_acc"],
        tr_err=results["tr_err"],
        te_loss=results["te_loss"],
        te_acc=results["te_acc"],
        te_err=results["te_err"],
        train_loss_barrier_avg=train_loss_barrier,
        test_loss_barrier_avg=test_loss_barrier,
        train_loss_barrier_max_endpoint=train_loss_barrier,
        test_loss_barrier_max_endpoint=test_loss_barrier,
        min_train_acc=float(np.min(results["tr_acc"])),
        min_test_acc=float(np.min(results["te_acc"])),
    )
