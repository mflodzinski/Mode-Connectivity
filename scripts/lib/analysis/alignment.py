"""Shared alignment analysis utilities.

Consolidates repeated model/barrier evaluation helpers used by alignment
scripts.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _get_external_modules():
    """Import external dnn-mode-connectivity modules lazily."""
    from lib.core.setup import add_external_path

    add_external_path()
    import models as model_module  # type: ignore
    import data as data_module  # type: ignore

    return model_module, data_module


def create_vgg16_model(num_classes: int = 10, device: torch.device | None = None):
    """Create a VGG16 base model."""
    model_module, _ = _get_external_modules()
    model = model_module.VGG16.base(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


def load_vgg16_model(
    checkpoint_path: str,
    num_classes: int = 10,
    map_location: str | torch.device = "cpu",
):
    """Load VGG16 model from checkpoint."""
    model = create_vgg16_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_cifar10_eval_loaders(
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
):
    """Load CIFAR10 eval-mode loaders used by alignment workflows."""
    _, data_module = _get_external_modules()
    return data_module.loaders(
        "CIFAR10",
        data_path,
        batch_size,
        num_workers=num_workers,
        transform_name="VGG",
        use_test=True,
        eval_mode=True,
    )


def evaluate_model(model, loader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction="sum")
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def evaluate_barrier(
    model_a,
    model_b,
    loaders,
    num_points: int = 11,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Evaluate linear interpolation barrier between two models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = model_a.to(device)
    model_b = model_b.to(device)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    interp_model = create_vgg16_model(num_classes=num_classes, device=device)

    ts = np.linspace(0, 1, num_points)
    results = {
        "t": ts.tolist(),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for t in ts:
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        train_res = evaluate_model(interp_model, loaders["train"], device)
        test_res = evaluate_model(interp_model, loaders["test"], device)

        results["train_loss"].append(train_res["loss"])
        results["train_acc"].append(train_res["accuracy"])
        results["test_loss"].append(test_res["loss"])
        results["test_acc"].append(test_res["accuracy"])

    endpoint_test_loss = (results["test_loss"][0] + results["test_loss"][-1]) / 2
    max_test_loss = max(results["test_loss"])
    min_test_acc = min(results["test_acc"])

    results["barrier"] = max_test_loss - endpoint_test_loss
    results["max_test_loss"] = max_test_loss
    results["min_test_acc"] = min_test_acc
    results["endpoint_avg_test_loss"] = endpoint_test_loss
    return results


def compute_state_dict_l2_distance(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute L2 statistics between two state dicts."""
    total_diff_sq = 0.0
    total_params = 0
    for key in state_a:
        diff = state_a[key].float() - state_b[key].float()
        total_diff_sq += (diff ** 2).sum().item()
        total_params += diff.numel()

    l2_dist = total_diff_sq ** 0.5
    rms_diff = (total_diff_sq / total_params) ** 0.5
    return {
        "l2_distance": l2_dist,
        "num_params": total_params,
        "rms_difference": rms_diff,
    }


def state_dict_to_perm_params(state_dict: Dict[str, torch.Tensor], perm_spec) -> Dict[str, torch.Tensor]:
    """Convert state dict to params matching permutation spec axes keys."""
    params = {}
    for key in perm_spec.axes_to_perm:
        if key in state_dict:
            params[key] = state_dict[key]
    return params


def convert_perm_to_apply_format(perm: Dict[str, Any]) -> Dict[str, Any]:
    """Convert keys from weight-matching format to random-permutation format."""
    converted = {}
    for key, val in perm.items():
        if key.startswith("P_Conv_"):
            new_key = f"conv_{key[7:]}"
        elif key.startswith("P_Dense_"):
            new_key = f"fc_{key[8:]}"
        else:
            new_key = key
        converted[new_key] = val
    return converted


def convert_perm_to_compare_format(perm: Dict[str, Any]) -> Dict[str, Any]:
    """Convert keys from random-permutation format to weight-matching format."""
    converted = {}
    for key, val in perm.items():
        if key.startswith("conv_"):
            new_key = f"P_Conv_{key[5:]}"
        elif key.startswith("fc_"):
            new_key = f"P_Dense_{key[3:]}"
        else:
            new_key = key
        converted[new_key] = val
    return converted


def max_abs_state_diff(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> float:
    """Return maximum absolute elementwise difference across state dicts."""
    max_diff = 0.0
    for key in state_a:
        diff = torch.abs(state_a[key] - state_b[key]).max().item()
        max_diff = max(max_diff, diff)
    return max_diff
