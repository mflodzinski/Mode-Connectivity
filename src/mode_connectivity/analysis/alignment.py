"""Alignment-analysis helpers built on the canonical evaluation modules.

The functions here provide reusable building blocks for loading models,
evaluating barriers, and converting permutation representations in analyses.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from mode_connectivity.core import data as core_data
from mode_connectivity.core.checkpoint import build_model_from_state_dict, load_checkpoint_state
from mode_connectivity.core.setup import add_external_path
from mode_connectivity.evaluation.interpolation import evaluate_classifier, evaluate_linear_interpolation
from mode_connectivity.evaluation.metrics import max_absolute_difference, state_distance_summary


def _get_external_models():
    add_external_path()
    import models as model_module  # type: ignore

    return model_module


def create_vgg16_model(num_classes: int = 10, device: torch.device | None = None):
    model_module = _get_external_models()
    model = model_module.VGG16.base(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


def load_vgg16_model(
    checkpoint_path: str,
    num_classes: int = 10,
    map_location: str | torch.device = "cpu",
):
    state_dict, checkpoint_family = load_checkpoint_state(checkpoint_path, map_location=map_location)
    model = build_model_from_state_dict(state_dict, checkpoint_family=checkpoint_family, num_classes=num_classes)
    return model


def load_dataset_eval_loaders(
    dataset: str = "CIFAR10",
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
):
    return core_data.get_loaders(
        dataset=dataset,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name="VGG",
        use_test=True,
        shuffle_train=False,
    )


def load_cifar10_eval_loaders(
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
):
    return load_dataset_eval_loaders(
        dataset="CIFAR10",
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def evaluate_model(model, loader, device: torch.device) -> Dict[str, float]:
    return evaluate_classifier(model, loader, device)


def evaluate_barrier(
    model_a,
    model_b,
    loaders,
    num_points: int = 11,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = model_a.to(device)
    model_b = model_b.to(device)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    interp_model = create_vgg16_model(num_classes=num_classes, device=device)
    results = evaluate_linear_interpolation(
        state_a=state_a,
        state_b=state_b,
        model=interp_model,
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        device=device,
        ts=np.linspace(0, 1, num_points),
    )

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
    return state_distance_summary(state_a, state_b)


def state_dict_to_perm_params(state_dict: Dict[str, torch.Tensor], perm_spec) -> Dict[str, torch.Tensor]:
    return {key: state_dict[key] for key in perm_spec.axes_to_perm if key in state_dict}


def convert_perm_to_apply_format(perm: Dict[str, Any]) -> Dict[str, Any]:
    converted = {}
    for key, val in perm.items():
        if key.startswith("P_Conv_"):
            converted[f"conv_{key[7:]}"] = val
        elif key.startswith("P_Dense_"):
            converted[f"fc_{key[8:]}"] = val
        else:
            converted[key] = val
    return converted


def convert_perm_to_compare_format(perm: Dict[str, Any]) -> Dict[str, Any]:
    converted = {}
    for key, val in perm.items():
        if key.startswith("conv_"):
            converted[f"P_Conv_{key[5:]}"] = val
        elif key.startswith("fc_"):
            converted[f"P_Dense_{key[3:]}"] = val
        else:
            converted[key] = val
    return converted


def max_abs_state_diff(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    return max_absolute_difference(state_a, state_b)
