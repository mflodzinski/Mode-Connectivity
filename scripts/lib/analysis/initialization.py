"""Shared utilities for initialization experiment analysis.

These helpers are used by scripts that analyze middle control points for
different curve initialization methods.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import torch


def _get_external_modules():
    """Import external dnn-mode-connectivity modules lazily."""
    from lib.core.setup import add_external_path

    add_external_path()
    import curves as curve_module  # type: ignore
    import models as model_module  # type: ignore

    return curve_module, model_module


def load_endpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load endpoint model state dict from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["model_state"]


def create_bezier_curve_model(
    num_classes: int = 10,
    architecture: str = "VGG16",
    num_bends: int = 3,
) -> Tuple[torch.nn.Module, Any]:
    """Create a Bezier curve model and return model + architecture metadata."""
    curve_module, model_module = _get_external_modules()

    arch = getattr(model_module, architecture)
    model = curve_module.CurveNet(
        num_classes=num_classes,
        curve=curve_module.Bezier,
        architecture=arch.curve,
        num_bends=num_bends,
        fix_start=True,
        fix_end=True,
        architecture_kwargs=arch.kwargs,
    )
    return model, arch


def import_endpoints_into_curve(
    model: torch.nn.Module,
    architecture: Any,
    endpoint0_state: Dict[str, torch.Tensor],
    endpoint1_state: Dict[str, torch.Tensor],
    num_classes: int = 10,
):
    """Import endpoint states into the first and last curve bends."""
    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    base_model.load_state_dict(endpoint0_state)
    model.import_base_parameters(base_model, index=0)

    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    base_model.load_state_dict(endpoint1_state)
    model.import_base_parameters(base_model, index=2)


def initialize_middle_point(
    model: torch.nn.Module,
    init_method: str,
    init_params: Dict[str, Any],
):
    """Initialize middle control point according to chosen method."""
    if init_method == "linear":
        model.init_linear()
    elif init_method == "biased":
        alpha = init_params.get("alpha", 0.5)
        model.init_linear_custom(alpha=alpha)
    elif init_method == "perturbed":
        alpha = init_params.get("alpha", 0.5)
        noise = init_params.get("noise", 0.01)
        model.init_perturbed_linear(alpha=alpha, noise_scale=noise)
    elif init_method == "sphere":
        alpha = init_params.get("alpha", 0.5)
        noise = init_params.get("noise", 0.01)
        inside = init_params.get("inside", True)
        model.init_sphere_constrained(alpha=alpha, noise_scale=noise, inside=inside)
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")


def build_initialized_curve_model(
    endpoint0_path: str,
    endpoint1_path: str,
    init_method: str,
    init_params: Dict[str, Any],
    seed: int = 1,
    num_classes: int = 10,
    architecture: str = "VGG16",
) -> torch.nn.Module:
    """Create a curve model with endpoints imported and middle point initialized."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    endpoint0_state = load_endpoint_state(endpoint0_path)
    endpoint1_state = load_endpoint_state(endpoint1_path)

    model, arch = create_bezier_curve_model(
        num_classes=num_classes,
        architecture=architecture,
        num_bends=3,
    )
    import_endpoints_into_curve(
        model,
        arch,
        endpoint0_state,
        endpoint1_state,
        num_classes=num_classes,
    )
    initialize_middle_point(model, init_method, init_params)
    return model


def extract_middle_point(model: torch.nn.Module) -> torch.Tensor:
    """Extract flattened middle bend parameters from a curve model."""
    middle_params = []
    for name, param in model.named_parameters():
        if "_1" in name:
            middle_params.append(param.data.flatten())
    return torch.cat(middle_params)


def extract_middle_point_from_checkpoint(checkpoint_path: str) -> torch.Tensor:
    """Extract flattened middle bend parameters directly from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state"]

    middle_params = []
    for key in sorted(model_state.keys()):
        if key.endswith("_1"):
            middle_params.append(model_state[key].flatten())
    return torch.cat(middle_params)


def compute_interpolated_l2_norm(model: torch.nn.Module, t: float = 0.5) -> float:
    """Compute L2 norm of interpolated weights at scalar t."""
    t_tensor = torch.FloatTensor([t])
    weights = model.weights(t_tensor)
    return float(np.sqrt(np.sum(np.square(weights))))


def calculate_l2_distance(point1: torch.Tensor, point2: torch.Tensor) -> float:
    """Calculate L2 distance between two vectors."""
    return torch.norm(point1 - point2).item()


def default_initialization_experiments() -> Dict[str, Dict[str, Any]]:
    """Canonical initialization experiment configuration map."""
    return {
        "alpha0.75": {
            "method": "biased",
            "params": {"alpha": 0.75},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "biased_linear/alpha_0.75/checkpoints/checkpoint-100.pt"
            ),
        },
        "alpha0.9": {
            "method": "biased",
            "params": {"alpha": 0.9},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "biased_linear/alpha_0.9/checkpoints/checkpoint-100.pt"
            ),
        },
        "perturbed_small": {
            "method": "perturbed",
            "params": {"alpha": 0.5, "noise": 0.01},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "perturbed/noise_0.01/checkpoints/checkpoint-100.pt"
            ),
        },
        "perturbed_large": {
            "method": "perturbed",
            "params": {"alpha": 0.5, "noise": 0.1},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "perturbed/noise_0.1/checkpoints/checkpoint-100.pt"
            ),
        },
        "sphere_inside": {
            "method": "sphere",
            "params": {"alpha": 0.5, "noise": 0.01, "inside": True},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "sphere_constrained/inside/checkpoints/checkpoint-100.pt"
            ),
        },
        "sphere_outside": {
            "method": "sphere",
            "params": {"alpha": 0.5, "noise": 0.01, "inside": False},
            "checkpoint": (
                "results/vgg16/cifar10/curves/initialization/"
                "sphere_constrained/outside/checkpoints/checkpoint-100.pt"
            ),
        },
    }
