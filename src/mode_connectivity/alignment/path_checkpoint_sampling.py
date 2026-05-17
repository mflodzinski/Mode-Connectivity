"""Helpers for sampling endpoint-like checkpoints from trained curves.

This module reconstructs individual parameter states from saved curve models so
they can be reused in downstream alignment and evaluation workflows.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from mode_connectivity.core.checkpoint import load_state_dict
from mode_connectivity.core.models import create_curve_model, create_model, get_architecture
from mode_connectivity.core.setup import add_external_path


def curve_state_dict_at_t(
    curve_model: torch.nn.Module,
    *,
    t: float,
    model_name: str,
    num_classes: int = 10,
    device: str = "cpu",
) -> OrderedDict:
    """Reconstruct a base-model state dict at position ``t`` on the curve."""

    architecture = get_architecture(model_name)
    base_model = create_model(architecture, num_classes=num_classes, device=torch.device(device))
    parameter_values = _curve_parameters_at_t(curve_model, t=t, device=device)

    state_dict = base_model.state_dict()
    parameter_names = {name for name, _ in base_model.named_parameters()}
    parameter_iter = iter(parameter_values)
    reconstructed = OrderedDict()

    for name, tensor in state_dict.items():
        if name in parameter_names:
            reconstructed[name] = next(parameter_iter).clone()
        else:
            reconstructed[name] = tensor.detach().cpu().clone()

    try:
        next(parameter_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("Curve parameter reconstruction produced more tensors than the base model expects.")

    return reconstructed


def load_curve_model(
    curve_checkpoint_path: str,
    *,
    model_name: str,
    curve_type: str,
    num_bends: int,
    num_classes: int = 10,
    device: str = "cpu",
) -> torch.nn.Module:
    """Instantiate a curve model and load the saved curve checkpoint into it."""

    add_external_path()
    architecture = get_architecture(model_name)
    curve_model = create_curve_model(
        architecture,
        num_classes=num_classes,
        curve_type=curve_type,
        num_bends=num_bends,
        device=torch.device(device),
    )
    checkpoint = torch.load(curve_checkpoint_path, map_location=device)
    curve_model.load_state_dict(_extract_model_state(checkpoint))
    curve_model.eval()
    return curve_model


def extract_curve_control_point_state_dicts(
    curve_checkpoint_path: str,
    *,
    model_name: str,
    curve_type: str,
    num_bends: int,
    num_classes: int = 10,
    device: str = "cpu",
) -> List[OrderedDict]:
    """Export each curve control point into a base-model state dict."""

    curve_model = load_curve_model(
        curve_checkpoint_path,
        model_name=model_name,
        curve_type=curve_type,
        num_bends=num_bends,
        num_classes=num_classes,
        device=device,
    )

    architecture = get_architecture(model_name)
    control_points = []
    for index in range(num_bends):
        base_model = create_model(architecture, num_classes=num_classes, device=torch.device(device))
        curve_model.export_base_parameters(base_model, index)
        control_points.append(
            OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in base_model.state_dict().items())
        )
    return control_points


def sample_curve_checkpoints(
    curve_checkpoint_path: str,
    *,
    output_dir: str,
    ts: Sequence[float],
    model_name: str,
    curve_type: str,
    num_bends: int,
    num_classes: int = 10,
    device: str = "cpu",
    source_metadata: Dict | None = None,
) -> List[str]:
    """Sample and save base checkpoints along the trained curve."""

    curve_model = load_curve_model(
        curve_checkpoint_path,
        model_name=model_name,
        curve_type=curve_type,
        num_bends=num_bends,
        num_classes=num_classes,
        device=device,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for index, t in enumerate(ts):
        checkpoint_path = output_path / f"C{index}.pt"
        state_dict = curve_state_dict_at_t(
            curve_model,
            t=t,
            model_name=model_name,
            num_classes=num_classes,
            device=device,
        )
        torch.save(
            {
                "model_state": state_dict,
                "sample_index": index,
                "curve_t": float(t),
                "source_curve_checkpoint": os.path.abspath(curve_checkpoint_path),
                "metadata": source_metadata or {},
            },
            checkpoint_path,
        )
        saved_paths.append(str(checkpoint_path))

    return saved_paths


def validate_endpoint_samples(
    sampled_state_dicts: Sequence[Dict[str, torch.Tensor]],
    *,
    endpoint_a_path: str,
    endpoint_b_path: str,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Ensure the sampled curve endpoints match the original endpoint checkpoints."""

    if len(sampled_state_dicts) < 2:
        raise ValueError("At least two sampled checkpoints are required to validate endpoints.")

    endpoint_a = load_state_dict(endpoint_a_path)
    endpoint_b = load_state_dict(endpoint_b_path)

    if not state_dicts_allclose(sampled_state_dicts[0], endpoint_a, atol=atol, rtol=rtol):
        raise ValueError("Sampled checkpoint C0 does not match endpoint A.")
    if not state_dicts_allclose(sampled_state_dicts[-1], endpoint_b, atol=atol, rtol=rtol):
        raise ValueError("Sampled checkpoint C4 does not match endpoint B.")


def load_sampled_state_dicts(checkpoint_paths: Iterable[str]) -> List[Dict[str, torch.Tensor]]:
    """Load sampled checkpoints in order and return their state dicts."""

    return [load_state_dict(path) for path in checkpoint_paths]


def state_dicts_allclose(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> bool:
    """Return ``True`` if every tensor in two state dicts matches within tolerance."""

    if set(state_a.keys()) != set(state_b.keys()):
        return False
    for key in state_a:
        if not torch.allclose(state_a[key], state_b[key], atol=atol, rtol=rtol):
            return False
    return True


def _curve_parameters_at_t(curve_model: torch.nn.Module, *, t: float, device: str) -> List[torch.Tensor]:
    with torch.no_grad():
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
        coeffs_t = curve_model.coeff_layer(t_tensor)
        parameters = []
        for module in curve_model.curve_modules:
            for weight in module.compute_weights_t(coeffs_t):
                if weight is not None:
                    parameters.append(weight.detach().cpu())
        return parameters


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint
