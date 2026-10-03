"""Dataset-specific models expressed through one state-dictionary interface."""

from __future__ import annotations

import torch
from torch.func import functional_call

from mode_connectivity.alignment.permutation_spec import (
    mlp_permutation_spec,
    vgg_features_permutation_spec,
)
from mode_connectivity.external.sinkhorn_rebasin import load_upstream_vgg_class
from mode_connectivity.fashion_mnist.model import FashionMLP
from .protocol import load


def make_model(cfg):
    if cfg["dataset"] == "fashion_mnist":
        return FashionMLP(int(cfg["hidden_width"]), int(cfg["hidden_layers"]))
    return load_upstream_vgg_class()(
        cfg["model"], in_channels=3, out_features=10, h_in=32, w_in=32
    )


def read_model(path, cfg):
    result = make_model(cfg)
    result.load_state_dict(load(path)["state_dict"])
    result.to(cfg["device"]).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def permutation_spec(cfg):
    if cfg["dataset"] == "fashion_mnist":
        return mlp_permutation_spec(int(cfg["hidden_layers"]))
    return vgg_features_permutation_spec(cfg["model"])


def cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def device_state(state, device):
    return {key: value.detach().to(device) for key, value in state.items()}


def positive_scaled_state(state, log_scales, spec):
    """Apply positive ReLU scales described by the permutation symmetry graph."""
    result = {}
    for key, value in state.items():
        transformed = value
        axes = spec.axes_to_perm[key]
        for axis, group in enumerate(axes):
            if group is None:
                continue
            scale = log_scales[group].exp()
            shape = [1] * transformed.ndim
            shape[axis] = len(scale)
            factor = scale.reshape(shape)
            transformed = transformed * factor if axis == 0 else transformed / factor
        result[key] = transformed
    return result


def interpolated_profile(template, left, right, loader, alphas, stop=None, cached=None):
    from mode_connectivity.training_stage.geometry import barriers

    losses, errors = [], []
    cached = cached or {}
    left = dict(left)
    with torch.no_grad():
        for alpha in [float(x) for x in alphas]:
            if stop:
                stop.check()
            key = f"{alpha:.12g}"
            if key in cached:
                losses.append(float(cached[key]["loss"]))
                errors.append(float(cached[key]["error"]))
                continue
            state = {key: torch.lerp(left[key], right[key], alpha) for key in left}
            total = wrong = count = 0
            for x, y in loader:
                x, y = x.to(next(iter(state.values())).device), y.to(next(iter(state.values())).device)
                logits = functional_call(template, state, (x,))
                total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
                wrong += (logits.argmax(1) != y).sum().item()
                count += len(y)
            losses.append(total / count)
            errors.append(100.0 * wrong / count)
    alpha_values = [float(x) for x in alphas]
    return dict(
        alphas=alpha_values, losses=losses, errors=errors,
        loss=barriers(alpha_values, losses), error=barriers(alpha_values, errors),
    )


@torch.no_grad()
def equivalence(template, original, transformed, loader, cfg):
    maximum = 0.0
    for x, _ in loader:
        x = x.to(cfg["device"])
        expected = functional_call(template, original, (x,))
        actual = functional_call(template, transformed, (x,))
        torch.testing.assert_close(actual, expected, atol=float(cfg["atol"]), rtol=float(cfg["rtol"]))
        maximum = max(maximum, (actual - expected).abs().max().item())
    return maximum
