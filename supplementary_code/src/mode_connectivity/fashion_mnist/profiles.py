"""Endpoint loading and sampled linear/path profiles."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call

from mode_connectivity.training_stage.geometry import barriers

from .model import make_model
from .protocol import load


def read_model(path, cfg):
    net = make_model(cfg).to(cfg["device"])
    net.load_state_dict(load(path)["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    total, wrong, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        wrong += (logits.argmax(1) != y).sum().item()
        count += len(y)
    return dict(loss=total / count, error=100.0 * wrong / count, count=count)


def interpolated_state(left, right, alpha):
    return {key: torch.lerp(left[key], right[key], alpha) for key in left}


@torch.no_grad()
def state_profile(template, left, right, loader, device, points, stop=None):
    alphas = np.linspace(0.0, 1.0, int(points)).tolist()
    losses, errors = [], []
    for alpha in alphas:
        if stop is not None:
            stop.check()
        params = interpolated_state(left, right, alpha)
        total, wrong, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = functional_call(template, params, (x,))
            total += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        losses.append(total / count)
        errors.append(100.0 * wrong / count)
    return dict(
        alphas=alphas,
        losses=losses,
        errors=errors,
        loss=barriers(alphas, losses),
        error=barriers(alphas, errors),
    )


@torch.no_grad()
def equivalence(template, original, transformed, loader, device, atol, rtol):
    maximum = 0.0
    for x, _ in loader:
        x = x.to(device)
        expected = functional_call(template, original, (x,))
        actual = functional_call(template, transformed, (x,))
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        maximum = max(maximum, (actual - expected).abs().max().item())
    return maximum
