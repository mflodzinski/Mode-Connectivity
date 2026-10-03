"""VGG construction, endpoint evaluation, and interpolation-barrier metrics."""

from __future__ import annotations

import numpy as np
import torch

from mode_connectivity.external.sinkhorn_rebasin import load_upstream_vgg_class


def model(cfg):
    return load_upstream_vgg_class()(
        cfg["model"], in_channels=3, out_features=10, h_in=32, w_in=32
    )


def barriers(alphas, values):
    grid = np.asarray(alphas, dtype=float)
    observations = np.asarray(values, dtype=float)
    if (
        len(grid) != len(observations)
        or len(grid) < 2
        or grid[0] != 0
        or grid[-1] != 1
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("Ordered interpolation grid must include 0 and 1")
    if not np.isfinite(observations).all():
        raise FloatingPointError("Nonfinite interpolation profile")
    chord = (1 - grid) * observations[0] + grid * observations[-1]
    excess = observations - chord
    return dict(
        chord=float(excess.max()),
        worse=float(observations.max() - max(observations[0], observations[-1])),
        mean=float(observations.mean()),
        peak_alpha=float(grid[observations.argmax()]),
        chord_peak_alpha=float(grid[excess.argmax()]),
    )


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    loss, errors, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        errors += (logits.argmax(1) != y).sum().item()
        count += len(y)
    if not count or not np.isfinite(loss):
        raise FloatingPointError("Empty or nonfinite evaluation")
    return dict(loss=loss / count, error=100 * errors / count, count=count)
