"""Interpolation helpers for retained mode-connectivity workflows.

The functions in this file build intermediate parameter states and evaluate
their behavior along linear or curve-based paths between endpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class Interpolator:
    """Path interpolation methods for connectivity evaluation."""

    @staticmethod
    def linear(w1: Dict[str, torch.Tensor], w2: Dict[str, torch.Tensor], t: float) -> Dict[str, torch.Tensor]:
        return {key: (1.0 - t) * w1[key] + t * w2[key] for key in w1.keys()}

    @staticmethod
    def symmetry_plane(
        w1: Dict[str, torch.Tensor],
        theta: Dict[str, torch.Tensor],
        w2: Dict[str, torch.Tensor],
        t: float,
    ) -> Dict[str, torch.Tensor]:
        if t <= 0.5:
            return Interpolator.linear(w1, theta, 2.0 * t)
        return Interpolator.linear(theta, w2, 2.0 * (t - 0.5))

    @staticmethod
    def apply_weights(model: nn.Module, weights: Dict[str, torch.Tensor]):
        model.load_state_dict(weights)

    @staticmethod
    def compute_l2_norm(weights: Dict[str, torch.Tensor]) -> float:
        total_norm = 0.0
        for param in weights.values():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                total_norm += torch.sum(param ** 2).item()
        return total_norm ** 0.5


class CurveInterpolator:
    """Wrapper for curve-based interpolation handled by CurveNet models."""

    def __init__(self, curve_model: nn.Module):
        self.curve_model = curve_model

    def evaluate_at_t(self, x: torch.Tensor, t: float) -> torch.Tensor:
        coeffs_t = torch.tensor([t], dtype=torch.float32, device=x.device)
        return self.curve_model(x, coeffs_t)

    def get_model(self) -> nn.Module:
        return self.curve_model


def evaluate_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate a classifier on one loader and return loss/accuracy."""

    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            total_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
            total += int(targets.size(0))

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * total_correct / total,
    }


def loss_barrier(values: list[float] | np.ndarray) -> float:
    values_np = np.asarray(values, dtype=np.float64)
    endpoint_avg = 0.5 * (values_np[0] + values_np[-1])
    return float(np.max(values_np) - endpoint_avg)


def acc_barrier(values: list[float] | np.ndarray) -> float:
    values_np = np.asarray(values, dtype=np.float64)
    endpoint_avg = 0.5 * (values_np[0] + values_np[-1])
    return float(endpoint_avg - np.min(values_np))


def summarize_interpolation_metrics(interpolation: Dict[str, list[float] | list]) -> Dict[str, float]:
    """Compute standard barrier summaries from sampled interpolation metrics."""

    return {
        "train_loss_barrier": loss_barrier(interpolation["train_loss"]),
        "test_loss_barrier": loss_barrier(interpolation["test_loss"]),
        "train_acc_barrier": acc_barrier(interpolation["train_acc"]),
        "test_acc_barrier": acc_barrier(interpolation["test_acc"]),
        "min_train_acc": float(np.min(np.asarray(interpolation["train_acc"], dtype=np.float64))),
        "min_test_acc": float(np.min(np.asarray(interpolation["test_acc"], dtype=np.float64))),
    }


def evaluate_linear_interpolation(
    *,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    ts: np.ndarray,
) -> Dict[str, list[float] | list]:
    """Evaluate a linear interpolation path between two state dicts."""

    results = {
        "t": ts.tolist(),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for t in ts.tolist():
        interp_state = OrderedDict(Interpolator.linear(state_a, state_b, float(t)))
        model.load_state_dict(interp_state)
        train_res = evaluate_classifier(model, train_loader, device)
        test_res = evaluate_classifier(model, test_loader, device)
        results["train_loss"].append(float(train_res["loss"]))
        results["train_acc"].append(float(train_res["accuracy"]))
        results["test_loss"].append(float(test_res["loss"]))
        results["test_acc"].append(float(test_res["accuracy"]))

    return results
