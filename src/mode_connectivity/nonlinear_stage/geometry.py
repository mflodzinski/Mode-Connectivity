"""Differentiable raw-endpoint paths and sampled path profiles."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from mode_connectivity.training_stage.geometry import barriers, freeze, model
from .protocol import endpoint_metadata, endpoint_path, file_hash, load


def read_endpoints(cfg, pair):
    paths = [
        endpoint_path(cfg, pair["replicate"], pair["left_seed"], pair["left_epoch"]),
        endpoint_path(cfg, pair["replicate"], pair["right_seed"], pair["right_epoch"]),
    ]
    expected = endpoint_metadata(cfg, pair)
    for path, record in zip(paths, expected):
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Endpoint changed after protocol freeze: {path}")
    nets = []
    for path in paths:
        net = model(cfg)
        net.load_state_dict(load(path)["state_dict"])
        nets.append(freeze(net.to(cfg["device"])))
    return nets, paths


class PathModel(nn.Module):
    """A quadratic Bézier or three-segment polygon with frozen endpoints."""

    def __init__(self, left, right, family="bezier", noise=0.0, generator=None):
        super().__init__()
        if family not in {"bezier", "polygon"}:
            raise ValueError(family)
        self.family = family
        self.template = left
        self.names = [name for name, _ in left.named_parameters()]
        right_params = dict(right.named_parameters())
        self.left = tuple(p.detach() for _, p in left.named_parameters())
        self.right = tuple(right_params[name].detach() for name in self.names)
        fractions = [0.5] if family == "bezier" else [1.0 / 3.0, 2.0 / 3.0]
        controls = []
        for fraction in fractions:
            group = nn.ParameterList()
            for a, b in zip(self.left, self.right):
                value = torch.lerp(a, b, fraction).detach().clone()
                if noise:
                    rms = (b - a).double().norm().item() / math.sqrt(a.numel())
                    sample = torch.randn(
                        value.shape,
                        dtype=value.dtype,
                        device=value.device,
                        generator=generator,
                    )
                    value.add_(sample, alpha=float(noise) * rms)
                group.append(nn.Parameter(value))
            controls.append(group)
        self.controls = nn.ModuleList(controls)

    def control_state(self):
        return [[p.detach().cpu().clone() for p in group] for group in self.controls]

    def load_control_state(self, state):
        if len(state) != len(self.controls):
            raise ValueError("Path-family control count mismatch.")
        with torch.no_grad():
            for target_group, source_group in zip(self.controls, state):
                if len(target_group) != len(source_group):
                    raise ValueError("Path control parameter count mismatch.")
                for target, source in zip(target_group, source_group):
                    target.copy_(source.to(target.device))

    def linear_control_state(self):
        fractions = [0.5] if self.family == "bezier" else [1 / 3, 2 / 3]
        return [
            [torch.lerp(a, b, fraction).detach().cpu() for a, b in zip(self.left, self.right)]
            for fraction in fractions
        ]

    def parameters_at(self, t):
        t = torch.as_tensor(t, device=self.left[0].device, dtype=self.left[0].dtype)
        if self.family == "bezier":
            return {
                name: (1 - t).square() * a + 2 * t * (1 - t) * m + t.square() * b
                for name, a, m, b in zip(self.names, self.left, self.controls[0], self.right)
            }
        scaled = t * 3
        points = [self.left, self.controls[0], self.controls[1], self.right]
        segment = min(int(torch.floor(scaled.detach()).item()), 2)
        local = scaled - segment
        return {
            name: torch.lerp(a, b, local)
            for name, a, b in zip(self.names, points[segment], points[segment + 1])
        }

    def forward(self, x, t):
        return functional_call(self.template, self.parameters_at(t), (x,))

    def forward_with_l2(self, x, t):
        params = self.parameters_at(t)
        logits = functional_call(self.template, params, (x,))
        return logits, sum(parameter.square().sum() for parameter in params.values())

    def l2_at(self, t):
        return sum(p.square().sum() for p in self.parameters_at(t).values())


def linear_logits(left, right, x, t):
    a = dict(left.named_parameters())
    b = dict(right.named_parameters())
    params = {name: torch.lerp(value, b[name], t) for name, value in a.items()}
    return functional_call(left, params, (x,))


@torch.no_grad()
def path_profile(path, loader, device, points, *, linear_endpoints=None, stop=None):
    path.eval()
    alphas = [float(t) for t in np.linspace(0, 1, int(points))]
    losses, errors = [], []
    for alpha in alphas:
        if stop:
            stop.check()
        total, wrong, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = path(x, alpha)
            total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        losses.append(total / count)
        errors.append(100 * wrong / count)
    if linear_endpoints is not None:
        losses[0], losses[-1] = linear_endpoints["losses"]
        errors[0], errors[-1] = linear_endpoints["errors"]
    return dict(
        alphas=alphas,
        losses=losses,
        errors=errors,
        loss=barriers(alphas, losses),
        error=barriers(alphas, errors),
        max_loss=float(max(losses)),
        max_error=float(max(errors)),
    )


@torch.no_grad()
def linear_profile(left, right, loader, device, points, stop=None):
    left.eval(), right.eval()
    alphas = [float(t) for t in np.linspace(0, 1, int(points))]
    losses, errors = [], []
    for alpha in alphas:
        if stop:
            stop.check()
        total, wrong, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = linear_logits(left, right, x, alpha)
            total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        losses.append(total / count)
        errors.append(100 * wrong / count)
    return dict(
        alphas=alphas,
        losses=losses,
        errors=errors,
        loss=barriers(alphas, losses),
        error=barriers(alphas, errors),
        max_loss=float(max(losses)),
        max_error=float(max(errors)),
    )


@torch.no_grad()
def path_geometry(path, points=61):
    midpoint_sq = 0.0
    for control_group, fraction in zip(
        path.controls, [0.5] if path.family == "bezier" else [1 / 3, 2 / 3]
    ):
        midpoint_sq += sum(
            (m - torch.lerp(a, b, fraction)).double().square().sum().item()
            for a, m, b in zip(path.left, control_group, path.right)
        )
    endpoint_distance = math.sqrt(
        sum((b - a).double().square().sum().item() for a, b in zip(path.left, path.right))
    )
    previous, arc = None, 0.0
    for t in np.linspace(0, 1, points):
        current = path.parameters_at(float(t))
        if previous is not None:
            arc += math.sqrt(
                sum(
                    (current[name] - previous[name]).double().square().sum().item()
                    for name in path.names
                )
            )
        previous = {k: v.detach().clone() for k, v in current.items()}
    return dict(
        endpoint_distance=endpoint_distance,
        control_displacement=math.sqrt(midpoint_sq),
        approximate_path_length=arc,
        normalized_path_length=arc / endpoint_distance if endpoint_distance else 1.0,
    )


def artifact(path, cfg, pair, family, restart, noise, selected_pass, score):
    return dict(
        family=family,
        restart=int(restart),
        noise=float(noise),
        controls=path.control_state(),
        selected_pass=int(selected_pass),
        selected_score=list(score),
        pair=pair,
        endpoints=endpoint_metadata(cfg, pair),
    )


def path_from_artifact(left, right, value):
    result = PathModel(left, right, value["family"])
    result.load_control_state(value["controls"])
    return result
