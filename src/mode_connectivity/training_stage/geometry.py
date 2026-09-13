"""Differentiable interpolation and hard-transform evaluation (no BN)."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call

from mode_connectivity.external.sinkhorn_rebasin import (
    load_upstream_vgg_class,
    import_external_sinkhorn,
)
from mode_connectivity.sinkhorn.shared import (
    enable_fixed_hard_permutation_scale_only_mode,
)
from .protocol import load


def model():
    return load_upstream_vgg_class()(
        "VGG16", in_channels=3, out_features=10, h_in=32, w_in=32
    )


def read_model(path, device):
    result = model()
    result.load_state_dict(load(path)["state_dict"])
    return result.to(device).eval()


def freeze(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def interpolated_logits(a, b, alpha, x):
    left = dict(a.named_parameters())
    params = {k: torch.lerp(left[k], value, alpha) for k, value in b.named_parameters()}
    return functional_call(a, params, (x,))


def barriers(alphas, values):
    t, y = np.asarray(alphas, dtype=float), np.asarray(values, dtype=float)
    if (
        len(t) != len(y)
        or len(t) < 2
        or t[0] != 0
        or t[-1] != 1
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Ordered interpolation grid must include 0 and 1.")
    if not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite interpolation profile.")
    excess = y - ((1 - t) * y[0] + t * y[-1])
    return dict(
        chord=float(excess.max()),
        worse=float(y.max() - max(y[0], y[-1])),
        mean=float(y.mean()),
        peak_alpha=float(t[y.argmax()]),
        chord_peak_alpha=float(t[excess.argmax()]),
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
        raise FloatingPointError("Empty or nonfinite evaluation.")
    return dict(loss=loss / count, error=100 * errors / count, count=count)


@torch.no_grad()
def profile(a, b, loader, device, alphas, endpoints=None, stop=None):
    a.eval()
    b.eval()
    alphas = [float(t) for t in alphas]
    losses, errors = [], []
    left = dict(a.named_parameters())
    # One interpolation parameter mapping at a time; never retain interpolated models.
    for i, alpha in enumerate(alphas):
        if stop:
            stop.check()
        cached = (
            endpoints[0 if i == 0 else 1]
            if endpoints and i in (0, len(alphas) - 1)
            else None
        )
        if cached is None:
            params = {
                k: torch.lerp(left[k], value, alpha)
                for k, value in b.named_parameters()
            }
            loss, wrong, count = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = functional_call(a, params, (x,))
                loss += torch.nn.functional.cross_entropy(
                    logits, y, reduction="sum"
                ).item()
                wrong += (logits.argmax(1) != y).sum().item()
                count += len(y)
            cached = dict(loss=loss / count, error=100 * wrong / count)
        losses.append(cached["loss"])
        errors.append(cached["error"])
    return dict(
        alphas=alphas,
        losses=losses,
        errors=errors,
        loss=barriers(alphas, losses),
        error=barriers(alphas, errors),
    )


@torch.no_grad()
def equivalence(original, transformed, loader, device, cfg):
    original.eval()
    transformed.eval()
    max_difference = 0.0
    # Fixed selection examples only; no test-dependent acceptance.
    for x, _ in loader:
        x = x.to(device)
        left, right = original(x), transformed(x)
        if not torch.isfinite(right).all():
            raise FloatingPointError("Nonfinite transformed endpoint logits.")
        torch.testing.assert_close(left, right, atol=cfg["atol"], rtol=cfg["rtol"])
        max_difference = max(max_difference, (left - right).abs().max().item())
    return max_difference


@torch.no_grad()
def diagnostics(a, b, loader, device):
    distance = (
        sum(
            (p - q).double().square().sum().item()
            for p, q in zip(a.parameters(), b.parameters())
        )
        ** 0.5
    )
    different, count = 0, 0
    for x, y in loader:
        x = x.to(device)
        different += (a(x).argmax(1) != b(x).argmax(1)).sum().item()
        count += len(y)
    return dict(
        parameter_distance=distance, prediction_disagreement=100 * different / count
    )


def make_rebasin(b, cfg, scale=False, artifact=None):
    _, RebasinNet, matching = import_external_sinkhorn()
    # Auto-graph discovery uses a CPU input and requires gradients on the source.
    b.cpu()
    for p in b.parameters():
        p.requires_grad_(True)
    pi = RebasinNet(
        b,
        input_shape=(1, 3, 32, 32),
        l=cfg["sinkhorn_l"],
        tau=cfg["tau"],
        n_iter=cfg["sinkhorn_iters"],
        scale_invariant=scale,
        lambda_scale=cfg["lambda_scale"],
    )
    freeze(b).to(cfg["device"])
    pi.to(cfg["device"])
    # Vendored RebasinNet.to moves alignment variables only.
    pi.reparamnet.to(cfg["device"])
    pi.identity_init()
    if artifact:
        set_alignment(pi, artifact)
    for p in pi.p:
        if p is not None:
            p.requires_grad_(not scale)
    for u in pi.u:
        if u is not None:
            u.requires_grad_(scale)
    if scale:
        enable_fixed_hard_permutation_scale_only_mode(pi, matching=matching)
    return pi


def alignment_state(pi):
    return dict(
        raw_parameters=[p.detach().cpu().clone() for p in pi.p if p is not None],
        raw_log_scales=[u.detach().cpu().clone() for u in pi.u if u is not None],
    )


def set_alignment(pi, artifact):
    with torch.no_grad():
        for name, key in [("p", "raw_parameters"), ("u", "raw_log_scales")]:
            targets = [p for p in getattr(pi, name) if p is not None]
            source = artifact[key]
            if len(targets) != len(source):
                raise ValueError("Alignment artifact architecture mismatch.")
            for p, value in zip(targets, source):
                p.copy_(value.to(p.device))


def hard_artifact(pi):
    _, _, matching = import_external_sinkhorn()
    result = alignment_state(pi)
    result["hard_permutations"] = [
        matching(p.numpy()).float() for p in result["raw_parameters"]
    ]
    result["scale_stats"] = pi.scale_stats()
    return result


def transformed(b, artifact, cfg):
    if "weight_permutation" in artifact:
        from mode_connectivity.alignment.permutation_spec import (
            vgg16_features_permutation_spec,
        )
        from mode_connectivity.alignment.weight_matching import apply_permutation

        output = model().to(cfg["device"]).eval()
        output.load_state_dict(
            apply_permutation(
                vgg16_features_permutation_spec(),
                artifact["weight_permutation"],
                b.state_dict(),
            )
        )
        return output
    pi = make_rebasin(b, cfg, scale=artifact.get("scale", False), artifact=artifact)
    pi.eval()
    with torch.no_grad():
        out = model().to(cfg["device"]).eval()
        out.load_state_dict(pi().state_dict())
    del pi
    return out
