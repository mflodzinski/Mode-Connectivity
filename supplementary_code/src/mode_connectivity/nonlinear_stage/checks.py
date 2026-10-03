"""Cheap functional checks used before submitting path arrays."""

import copy

import torch

from .geometry import PathModel, linear_logits


def smoke(cfg):
    device = cfg["device"]
    torch.manual_seed(7)
    left = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(12, 8),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(8, 3),
    ).to(device)
    right = copy.deepcopy(left)
    with torch.no_grad():
        for p in right.parameters():
            p.add_(0.01 * torch.randn_like(p))
    before_left = {k: v.detach().clone() for k, v in left.state_dict().items()}
    before_right = {k: v.detach().clone() for k, v in right.state_dict().items()}
    for p in list(left.parameters()) + list(right.parameters()):
        p.requires_grad_(False)
    path = PathModel(left, right, "bezier").to(device).eval()
    x = torch.randn(5, 3, 2, 2, device=device)
    torch.testing.assert_close(path(x, 0.0), left.eval()(x), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(path(x, 1.0), right.eval()(x), atol=1e-6, rtol=1e-5)
    for t in [0.1, 0.37, 0.8]:
        torch.testing.assert_close(
            path(x, t), linear_logits(left, right, x, t), atol=1e-6, rtol=1e-5
        )
    path.train()
    loss = torch.nn.functional.cross_entropy(path(x, 0.41), torch.arange(5, device=device) % 3)
    loss.backward()
    assert all(p.grad is None for p in left.parameters())
    assert all(p.grad is None for p in right.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in path.controls.parameters())
    assert path.template.training and any(isinstance(m, torch.nn.Dropout) and m.training for m in path.modules())
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, before_left[key], atol=0, rtol=0)
    for key, value in right.state_dict().items():
        torch.testing.assert_close(value, before_right[key], atol=0, rtol=0)
    return dict(
        exact_endpoints=True,
        linear_initialization=True,
        frozen_endpoints=True,
        control_gradients=True,
        dropout_training=True,
    )
