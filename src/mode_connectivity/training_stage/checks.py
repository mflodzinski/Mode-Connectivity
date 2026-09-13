"""Small, real autograd/functional-symmetry checks for the cluster pilot."""

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from .geometry import (
    make_rebasin,
    freeze,
    interpolated_logits,
    equivalence,
    hard_artifact,
)


def smoke(cfg):
    torch.manual_seed(7)
    device = cfg["device"]
    # Uses the same input shape/graph discovery as VGG, with cheap hidden layers.
    source = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3072, 8),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(8, 10),
    ).eval()
    reference = freeze(copy.deepcopy(source).to(device))
    x, y = torch.randn(4, 3, 32, 32), torch.arange(4)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    original = {k: v.detach().clone() for k, v in source.state_dict().items()}
    pi = make_rebasin(source, cfg)
    optimizer = torch.optim.AdamW(pi.parameters(), lr=cfg["base_lr"])
    for _ in range(2):
        pi.train()
        pi.reparamnet.output.eval()
        rebased = pi()
        rebased.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            interpolated_logits(reference, rebased, 0.4, x.to(device)), y.to(device)
        )
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pi.p)
        optimizer.step()
    pi.eval()
    with torch.no_grad():
        hard = pi()
        hard.eval()
        equivalence(source, hard, loader, device, cfg)
    artifact = hard_artifact(pi)
    del pi
    scaled = make_rebasin(source, cfg, scale=True, artifact=artifact)
    fixed = [p.detach().clone() for p in scaled.p]
    optimizer = torch.optim.AdamW(
        [u for u in scaled.u if u is not None], lr=cfg["scale_lr"]
    )
    for _ in range(2):
        scaled.train()
        scaled.reparamnet.output.eval()
        rebased = scaled()
        rebased.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            interpolated_logits(reference, rebased, 0.4, x.to(device)), y.to(device)
        )
        loss.backward()
        assert any(u.grad is not None and u.grad.abs().sum() > 0 for u in scaled.u)
        optimizer.step()
    scaled.eval()
    with torch.no_grad():
        equivalence(source, scaled().eval(), loader, device, cfg)
    for before, after in zip(fixed, scaled.p):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[key], atol=0, rtol=0)
    return dict(
        autograd=True,
        fixed_permutation=True,
        endpoint_equivalence=True,
        frozen_endpoints=True,
    )
