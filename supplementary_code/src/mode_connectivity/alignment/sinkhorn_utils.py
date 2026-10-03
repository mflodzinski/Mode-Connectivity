"""Generic Sinkhorn utilities shared across retained alignment workflows.

The main helper computes numerically stable doubly-stochastic relaxations used
by the differentiable permutation experiments.
"""

from __future__ import annotations

import torch


def stable_sinkhorn(logits: torch.Tensor, tau: float, num_iters: int) -> torch.Tensor:
    """Compute a differentiable doubly stochastic matrix with log-space normalization."""

    if tau <= 0:
        raise ValueError(f"Sinkhorn temperature must be positive, received tau={tau}.")

    log_transport = logits / tau
    for _ in range(num_iters):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True)
        log_transport = log_transport - torch.logsumexp(log_transport, dim=0, keepdim=True)
    return torch.exp(log_transport)
