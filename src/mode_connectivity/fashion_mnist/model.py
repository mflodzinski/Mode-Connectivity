"""The symmetry-clean Fashion-MNIST MLP and exact symmetry transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from mode_connectivity.alignment.permutation_spec import mlp_permutation_spec
from mode_connectivity.alignment.weight_matching import apply_permutation


class FashionMLP(nn.Module):
    """784 -> 512 x 10 -> 10, with exactly ten hidden ReLU layers.

    Dense names deliberately match :func:`mlp_permutation_spec`.  There are no
    buffers, normalization layers, residual connections, or stochastic layers.
    """

    def __init__(self, width: int = 512, hidden_layers: int = 10):
        super().__init__()
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive")
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        for index in range(self.hidden_layers):
            inputs = 28 * 28 if index == 0 else self.width
            setattr(self, f"Dense_{index}", nn.Linear(inputs, self.width))
        setattr(self, f"Dense_{self.hidden_layers}", nn.Linear(self.width, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 28 * 28)
        for index in range(self.hidden_layers):
            x = torch.relu(getattr(self, f"Dense_{index}")(x))
        return getattr(self, f"Dense_{self.hidden_layers}")(x)


def make_model(cfg) -> FashionMLP:
    return FashionMLP(
        width=int(cfg["hidden_width"]),
        hidden_layers=int(cfg["hidden_layers"]),
    )


def permutation_spec(cfg):
    return mlp_permutation_spec(int(cfg["hidden_layers"]))


def permuted_state(
    state: Mapping[str, torch.Tensor], permutation, cfg
) -> dict[str, torch.Tensor]:
    return apply_permutation(permutation_spec(cfg), permutation, dict(state))


def positively_scaled_state(
    state: Mapping[str, torch.Tensor],
    log_scales: Sequence[torch.Tensor],
    hidden_layers: int,
) -> dict[str, torch.Tensor]:
    """Apply the exact positive-homogeneity symmetry to an MLP state.

    If hidden activation ``h_i`` is multiplied by ``s_i > 0``, the incoming
    weight and bias are multiplied by ``s_i`` and the next layer's input axis
    is divided by ``s_i``.  ReLU commutes with this positive rescaling.
    """

    if len(log_scales) != hidden_layers:
        raise ValueError("one log-scale vector is required per hidden layer")
    result: dict[str, torch.Tensor] = {}
    previous = None
    for index in range(hidden_layers):
        scale = log_scales[index].exp()
        weight = state[f"Dense_{index}.weight"]
        if previous is not None:
            weight = weight / previous[None, :]
        result[f"Dense_{index}.weight"] = weight * scale[:, None]
        result[f"Dense_{index}.bias"] = state[f"Dense_{index}.bias"] * scale
        previous = scale
    output = hidden_layers
    result[f"Dense_{output}.weight"] = (
        state[f"Dense_{output}.weight"] / previous[None, :]
    )
    result[f"Dense_{output}.bias"] = state[f"Dense_{output}.bias"]
    if set(result) != set(state):
        raise ValueError("state dictionary does not match the configured MLP")
    return result
