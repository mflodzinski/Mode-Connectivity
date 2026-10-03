"""Architecture and exact-symmetry checks for the Fashion-MNIST benchmark."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mode_connectivity.alignment.weight_matching import apply_permutation
from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.fashion_mnist.model import (
    FashionMLP,
    permutation_spec,
    positively_scaled_state,
)
from mode_connectivity.fashion_mnist.runner import smoke, validate_config
from omegaconf import OmegaConf
from torch.func import functional_call


def config():
    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="fashion_mnist/default",
            caller_file=__file__,
            argv=["device=cpu", "workers=0"],
        ),
        resolve=True,
    )


class FashionMNISTTests(unittest.TestCase):
    def test_exact_requested_architecture(self):
        cfg = config()
        validate_config(cfg)
        self.assertEqual(len(cfg["checkpoints"]), 16)
        self.assertEqual(cfg["stages"], cfg["checkpoints"])
        self.assertEqual(cfg["checkpoints"][:6], [0, 1, 2, 3, 4, 5])
        self.assertEqual(cfg["checkpoints"][-1], 100)
        net = FashionMLP()
        linears = [
            module for module in net.modules() if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(len(linears), 11)
        self.assertEqual(
            [tuple(layer.weight.shape) for layer in linears],
            [(512, 784)] + [(512, 512)] * 9 + [(10, 512)],
        )
        self.assertEqual(sum(p.numel() for p in net.parameters()), 2_770_954)
        self.assertFalse(
            any(
                isinstance(module, (torch.nn.BatchNorm1d, torch.nn.Dropout))
                for module in net.modules()
            )
        )
        self.assertEqual(smoke(cfg)["output_shape"], [2, 10])
        self.assertEqual(set(net.state_dict()), set(permutation_spec(cfg).axes_to_perm))

    def test_positive_scaling_is_function_preserving(self):
        torch.manual_seed(4)
        net = FashionMLP(width=8, hidden_layers=3).eval()
        state = net.state_dict()
        log_scales = [torch.randn(8) * 0.5 for _ in range(3)]
        transformed = positively_scaled_state(state, log_scales, 3)
        x = torch.randn(7, 1, 28, 28)
        torch.testing.assert_close(
            net(x),
            functional_call(net, transformed, (x,)),
            atol=2e-6,
            rtol=2e-5,
        )

    def test_hidden_permutation_is_function_preserving(self):
        torch.manual_seed(5)
        net = FashionMLP(width=8, hidden_layers=3).eval()
        cfg = {"hidden_layers": 3}
        permutation = {
            f"P_{index}": np.random.default_rng(index).permutation(8)
            for index in range(3)
        }
        transformed = apply_permutation(
            permutation_spec(cfg), permutation, net.state_dict()
        )
        x = torch.randn(7, 1, 28, 28)
        torch.testing.assert_close(
            net(x),
            functional_call(net, transformed, (x,)),
            atol=2e-6,
            rtol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
