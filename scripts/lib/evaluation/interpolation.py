"""Interpolation methods for mode connectivity evaluation.

Provides different path interpolation methods:
- Linear interpolation between two endpoints
- Piecewise linear through symmetry plane
- Curve-based interpolation (handled by CurveNet)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class Interpolator:
    """Path interpolation methods for connectivity evaluation."""

    @staticmethod
    def linear(w1: Dict[str, torch.Tensor],
               w2: Dict[str, torch.Tensor],
               t: float) -> Dict[str, torch.Tensor]:
        """Linear interpolation between two weight dictionaries.

        Formula: w(t) = (1-t) * w1 + t * w2

        Args:
            w1: First endpoint weights (at t=0)
            w2: Second endpoint weights (at t=1)
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated weights at parameter t
        """
        weights = {}
        for key in w1.keys():
            weights[key] = (1.0 - t) * w1[key] + t * w2[key]
        return weights

    @staticmethod
    def symmetry_plane(w1: Dict[str, torch.Tensor],
                       theta: Dict[str, torch.Tensor],
                       w2: Dict[str, torch.Tensor],
                       t: float) -> Dict[str, torch.Tensor]:
        """Piecewise linear interpolation through symmetry point.

        Path consists of two segments:
        - t ∈ [0, 0.5]: Interpolate from w1 to theta
        - t ∈ [0.5, 1]: Interpolate from theta to w2

        Args:
            w1: First endpoint weights (at t=0)
            theta: Symmetry point weights (at t=0.5)
            w2: Second endpoint weights (at t=1)
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated weights at parameter t
        """
        if t <= 0.5:
            # First segment: w1 → theta
            # Map t ∈ [0, 0.5] to alpha ∈ [0, 1]
            alpha = 2.0 * t
            weights = Interpolator.linear(w1, theta, alpha)
        else:
            # Second segment: theta → w2
            # Map t ∈ [0.5, 1] to beta ∈ [0, 1]
            beta = 2.0 * (t - 0.5)
            weights = Interpolator.linear(theta, w2, beta)

        return weights

    @staticmethod
    def apply_weights(model: nn.Module, weights: Dict[str, torch.Tensor]):
        """Load interpolated weights into model.

        Args:
            model: Model to load weights into
            weights: State dictionary to load
        """
        model.load_state_dict(weights)

    @staticmethod
    def compute_l2_norm(weights: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of weight dictionary.

        Args:
            weights: State dictionary

        Returns:
            L2 norm (scalar)
        """
        total_norm = 0.0
        for param in weights.values():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                total_norm += torch.sum(param ** 2).item()
        return total_norm ** 0.5


class CurveInterpolator:
    """Wrapper for curve-based interpolation.

    Curve interpolation is handled internally by CurveNet models.
    This class provides a consistent interface.
    """

    def __init__(self, curve_model: nn.Module):
        """Initialize curve interpolator.

        Args:
            curve_model: CurveNet instance
        """
        self.curve_model = curve_model

    def evaluate_at_t(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Evaluate curve model at parameter t.

        Args:
            x: Input tensor
            t: Curve parameter in [0, 1]

        Returns:
            Model output at parameter t
        """
        # CurveNet models take t as second argument
        coeffs_t = torch.tensor([t], dtype=torch.float32, device=x.device)
        return self.curve_model(x, coeffs_t)

    def get_model(self) -> nn.Module:
        """Get underlying curve model.

        Returns:
            CurveNet instance
        """
        return self.curve_model
