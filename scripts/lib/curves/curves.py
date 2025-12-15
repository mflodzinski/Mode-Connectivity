"""
Curve-specific utilities.

Provides functions for working with Bezier curves and mode connectivity.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional

# Add external dependencies to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import curves as dnn_curves


def load_curve_checkpoint(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """
    Load curve checkpoint.

    Args:
        path: Path to curve checkpoint file
        map_location: Device to map tensors to

    Returns:
        Curve checkpoint dictionary
    """
    return torch.load(path, map_location=map_location)


def extract_bends_from_curve(curve_state: Dict[str, torch.Tensor],
                             num_bends: int = 3) -> List[Dict[str, torch.Tensor]]:
    """
    Extract bend point parameters from curve state dict.

    Args:
        curve_state: Curve model state dict
        num_bends: Number of bend points

    Returns:
        List of state dicts, one per bend point
    """
    bends = []

    for i in range(num_bends):
        bend_state = {}
        for key in curve_state.keys():
            if key.endswith(f'.{i}'):
                # Remove the bend index suffix to get parameter name
                param_name = key[:-2]  # Remove '.i'
                bend_state[param_name] = curve_state[key]
        bends.append(bend_state)

    return bends


def create_curve_model(architecture: Any,
                      curve_type: str = 'PolyChain',
                      num_bends: int = 3,
                      num_classes: int = 10,
                      **model_kwargs) -> nn.Module:
    """
    Create a curve model.

    Args:
        architecture: Model architecture (from models module)
        curve_type: Type of curve ('PolyChain', 'Bezier', etc.)
        num_bends: Number of bend points
        num_classes: Number of output classes
        **model_kwargs: Additional model kwargs

    Returns:
        Curve model instance
    """
    # Get curve class
    if not hasattr(dnn_curves, curve_type):
        raise ValueError(f"Curve type '{curve_type}' not found")

    curve_class = getattr(dnn_curves, curve_type)

    # Handle architecture objects with .base and .kwargs
    if hasattr(architecture, 'base') and hasattr(architecture, 'kwargs'):
        base_class = architecture.base
        kwargs = {**architecture.kwargs, **model_kwargs}
    else:
        base_class = architecture
        kwargs = model_kwargs

    # Create curve model
    curve_model = curve_class(
        num_classes=num_classes,
        num_bends=num_bends,
        **kwargs
    )

    return curve_model


def evaluate_curve_at_t(curve_model: nn.Module,
                       t: float,
                       loader: torch.utils.data.DataLoader,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
    """
    Evaluate curve model at specific t value.

    Args:
        curve_model: Curve model instance
        t: Parameter value (typically in [0, 1])
        loader: Data loader
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation results
    """
    from . import evaluation

    # Set curve parameter
    if hasattr(curve_model, 'import_base_parameters'):
        # For some curve implementations
        weights = curve_model.get_weights_at_t(t)
        curve_model.import_base_parameters(weights)
    else:
        # Standard approach - just evaluate with t parameter
        curve_model.t = t

    # Evaluate
    results = evaluation.evaluate_model(curve_model, loader, device)

    return results


def get_curve_weights_at_t(curve_state: Dict[str, torch.Tensor],
                           t: float,
                           num_bends: int = 3) -> Dict[str, torch.Tensor]:
    """
    Compute interpolated weights at parameter value t.

    For Bezier curves with control points w0, w1, ..., wn:
    w(t) = sum_i ( binom(n,i) * (1-t)^(n-i) * t^i * w_i )

    Args:
        curve_state: Curve model state dict with bend parameters
        t: Parameter value (typically in [0, 1])
        num_bends: Number of bend points

    Returns:
        State dict with interpolated weights
    """
    from scipy.special import comb

    # Extract bend points
    bends = extract_bends_from_curve(curve_state, num_bends)

    if len(bends) == 0:
        raise ValueError("No bend points found in curve state")

    # Initialize result
    result = {}

    # Get parameter names from first bend
    param_names = bends[0].keys()

    # Bezier interpolation
    n = num_bends - 1  # degree of curve

    for param_name in param_names:
        weighted_sum = None

        for i, bend in enumerate(bends):
            # Bezier basis function
            coeff = comb(n, i, exact=True) * ((1 - t) ** (n - i)) * (t ** i)

            if weighted_sum is None:
                weighted_sum = coeff * bend[param_name]
            else:
                weighted_sum = weighted_sum + coeff * bend[param_name]

        result[param_name] = weighted_sum

    return result


def compute_curve_statistics(curve_state: Dict[str, torch.Tensor],
                             num_bends: int = 3) -> Dict[str, Any]:
    """
    Compute statistics about curve parameters.

    Args:
        curve_state: Curve model state dict
        num_bends: Number of bend points

    Returns:
        Dictionary with statistics
    """
    bends = extract_bends_from_curve(curve_state, num_bends)

    if len(bends) == 0:
        return {'num_bends': 0}

    # Compute distances between consecutive bends
    from . import metrics

    bend_distances = []
    for i in range(len(bends) - 1):
        dist = metrics.l2_distance(bends[i], bends[i+1], compute_per_layer=False)
        bend_distances.append(dist['total_l2'])

    return {
        'num_bends': len(bends),
        'bend_to_bend_distances': bend_distances,
        'total_path_length': sum(bend_distances),
        'mean_bend_distance': np.mean(bend_distances) if bend_distances else 0,
    }
