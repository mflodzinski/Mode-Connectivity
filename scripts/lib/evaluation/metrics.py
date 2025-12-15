"""
Distance and metric calculations for model comparison.

Provides L2 distance, cosine similarity, and other metrics
for comparing model parameters.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


def l2_distance(state1: Dict[str, torch.Tensor],
                state2: Dict[str, torch.Tensor],
                compute_per_layer: bool = True) -> Dict[str, Any]:
    """
    Calculate L2 distance between two model state dictionaries.

    Args:
        state1: First model state dict
        state2: Second model state dict
        compute_per_layer: Whether to compute per-layer distances

    Returns:
        Dictionary containing:
        - 'total_l2': Total L2 distance
        - 'normalized_total_l2': L2 distance normalized by sqrt(total_params)
        - 'total_params': Total number of parameters
        - 'layer_distances': Dict of per-layer metrics (if compute_per_layer=True)
            Each layer contains: 'raw_l2', 'normalized_l2', 'n_params'
    """
    # Find common keys
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if len(common_keys) == 0:
        raise ValueError("No common keys found between state dicts")

    total_l2_squared = 0.0
    total_params = 0
    layer_distances = {}

    for key in sorted(common_keys):
        param1 = state1[key]
        param2 = state2[key]

        # Ensure same shape
        if param1.shape != param2.shape:
            raise ValueError(f"Shape mismatch for key '{key}': {param1.shape} vs {param2.shape}")

        # Calculate difference
        diff = param1 - param2
        layer_l2_squared = torch.sum(diff ** 2).item()
        layer_l2 = np.sqrt(layer_l2_squared)
        n_params = param1.numel()

        # Normalized L2 (per sqrt of parameter count)
        normalized_l2 = layer_l2 / np.sqrt(n_params)

        if compute_per_layer:
            layer_distances[key] = {
                'raw_l2': layer_l2,
                'normalized_l2': normalized_l2,
                'n_params': n_params
            }

        total_l2_squared += layer_l2_squared
        total_params += n_params

    total_l2 = np.sqrt(total_l2_squared)
    normalized_total_l2 = total_l2 / np.sqrt(total_params)

    result = {
        'total_l2': total_l2,
        'normalized_total_l2': normalized_total_l2,
        'total_params': total_params,
    }

    if compute_per_layer:
        result['layer_distances'] = layer_distances

    return result


def cosine_similarity(state1: Dict[str, torch.Tensor],
                      state2: Dict[str, torch.Tensor]) -> float:
    """
    Calculate cosine similarity between two model state dictionaries.

    Args:
        state1: First model state dict
        state2: Second model state dict

    Returns:
        Cosine similarity (1.0 = identical direction, -1.0 = opposite)
    """
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if len(common_keys) == 0:
        raise ValueError("No common keys found between state dicts")

    dot_product = 0.0
    norm1_squared = 0.0
    norm2_squared = 0.0

    for key in common_keys:
        param1 = state1[key].flatten()
        param2 = state2[key].flatten()

        dot_product += torch.dot(param1, param2).item()
        norm1_squared += torch.sum(param1 ** 2).item()
        norm2_squared += torch.sum(param2 ** 2).item()

    norm1 = np.sqrt(norm1_squared)
    norm2 = np.sqrt(norm2_squared)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def max_absolute_difference(state1: Dict[str, torch.Tensor],
                            state2: Dict[str, torch.Tensor]) -> float:
    """
    Calculate maximum absolute difference between parameters.

    Args:
        state1: First model state dict
        state2: Second model state dict

    Returns:
        Maximum absolute difference across all parameters
    """
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if len(common_keys) == 0:
        raise ValueError("No common keys found between state dicts")

    max_diff = 0.0

    for key in common_keys:
        param1 = state1[key]
        param2 = state2[key]

        diff = torch.abs(param1 - param2)
        layer_max = torch.max(diff).item()

        max_diff = max(max_diff, layer_max)

    return max_diff


def layer_wise_l2(state1: Dict[str, torch.Tensor],
                  state2: Dict[str, torch.Tensor],
                  layer_keys: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate L2 distance for specific layers.

    Args:
        state1: First model state dict
        state2: Second model state dict
        layer_keys: Specific layer keys to compute (None = all common keys)

    Returns:
        Dictionary mapping layer keys to normalized L2 distances
    """
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if layer_keys is not None:
        common_keys = common_keys & set(layer_keys)

    distances = {}

    for key in sorted(common_keys):
        param1 = state1[key]
        param2 = state2[key]

        diff = param1 - param2
        l2 = torch.sqrt(torch.sum(diff ** 2)).item()
        n_params = param1.numel()
        normalized_l2 = l2 / np.sqrt(n_params)

        distances[key] = normalized_l2

    return distances


def permutation_invariant_distance(state1: Dict[str, torch.Tensor],
                                   state2: Dict[str, torch.Tensor],
                                   layer_keys: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate permutation-invariant distance (difference of norms).

    This metric is invariant to neuron permutations within layers.
    For each layer: |norm(w1) - norm(w2)|

    Args:
        state1: First model state dict
        state2: Second model state dict
        layer_keys: Specific layer keys to compute (None = all common keys)

    Returns:
        Dictionary mapping layer keys to permutation-invariant distances
    """
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if layer_keys is not None:
        common_keys = common_keys & set(layer_keys)

    distances = {}

    for key in sorted(common_keys):
        param1 = state1[key]
        param2 = state2[key]

        norm1 = torch.norm(param1).item()
        norm2 = torch.norm(param2).item()

        distances[key] = abs(norm1 - norm2)

    return distances
