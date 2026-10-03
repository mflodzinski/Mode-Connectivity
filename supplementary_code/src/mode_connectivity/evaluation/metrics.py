"""Distance and barrier metrics for retained model-comparison workflows.

The module collects the basic state-dict distance measures and barrier-summary
helpers that are reused across evaluation, alignment, and reporting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mode_connectivity.core.checkpoint import load_state_dict


def l2_distance(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    compute_per_layer: bool = True,
) -> Dict[str, Any]:
    """Calculate L2 distance between two model state dictionaries."""

    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2

    if not common_keys:
        raise ValueError("No common keys found between state dicts")

    total_l2_squared = 0.0
    total_params = 0
    layer_distances = {}

    for key in sorted(common_keys):
        param1 = state1[key]
        param2 = state2[key]
        if param1.shape != param2.shape:
            raise ValueError(f"Shape mismatch for key '{key}': {param1.shape} vs {param2.shape}")

        diff = param1.detach().cpu().to(torch.float64) - param2.detach().cpu().to(torch.float64)
        layer_l2_squared = torch.sum(diff ** 2).item()
        layer_l2 = float(np.sqrt(layer_l2_squared))
        n_params = int(param1.numel())
        normalized_l2 = layer_l2 / float(np.sqrt(n_params))

        if compute_per_layer:
            layer_distances[key] = {
                "raw_l2": layer_l2,
                "normalized_l2": normalized_l2,
                "n_params": n_params,
            }

        total_l2_squared += layer_l2_squared
        total_params += n_params

    total_l2 = float(np.sqrt(total_l2_squared))
    normalized_total_l2 = total_l2 / float(np.sqrt(total_params))
    result = {
        "total_l2": total_l2,
        "normalized_total_l2": normalized_total_l2,
        "total_params": total_params,
    }
    if compute_per_layer:
        result["layer_distances"] = layer_distances
    return result


def state_distance_summary(state1: Dict[str, torch.Tensor], state2: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Legacy-friendly summary schema on top of the canonical L2 metric."""

    result = l2_distance(state1, state2, compute_per_layer=False)
    return {
        "l2_distance": float(result["total_l2"]),
        "num_params": int(result["total_params"]),
        "rms_difference": float(result["normalized_total_l2"]),
    }


def calculate_checkpoint_l2_distance(checkpoint1_path: str | Path, checkpoint2_path: str | Path) -> Dict[str, float]:
    """Calculate L2 distance between two retained checkpoints."""

    state1 = load_state_dict(checkpoint1_path, normalize_keys=True)
    state2 = load_state_dict(checkpoint2_path, normalize_keys=True)
    result = l2_distance(state1, state2, compute_per_layer=False)
    return {
        "total_l2": float(result["total_l2"]),
        "normalized_l2": float(result["normalized_total_l2"]),
        "total_params": int(result["total_params"]),
    }


def save_l2_distance_report(
    run_dir: str | Path,
    l2_stats: Dict[str, float],
    endpoint_names: Tuple[str, str],
    filename: str = "endpoint_l2_distance.txt",
) -> str:
    """Save L2 distance statistics to a text file."""

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w") as handle:
        handle.write("L2 Distance Between Endpoints\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Endpoint 0: {endpoint_names[0]}\n")
        handle.write(f"Endpoint 1: {endpoint_names[1]}\n\n")
        handle.write(f"Total L2 distance:      {l2_stats['total_l2']:.6f}\n")
        handle.write(f"Normalized L2 distance: {l2_stats['normalized_l2']:.6f}\n")
        handle.write(f"Total parameters:       {int(l2_stats['total_params']):,}\n")
    return str(output_path)


def print_l2_statistics(
    l2_stats: Dict[str, float],
    endpoint_names: Optional[Tuple[str, str]] = None,
    title: str = "L2 DISTANCE BETWEEN ENDPOINTS",
) -> None:
    """Print L2 distance statistics to console."""

    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)
    if endpoint_names:
        print(f"Endpoint 0: {endpoint_names[0]}")
        print(f"Endpoint 1: {endpoint_names[1]}")
        print()
    print("L2 Distance Statistics:")
    print(f"  Total L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"  Normalized L2 distance: {l2_stats['normalized_l2']:.6f}")
    print(f"  Total parameters:       {int(l2_stats['total_params']):,}")


def cosine_similarity(state1: Dict[str, torch.Tensor], state2: Dict[str, torch.Tensor]) -> float:
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2
    if not common_keys:
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

    norm1 = float(np.sqrt(norm1_squared))
    norm2 = float(np.sqrt(norm2_squared))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def max_absolute_difference(state1: Dict[str, torch.Tensor], state2: Dict[str, torch.Tensor]) -> float:
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common_keys = keys1 & keys2
    if not common_keys:
        raise ValueError("No common keys found between state dicts")

    max_diff = 0.0
    for key in common_keys:
        diff = torch.abs(state1[key] - state2[key])
        max_diff = max(max_diff, float(torch.max(diff).item()))
    return max_diff


def layer_wise_l2(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    layer_keys: Optional[list] = None,
) -> Dict[str, float]:
    common_keys = set(state1.keys()) & set(state2.keys())
    if layer_keys is not None:
        common_keys &= set(layer_keys)

    distances = {}
    for key in sorted(common_keys):
        diff = state1[key] - state2[key]
        l2 = torch.sqrt(torch.sum(diff ** 2)).item()
        n_params = state1[key].numel()
        distances[key] = float(l2 / np.sqrt(n_params))
    return distances


def permutation_invariant_distance(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    layer_keys: Optional[list] = None,
) -> Dict[str, float]:
    common_keys = set(state1.keys()) & set(state2.keys())
    if layer_keys is not None:
        common_keys &= set(layer_keys)

    distances = {}
    for key in sorted(common_keys):
        distances[key] = abs(torch.norm(state1[key]).item() - torch.norm(state2[key]).item())
    return distances
