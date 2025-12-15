"""Distance metrics for checkpoint analysis.

Provides utilities for calculating L2 distances between model checkpoints.
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_checkpoint_l2_distance(checkpoint1_path: str,
                                     checkpoint2_path: str) -> Dict[str, float]:
    """Calculate L2 distance between two checkpoints.

    Args:
        checkpoint1_path: Path to first checkpoint file
        checkpoint2_path: Path to second checkpoint file

    Returns:
        Dictionary containing:
            - total_l2: Total L2 distance
            - normalized_l2: L2 distance normalized by sqrt(num_params)
            - total_params: Total number of parameters
    """
    # Load checkpoints
    ckpt1 = torch.load(checkpoint1_path, map_location='cpu')
    ckpt2 = torch.load(checkpoint2_path, map_location='cpu')

    # Get state dicts (handle both raw state_dict and checkpoint formats)
    state1 = ckpt1.get('model_state', ckpt1)
    state2 = ckpt2.get('model_state', ckpt2)

    # Calculate L2 distance
    total_l2_squared = 0.0
    total_params = 0

    for key in state1.keys():
        if key in state2 and isinstance(state1[key], torch.Tensor):
            diff = state1[key] - state2[key]
            total_l2_squared += torch.sum(diff ** 2).item()
            total_params += state1[key].numel()

    total_l2 = np.sqrt(total_l2_squared)
    normalized_l2 = total_l2 / np.sqrt(total_params) if total_params > 0 else 0

    return {
        'total_l2': total_l2,
        'normalized_l2': normalized_l2,
        'total_params': total_params
    }


def save_l2_distance_report(run_dir: str,
                            l2_stats: Dict[str, float],
                            endpoint_names: Tuple[str, str],
                            filename: str = "endpoint_l2_distance.txt") -> str:
    """Save L2 distance statistics to a text file.

    Args:
        run_dir: Directory to save the report
        l2_stats: Dictionary from calculate_checkpoint_l2_distance()
        endpoint_names: Tuple of (endpoint0_name, endpoint1_name)
        filename: Output filename (default: endpoint_l2_distance.txt)

    Returns:
        Path to saved file
    """
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, filename)

    with open(output_path, 'w') as f:
        f.write(f"L2 Distance Between Endpoints\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Endpoint 0: {endpoint_names[0]}\n")
        f.write(f"Endpoint 1: {endpoint_names[1]}\n\n")
        f.write(f"Total L2 distance:      {l2_stats['total_l2']:.6f}\n")
        f.write(f"Normalized L2 distance: {l2_stats['normalized_l2']:.6f}\n")
        f.write(f"Total parameters:       {l2_stats['total_params']:,}\n")

    return output_path


def print_l2_statistics(l2_stats: Dict[str, float],
                       endpoint_names: Optional[Tuple[str, str]] = None,
                       title: str = "L2 DISTANCE BETWEEN ENDPOINTS") -> None:
    """Print L2 distance statistics to console.

    Args:
        l2_stats: Dictionary from calculate_checkpoint_l2_distance()
        endpoint_names: Optional tuple of (endpoint0_name, endpoint1_name)
        title: Title for the output section
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)

    if endpoint_names:
        print(f"Endpoint 0: {endpoint_names[0]}")
        print(f"Endpoint 1: {endpoint_names[1]}")
        print()

    print(f"L2 Distance Statistics:")
    print(f"  Total L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"  Normalized L2 distance: {l2_stats['normalized_l2']:.6f}")
    print(f"  Total parameters:       {l2_stats['total_params']:,}")
