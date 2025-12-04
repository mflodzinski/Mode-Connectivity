"""Calculate L2 distance between two model checkpoints.

This utility calculates the L2 norm of the difference between two neural network
checkpoints. Useful for:
- Measuring distance between curve endpoints before training
- Comparing different initializations
- Analyzing weight space geometry
"""

import argparse
import torch
import numpy as np
import sys
import os
import json

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import models


def calculate_l2_distance(model1_state, model2_state):
    """Calculate L2 distance between two model state dicts.

    Args:
        model1_state: State dict of first model
        model2_state: State dict of second model

    Returns:
        dict with L2 distance statistics
    """
    total_l2_squared = 0.0
    total_params = 0
    layer_distances = {}

    # Get all parameter keys (should be same for both models)
    keys1 = set(model1_state.keys())
    keys2 = set(model2_state.keys())

    if keys1 != keys2:
        print(f"Warning: Models have different parameters!")
        print(f"  Only in model1: {keys1 - keys2}")
        print(f"  Only in model2: {keys2 - keys1}")

    common_keys = keys1 & keys2

    for key in sorted(common_keys):
        param1 = model1_state[key]
        param2 = model2_state[key]

        # Skip non-tensor parameters
        if not isinstance(param1, torch.Tensor) or not isinstance(param2, torch.Tensor):
            continue

        # Check shapes match
        if param1.shape != param2.shape:
            print(f"Warning: Shape mismatch for {key}: {param1.shape} vs {param2.shape}")
            continue

        # Calculate L2 distance for this parameter
        diff = param1 - param2
        layer_l2_squared = torch.sum(diff ** 2).item()
        layer_l2 = np.sqrt(layer_l2_squared)
        n_params = param1.numel()

        # Normalized L2 (per-parameter RMS)
        normalized_l2 = layer_l2 / np.sqrt(n_params)

        # Relative distance (percentage change)
        param1_norm = torch.norm(param1, p=2).item()
        relative_distance = layer_l2 / param1_norm if param1_norm > 0 else 0

        layer_distances[key] = {
            'raw_l2': layer_l2,
            'normalized_l2': normalized_l2,
            'relative_distance': relative_distance,
            'n_params': n_params
        }

        total_l2_squared += layer_l2_squared
        total_params += n_params

    # Total L2 distance across all parameters
    total_l2 = np.sqrt(total_l2_squared)
    normalized_total_l2 = total_l2 / np.sqrt(total_params) if total_params > 0 else 0

    return {
        'total_l2': total_l2,
        'normalized_total_l2': normalized_total_l2,
        'total_params': total_params,
        'num_layers': len(layer_distances),
        'layer_distances': layer_distances
    }


def load_checkpoint_state(checkpoint_path):
    """Load state dict from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        State dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state' in checkpoint:
        return checkpoint['model_state']
    else:
        return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Calculate L2 distance between two model checkpoints'
    )
    parser.add_argument('--checkpoint1', type=str, required=True,
                        help='Path to first checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True,
                        help='Path to second checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON (optional)')
    parser.add_argument('--show-top-k', type=int, default=10,
                        help='Number of top layers to display (default: 10)')
    parser.add_argument('--sort-by', type=str, default='normalized_l2',
                        choices=['normalized_l2', 'raw_l2', 'relative_distance'],
                        help='Metric to sort layers by (default: normalized_l2)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("L2 DISTANCE CALCULATOR")
    print("="*70)
    print(f"\nCheckpoint 1: {args.checkpoint1}")
    print(f"Checkpoint 2: {args.checkpoint2}")

    # Load checkpoints
    print("\nLoading checkpoints...")
    state1 = load_checkpoint_state(args.checkpoint1)
    state2 = load_checkpoint_state(args.checkpoint2)
    print("✓ Checkpoints loaded")

    # Calculate L2 distance
    print("\nCalculating L2 distance...")
    l2_stats = calculate_l2_distance(state1, state2)
    print("✓ Calculation complete")

    # Display results
    print("\n" + "="*70)
    print("L2 DISTANCE STATISTICS")
    print("="*70)
    print(f"\nTotal L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"Normalized L2 distance: {l2_stats['normalized_total_l2']:.6f}")
    print(f"Total parameters:       {l2_stats['total_params']:,}")
    print(f"Number of layers:       {l2_stats['num_layers']}")

    # Show top K layers
    layer_dists = sorted(
        l2_stats['layer_distances'].items(),
        key=lambda x: x[1][args.sort_by],
        reverse=True
    )

    print(f"\nTop {args.show_top_k} layers by {args.sort_by}:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Layer Name':<40} {'Norm L2':<12} {'Raw L2':<12}")
    print("-" * 70)

    for i, (layer_name, stats) in enumerate(layer_dists[:args.show_top_k]):
        # Truncate long layer names
        display_name = layer_name if len(layer_name) <= 40 else layer_name[:37] + "..."
        print(f"{i+1:<6} {display_name:<40} {stats['normalized_l2']:<12.6f} {stats['raw_l2']:<12.6f}")

    print("-" * 70)

    # Show layers with zero distance (if any)
    zero_dist_layers = [name for name, stats in l2_stats['layer_distances'].items()
                       if stats['raw_l2'] < 1e-10]
    if zero_dist_layers:
        print(f"\nLayers with zero distance: {len(zero_dist_layers)}")
        for name in zero_dist_layers[:5]:
            print(f"  - {name}")
        if len(zero_dist_layers) > 5:
            print(f"  ... and {len(zero_dist_layers) - 5} more")

    print("="*70)

    # Save results if requested
    if args.output:
        # Convert to serializable format
        serializable_layer_distances = {
            name: {
                'raw_l2': float(stats['raw_l2']),
                'normalized_l2': float(stats['normalized_l2']),
                'relative_distance': float(stats['relative_distance']),
                'n_params': int(stats['n_params'])
            }
            for name, stats in l2_stats['layer_distances'].items()
        }

        results = {
            'checkpoint1': args.checkpoint1,
            'checkpoint2': args.checkpoint2,
            'total_l2': float(l2_stats['total_l2']),
            'normalized_total_l2': float(l2_stats['normalized_total_l2']),
            'total_params': int(l2_stats['total_params']),
            'num_layers': int(l2_stats['num_layers']),
            'top_10_layers': [
                {
                    'rank': i + 1,
                    'layer_name': name,
                    'normalized_l2': float(stats['normalized_l2']),
                    'raw_l2': float(stats['raw_l2']),
                    'relative_distance': float(stats['relative_distance']),
                    'n_params': int(stats['n_params'])
                }
                for i, (name, stats) in enumerate(layer_dists[:10])
            ],
            'all_layer_distances': serializable_layer_distances
        }

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {args.output}")
        print("="*70)

    # Print interpretation
    print("\nINTERPRETATION:")
    print("-" * 70)

    if l2_stats['normalized_total_l2'] < 0.01:
        print("Very small distance - models are nearly identical in weight space")
    elif l2_stats['normalized_total_l2'] < 0.1:
        print("Small distance - models are similar but distinguishable")
    elif l2_stats['normalized_total_l2'] < 1.0:
        print("Moderate distance - models have significant differences")
    else:
        print("Large distance - models are quite different in weight space")

    print("\nNote: These distances represent the Euclidean distance in weight space.")
    print("Functionally equivalent networks (e.g., via permutation) can have large")
    print("weight space distances while producing identical outputs.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
