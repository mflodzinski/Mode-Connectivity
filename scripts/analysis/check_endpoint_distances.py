"""
Check L2 distances between endpoint checkpoints from different seeds.
This verifies that the three seeds (0, 42, 123) actually converged to different points in parameter space.
"""

import argparse
import torch
import numpy as np
import os
import sys

# Add lib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(script_dir)
sys.path.insert(0, scripts_dir)

from lib.core import models


def load_model_params(checkpoint_path, model_name, dataset_name):
    """Load model parameters from checkpoint."""
    print(f"Loading {checkpoint_path}...")

    # Get number of classes based on dataset
    num_classes = 10  # CIFAR10 and FashionMNIST both have 10 classes

    # Create model
    architecture = models.get_architecture(model_name)
    model = models.create_model(architecture, num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    # Flatten all parameters into a single vector
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    param_vector = torch.cat(params)

    return param_vector


def compute_l2_distance(params1, params2):
    """Compute L2 distance between two parameter vectors."""
    diff = params1 - params2
    return torch.norm(diff).item()


def compute_pairwise_distances(checkpoint_paths, model_name, dataset_name, seed_names):
    """Compute pairwise L2 distances between all checkpoints."""

    # Load all parameter vectors
    print("\nLoading checkpoints...")
    param_vectors = []
    for ckpt_path in checkpoint_paths:
        params = load_model_params(ckpt_path, model_name, dataset_name)
        param_vectors.append(params)

    # Compute parameter norm for each checkpoint
    print("\nParameter L2 norms:")
    for i, (params, seed_name) in enumerate(zip(param_vectors, seed_names)):
        norm = torch.norm(params).item()
        print(f"  {seed_name}: {norm:.4f}")

    # Compute pairwise distances
    print("\nPairwise L2 distances:")
    n = len(param_vectors)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = compute_l2_distance(param_vectors[i], param_vectors[j])
            distances[i, j] = dist
            distances[j, i] = dist
            print(f"  {seed_names[i]} <-> {seed_names[j]}: {dist:.4f}")

    # Compute statistics
    print("\nDistance statistics:")
    non_zero_distances = distances[np.triu_indices(n, k=1)]
    print(f"  Mean distance: {np.mean(non_zero_distances):.4f}")
    print(f"  Std distance:  {np.std(non_zero_distances):.4f}")
    print(f"  Min distance:  {np.min(non_zero_distances):.4f}")
    print(f"  Max distance:  {np.max(non_zero_distances):.4f}")

    # Check if distances are significant
    print("\nVerification:")
    avg_norm = np.mean([torch.norm(p).item() for p in param_vectors])
    avg_distance = np.mean(non_zero_distances)
    relative_distance = avg_distance / avg_norm

    print(f"  Average parameter norm: {avg_norm:.4f}")
    print(f"  Average pairwise distance: {avg_distance:.4f}")
    print(f"  Relative distance (dist/norm): {relative_distance:.4f}")

    if relative_distance > 0.01:  # More than 1% difference
        print(f"\n✓ Checkpoints are SIGNIFICANTLY different!")
        print(f"  The three seeds converged to distinct optima in parameter space.")
    elif relative_distance > 0.001:  # More than 0.1% difference
        print(f"\n✓ Checkpoints are moderately different")
        print(f"  The seeds found different local optima.")
    else:
        print(f"\n✗ WARNING: Checkpoints are very similar!")
        print(f"  The seeds may have converged to nearly the same point.")

    return distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check L2 distances between endpoint checkpoints from different seeds'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., VGG16, ConvFC)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., CIFAR10, FashionMNIST)')
    parser.add_argument('--checkpoint1', type=str, required=True,
                       help='Path to first checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True,
                       help='Path to second checkpoint')
    parser.add_argument('--checkpoint3', type=str, required=True,
                       help='Path to third checkpoint')
    parser.add_argument('--seed-names', type=str, nargs=3, default=['seed0', 'seed1', 'seed2'],
                       help='Names for the three seeds (default: seed0 seed1 seed2)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"ENDPOINT DISTANCE ANALYSIS: {args.model} on {args.dataset}")
    print("=" * 80)

    checkpoint_paths = [args.checkpoint1, args.checkpoint2, args.checkpoint3]

    # Check that all checkpoints exist
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

    distances = compute_pairwise_distances(
        checkpoint_paths,
        args.model,
        args.dataset,
        args.seed_names
    )

    print("\n" + "=" * 80)
