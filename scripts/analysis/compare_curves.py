"""
Script to compare multiple Bezier curves trained with different seeds.
Verifies that the curves are indeed different due to stochasticity.
"""
import os
import argparse
import torch
import numpy as np
from pathlib import Path


def load_checkpoint(path):
    """Load a checkpoint and return the model state dict."""
    ckpt = torch.load(path, map_location='cpu')
    # Handle both wrapped and unwrapped checkpoints
    if 'model_state' in ckpt:
        return ckpt['model_state']
    return ckpt


def calculate_parameter_distance(state1, state2, metric='l2'):
    """
    Calculate distance between two model state dicts.

    Args:
        state1, state2: Model state dictionaries
        metric: 'l2' for L2 norm, 'cosine' for cosine similarity

    Returns:
        Dictionary with distance metrics
    """
    total_diff_squared = 0.0
    total_norm1_squared = 0.0
    total_norm2_squared = 0.0
    total_params = 0
    max_diff = 0.0

    param_diffs = {}

    for key in state1.keys():
        if key in state2 and isinstance(state1[key], torch.Tensor):
            diff = state1[key] - state2[key]
            diff_squared = torch.sum(diff ** 2).item()

            total_diff_squared += diff_squared
            total_norm1_squared += torch.sum(state1[key] ** 2).item()
            total_norm2_squared += torch.sum(state2[key] ** 2).item()
            total_params += state1[key].numel()

            param_max_diff = torch.max(torch.abs(diff)).item()
            max_diff = max(max_diff, param_max_diff)

            param_diffs[key] = {
                'l2': np.sqrt(diff_squared),
                'max': param_max_diff,
                'numel': state1[key].numel()
            }

    l2_distance = np.sqrt(total_diff_squared)
    normalized_l2 = l2_distance / np.sqrt(total_params) if total_params > 0 else 0

    # Cosine similarity
    dot_product = np.sqrt(total_norm1_squared * total_norm2_squared)
    cosine_sim = 1.0 - (total_diff_squared / (2 * dot_product)) if dot_product > 0 else 0

    return {
        'l2_distance': l2_distance,
        'normalized_l2': normalized_l2,
        'max_absolute_diff': max_diff,
        'cosine_similarity': cosine_sim,
        'total_params': total_params,
        'param_diffs': param_diffs
    }


def compare_curves(checkpoint_paths, checkpoint_name='checkpoint-200.pt'):
    """
    Compare multiple curve checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoint directories
        checkpoint_name: Name of checkpoint file to compare

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*80)
    print("LOADING CHECKPOINTS")
    print("="*80)

    # Load all checkpoints
    states = {}
    for path in checkpoint_paths:
        ckpt_file = Path(path) / checkpoint_name
        if not ckpt_file.exists():
            print(f"⚠️  WARNING: Checkpoint not found: {ckpt_file}")
            continue

        print(f"Loading: {ckpt_file}")
        states[str(path)] = load_checkpoint(str(ckpt_file))

    if len(states) < 2:
        print("\n❌ ERROR: Need at least 2 checkpoints to compare")
        return None

    print(f"\n✅ Loaded {len(states)} checkpoints")

    # Pairwise comparisons
    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS")
    print("="*80)

    paths = list(states.keys())
    comparisons = {}

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            key = f"{Path(path1).name} vs {Path(path2).name}"

            print(f"\n{key}:")
            print("-" * 80)

            metrics = calculate_parameter_distance(states[path1], states[path2])
            comparisons[key] = metrics

            print(f"  L2 Distance:           {metrics['l2_distance']:.6f}")
            print(f"  Normalized L2:         {metrics['normalized_l2']:.6f}")
            print(f"  Max Absolute Diff:     {metrics['max_absolute_diff']:.6f}")
            print(f"  Cosine Similarity:     {metrics['cosine_similarity']:.6f}")
            print(f"  Total Parameters:      {metrics['total_params']:,}")

    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Compare Bezier curves trained with different seeds"
    )
    parser.add_argument(
        "--checkpoint-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to checkpoint directories to compare"
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint-200.pt",
        help="Name of checkpoint file to compare (default: checkpoint-200.pt)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for considering curves 'different' (default: 1e-6)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BEZIER CURVE COMPARISON")
    print("="*80)
    print(f"Checkpoint file: {args.checkpoint_name}")
    print(f"Difference threshold: {args.threshold}")
    print("="*80)

    # Compare curves
    comparisons = compare_curves(args.checkpoint_dirs, args.checkpoint_name)

    if comparisons is None:
        return 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_different = True
    for key, metrics in comparisons.items():
        is_different = metrics['normalized_l2'] > args.threshold
        status = "✅ DIFFERENT" if is_different else "⚠️  IDENTICAL (or very similar)"
        print(f"{status}: {key}")
        print(f"  Normalized L2: {metrics['normalized_l2']:.6e}")

        if not is_different:
            all_different = False

    print("="*80)

    if all_different:
        print("\n✅ SUCCESS: All curves are different!")
        print("   This confirms that stochasticity is working correctly.")
        return 0
    else:
        print("\n⚠️  WARNING: Some curves appear identical!")
        print("   This might indicate that seeds are not being applied correctly.")
        return 1


if __name__ == "__main__":
    exit(main())
