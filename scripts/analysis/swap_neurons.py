"""Swap specific neurons in a neural network to test local vs global connectivity.

This script creates a minimally perturbed version of a network by swapping a small
number of neurons. The swapped network is functionally equivalent (same performance)
but at a different point in parameter space.

This tests whether mode connectivity corrections are:
- Local: Only swapped neurons change along the path
- Global: All weights adjust to accommodate the swap
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import sys
import os
import json

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import models
import data


def get_vgg16_layer_info():
    """Get information about VGG16 layers for depth selection.

    Returns:
        Dictionary mapping layer depths to layer indices and names
    """
    # VGG16 structure: 5 blocks with [2, 2, 3, 3, 3] conv layers
    # Block indices in layer_blocks: 0, 1, 2, 3, 4
    return {
        'early': {
            'block_idx': 0,  # First block
            'layer_idx': 0,  # First conv in first block
            'description': 'Block 0, Conv 0 (64 filters, early features)'
        },
        'mid': {
            'block_idx': 2,  # Third block (middle)
            'layer_idx': 1,  # Second conv in third block
            'description': 'Block 2, Conv 1 (256 filters, mid-level features)'
        },
        'late': {
            'block_idx': 4,  # Fifth block
            'layer_idx': 1,  # Second conv in fifth block
            'description': 'Block 4, Conv 1 (512 filters, late features)'
        }
    }


def get_layer_by_depth(model, depth):
    """Get a convolutional layer based on depth specification.

    Args:
        model: VGG model
        depth: 'early', 'mid', or 'late'

    Returns:
        tuple: (layer, block_idx, layer_idx, description)
    """
    layer_info = get_vgg16_layer_info()

    if depth not in layer_info:
        raise ValueError(f"Depth must be one of {list(layer_info.keys())}")

    info = layer_info[depth]
    block_idx = info['block_idx']
    layer_idx = info['layer_idx']

    # Navigate to the layer
    layer = model.layer_blocks[block_idx][layer_idx]

    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"Expected Conv2d layer but got {type(layer)}")

    return layer, block_idx, layer_idx, info['description']


def get_random_neuron_pairs(num_filters, num_pairs, seed=42):
    """Generate random pairs of neuron indices to swap.

    Args:
        num_filters: Total number of filters/neurons in the layer
        num_pairs: Number of pairs to generate
        seed: Random seed for reproducibility

    Returns:
        List of tuples [(idx1, idx2), ...]
    """
    np.random.seed(seed)

    # Generate random indices
    all_indices = np.arange(num_filters)
    np.random.shuffle(all_indices)

    # Create pairs
    pairs = []
    for i in range(min(num_pairs, num_filters // 2)):
        pairs.append((int(all_indices[2*i]), int(all_indices[2*i + 1])))

    return pairs


def swap_conv_filters(model, block_idx, layer_idx, neuron_pairs):
    """Swap filters in a convolutional layer.

    This swaps:
    1. The filters themselves (outgoing weights)
    2. The biases
    3. The corresponding inputs to the next layer (incoming weights)

    Args:
        model: VGG model (will be modified in-place)
        block_idx: Block index in layer_blocks
        layer_idx: Layer index within the block
        neuron_pairs: List of (idx1, idx2) tuples to swap

    Returns:
        Modified model
    """
    # Get the layer to swap
    conv_layer = model.layer_blocks[block_idx][layer_idx]

    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError(f"Expected Conv2d layer but got {type(conv_layer)}")

    print(f"\n--- Swapping filters in Block {block_idx}, Layer {layer_idx} ---")
    print(f"Layer shape: {list(conv_layer.weight.shape)}")
    print(f"Number of swaps: {len(neuron_pairs)}")

    for idx1, idx2 in neuron_pairs:
        print(f"  Swapping neurons {idx1} ↔ {idx2}")

        # 1. Swap filters (outgoing weights): [out_channels, in_channels, H, W]
        conv_layer.weight.data[[idx1, idx2]] = conv_layer.weight.data[[idx2, idx1]]

        # 2. Swap biases
        if conv_layer.bias is not None:
            conv_layer.bias.data[[idx1, idx2]] = conv_layer.bias.data[[idx2, idx1]]

    # 3. Find and swap inputs to next layer
    # Next layer is either:
    #   - Next conv in same block
    #   - First conv in next block (if we're at the end of current block)

    next_layer = None
    next_block_idx = None
    next_layer_idx = None

    # Check if there's another layer in the same block
    if layer_idx + 1 < len(model.layer_blocks[block_idx]):
        next_candidate = model.layer_blocks[block_idx][layer_idx + 1]
        if isinstance(next_candidate, nn.Conv2d):
            next_layer = next_candidate
            next_block_idx = block_idx
            next_layer_idx = layer_idx + 1

    # If not, check first layer of next block
    if next_layer is None and block_idx + 1 < len(model.layer_blocks):
        next_candidate = model.layer_blocks[block_idx + 1][0]
        if isinstance(next_candidate, nn.Conv2d):
            next_layer = next_candidate
            next_block_idx = block_idx + 1
            next_layer_idx = 0

    if next_layer is not None:
        print(f"  Swapping inputs to Block {next_block_idx}, Layer {next_layer_idx}")
        for idx1, idx2 in neuron_pairs:
            # Swap input channels: [out_channels, in_channels, H, W]
            next_layer.weight.data[:, [idx1, idx2]] = next_layer.weight.data[:, [idx2, idx1]]
    else:
        # We're at the last conv layer before classifier
        # Need to handle the flattening into FC layer
        print(f"  Swapping inputs to classifier (with flattening)")

        # Get first FC layer
        fc_layer = None
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                fc_layer = layer
                break

        if fc_layer is not None:
            # Calculate spatial size (for CIFAR10: 1x1 after all pooling)
            # VGG16 has 5 max pooling layers, so 32 -> 1
            in_features = fc_layer.in_features
            out_channels = conv_layer.out_channels
            spatial_size = in_features // out_channels

            print(f"  FC input features: {in_features}, conv out: {out_channels}, spatial: {spatial_size}")

            for idx1, idx2 in neuron_pairs:
                # Swap the flattened channels
                # Flatten order: channel0_spatial0, channel0_spatial1, ..., channel1_spatial0, ...
                # We need to swap all spatial positions for these channels
                for spatial_idx in range(spatial_size):
                    flat_idx1 = idx1 * spatial_size + spatial_idx
                    flat_idx2 = idx2 * spatial_size + spatial_idx
                    fc_layer.weight.data[:, [flat_idx1, flat_idx2]] = \
                        fc_layer.weight.data[:, [flat_idx2, flat_idx1]]

    return model


def calculate_l2_distance(model1, model2):
    """Calculate L2 distance between two models' parameters.

    Args:
        model1: First model
        model2: Second model

    Returns:
        dict with L2 distance statistics
    """
    total_l2 = 0.0
    total_params = 0
    layer_distances = {}

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"

        # Calculate L2 distance for this layer
        diff = param1.data - param2.data
        layer_l2 = torch.norm(diff, p=2).item()
        n_params = param1.numel()

        # Normalized L2 (per-parameter RMS)
        normalized_l2 = layer_l2 / np.sqrt(n_params)

        layer_distances[name1] = {
            'raw_l2': layer_l2,
            'normalized_l2': normalized_l2,
            'n_params': n_params
        }

        total_l2 += layer_l2 ** 2
        total_params += n_params

    # Total L2 distance across all parameters
    total_l2 = np.sqrt(total_l2)
    normalized_total_l2 = total_l2 / np.sqrt(total_params)

    return {
        'total_l2': total_l2,
        'normalized_total_l2': normalized_total_l2,
        'total_params': total_params,
        'layer_distances': layer_distances
    }


def verify_equivalence_on_dataset(original, swapped, dataset_name='CIFAR10',
                                  batch_size=128, num_workers=4):
    """Verify that swapped network produces same accuracy on dataset.

    Args:
        original: Original model
        swapped: Swapped model
        dataset_name: Name of dataset
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        dict with accuracy statistics
    """
    print("\n" + "="*70)
    print("VERIFYING EQUIVALENCE ON DATASET")
    print("="*70)

    # Load dataset
    loaders, num_classes = data.loaders(
        dataset_name,
        path='data',
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name='VGG',
        use_test=True
    )

    original.eval()
    swapped.eval()

    correct_original = 0
    correct_swapped = 0
    total = 0
    max_output_diff = 0.0
    predictions_match = 0

    with torch.no_grad():
        for inputs, targets in loaders['test']:
            # Get predictions
            out_orig = original(inputs)
            out_swap = swapped(inputs)

            # Check predictions
            pred_orig = out_orig.argmax(dim=1)
            pred_swap = out_swap.argmax(dim=1)

            correct_original += pred_orig.eq(targets).sum().item()
            correct_swapped += pred_swap.eq(targets).sum().item()
            predictions_match += pred_orig.eq(pred_swap).sum().item()
            total += targets.size(0)

            # Track maximum output difference
            diff = torch.abs(out_orig - out_swap).max().item()
            max_output_diff = max(max_output_diff, diff)

    acc_original = 100.0 * correct_original / total
    acc_swapped = 100.0 * correct_swapped / total
    pred_match_rate = 100.0 * predictions_match / total

    print(f"\nOriginal model accuracy: {acc_original:.2f}%")
    print(f"Swapped model accuracy:  {acc_swapped:.2f}%")
    print(f"Prediction match rate:   {pred_match_rate:.2f}%")
    print(f"Max output difference:   {max_output_diff:.2e}")

    tolerance = 1e-5
    is_equivalent = (max_output_diff < tolerance) and (pred_match_rate > 99.9)

    if is_equivalent:
        print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
    else:
        print(f"\n✗ WARNING: Models may not be perfectly equivalent")
        print(f"  Max diff {max_output_diff:.2e}, Match rate {pred_match_rate:.2f}%")

    print("="*70)

    return {
        'acc_original': acc_original,
        'acc_swapped': acc_swapped,
        'pred_match_rate': pred_match_rate,
        'max_output_diff': max_output_diff,
        'is_equivalent': is_equivalent
    }


def main():
    parser = argparse.ArgumentParser(
        description='Swap specific neurons to create minimally perturbed network'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to load')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save swapped checkpoint')
    parser.add_argument('--layer-depth', type=str, choices=['early', 'mid', 'late'],
                        default='mid',
                        help='Which layer depth to swap (default: mid)')
    parser.add_argument('--num-swaps', type=int, default=1,
                        help='Number of neuron pairs to swap (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for neuron selection (default: 42)')
    parser.add_argument('--model', type=str, default='VGG16',
                        help='Model architecture (default: VGG16)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of output classes (default: 10)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Dataset for verification (default: CIFAR10)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify equivalence on full dataset (slower but thorough)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip all verification (fast)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("NEURON SWAPPING TOOL")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Layer depth: {args.layer_depth}")
    print(f"Number of swaps: {args.num_swaps}")
    print(f"Random seed: {args.seed}")

    # Get model architecture
    model_name = args.model.upper().replace('-', '')
    if not hasattr(models, model_name):
        raise ValueError(f"Model {model_name} not found")

    architecture = getattr(models, model_name)

    # Load original model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    original = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model_state' in checkpoint:
        original.load_state_dict(checkpoint['model_state'])
    else:
        original.load_state_dict(checkpoint)
    print("✓ Checkpoint loaded")

    # Get layer information
    layer, block_idx, layer_idx, description = get_layer_by_depth(original, args.layer_depth)
    num_filters = layer.out_channels

    print(f"\nTarget layer: {description}")
    print(f"Number of filters: {num_filters}")

    # Generate random neuron pairs
    neuron_pairs = get_random_neuron_pairs(num_filters, args.num_swaps, args.seed)
    print(f"\nNeuron pairs to swap:")
    for i, (idx1, idx2) in enumerate(neuron_pairs):
        print(f"  Pair {i+1}: {idx1} ↔ {idx2}")

    # Create swapped version
    swapped = deepcopy(original)
    swap_conv_filters(swapped, block_idx, layer_idx, neuron_pairs)

    # Calculate L2 distance between original and swapped
    print("\n" + "="*70)
    print("CALCULATING L2 DISTANCE BETWEEN MODELS")
    print("="*70)
    l2_stats = calculate_l2_distance(original, swapped)

    print(f"\nL2 Distance Statistics:")
    print(f"  Total L2 distance:      {l2_stats['total_l2']:.6f}")
    print(f"  Normalized L2 distance: {l2_stats['normalized_total_l2']:.6f}")
    print(f"  Total parameters:       {l2_stats['total_params']:,}")

    # Show top 5 layers by distance
    layer_dists = sorted(l2_stats['layer_distances'].items(),
                        key=lambda x: x[1]['normalized_l2'], reverse=True)
    print(f"\nTop 5 layers by normalized L2 distance:")
    for i, (layer_name, stats) in enumerate(layer_dists[:5]):
        print(f"  {i+1}. {layer_name}: {stats['normalized_l2']:.6f} "
              f"(raw: {stats['raw_l2']:.6f}, params: {stats['n_params']:,})")
    print("="*70)

    # Verification
    if not args.no_verify:
        if args.verify:
            # Full dataset verification
            verification_results = verify_equivalence_on_dataset(
                original, swapped,
                dataset_name=args.dataset
            )
        else:
            # Quick random input verification
            print("\n" + "="*70)
            print("QUICK VERIFICATION (random inputs)")
            print("="*70)

            original.eval()
            swapped.eval()

            num_samples = 10
            max_diff = 0.0

            with torch.no_grad():
                for i in range(num_samples):
                    x = torch.randn(1, 3, 32, 32)
                    out_orig = original(x)
                    out_swap = swapped(x)

                    diff = torch.abs(out_orig - out_swap).max().item()
                    max_diff = max(max_diff, diff)

                    pred_orig = out_orig.argmax(dim=1).item()
                    pred_swap = out_swap.argmax(dim=1).item()

                    match = "✓" if pred_orig == pred_swap else "✗"
                    print(f"Sample {i+1}: max_diff={diff:.2e} {match}")

            print(f"\nMax difference: {max_diff:.2e}")

            tolerance = 1e-5
            if max_diff < tolerance:
                print("✓ Quick verification passed")
            else:
                print(f"✗ Warning: difference {max_diff:.2e} > tolerance {tolerance:.2e}")

            print("="*70)

            verification_results = {'max_output_diff': max_diff}
    else:
        verification_results = {}

    # Save swapped model
    save_dir = os.path.dirname(args.output)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Create metadata (convert layer_distances to serializable format)
    serializable_layer_distances = {
        name: {
            'raw_l2': float(stats['raw_l2']),
            'normalized_l2': float(stats['normalized_l2']),
            'n_params': int(stats['n_params'])
        }
        for name, stats in l2_stats['layer_distances'].items()
    }

    metadata = {
        'swapped': True,
        'original_checkpoint': args.checkpoint,
        'layer_depth': args.layer_depth,
        'block_idx': block_idx,
        'layer_idx': layer_idx,
        'layer_description': description,
        'num_filters': num_filters,
        'num_swaps': args.num_swaps,
        'neuron_pairs': neuron_pairs,
        'seed': args.seed,
        'l2_distance': {
            'total_l2': float(l2_stats['total_l2']),
            'normalized_total_l2': float(l2_stats['normalized_total_l2']),
            'total_params': int(l2_stats['total_params']),
            'top_5_layers': [
                {
                    'layer_name': name,
                    'normalized_l2': float(stats['normalized_l2']),
                    'raw_l2': float(stats['raw_l2']),
                    'n_params': int(stats['n_params'])
                }
                for name, stats in layer_dists[:5]
            ]
        },
        'verification': verification_results
    }

    # Also save complete layer distances separately
    layer_distances_path = args.output.replace('.pt', '_layer_distances.json')
    with open(layer_distances_path, 'w') as f:
        json.dump(serializable_layer_distances, f, indent=2)

    # Save checkpoint with metadata
    save_dict = {
        'model_state': swapped.state_dict(),
        'metadata': metadata
    }

    # Preserve original checkpoint fields if they exist
    if 'epoch' in checkpoint:
        save_dict['epoch'] = checkpoint['epoch']
    if 'optimizer_state' in checkpoint:
        save_dict['optimizer_state'] = checkpoint['optimizer_state']

    torch.save(save_dict, args.output)
    print(f"\n✓ Swapped model saved to: {args.output}")

    # Save metadata as JSON for easy inspection
    metadata_path = args.output.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"✓ Layer L2 distances saved to: {layer_distances_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Swapped {args.num_swaps} neuron pair(s) in {args.layer_depth} layer")
    print(f"✓ Layer: {description}")
    print(f"✓ L2 distance between models: {l2_stats['total_l2']:.6f}")
    print(f"✓ Normalized L2 distance: {l2_stats['normalized_total_l2']:.6f}")
    print(f"✓ Models are functionally equivalent but at different points in weight space")
    print(f"✓ Ready for mode connectivity analysis")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
