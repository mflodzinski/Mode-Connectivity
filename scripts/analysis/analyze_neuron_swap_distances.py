"""Analyze layer-wise weight distances along the mode connectivity curve.

This script evaluates how much each layer changes along the curve between the
original network and the neuron-swapped version. It answers the question:
- Local correction: Only the swapped layer changes?
- Global adjustment: All layers change?
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
from tqdm import tqdm

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import models
import curves
import data


def get_layer_names_and_params(model):
    """Extract all layer names and parameter counts.

    Returns:
        list of (layer_name, num_params, layer_type) tuples
    """
    layer_info = []

    # Convolutional layers
    if hasattr(model, 'layer_blocks'):
        for block_idx, block in enumerate(model.layer_blocks):
            for layer_idx, layer in enumerate(block):
                if isinstance(layer, (nn.Conv2d, curves.Conv2d)):
                    name = f"block{block_idx}_conv{layer_idx}"
                    num_params = sum(p.numel() for p in layer.parameters())
                    layer_info.append((name, num_params, 'conv', block_idx, layer_idx, layer))

    # Classifier layers
    if hasattr(model, 'classifier'):
        for layer_idx, layer in enumerate(model.classifier):
            if isinstance(layer, (nn.Linear, curves.Linear)):
                name = f"fc{layer_idx}"
                num_params = sum(p.numel() for p in layer.parameters())
                layer_info.append((name, num_params, 'fc', None, layer_idx, layer))

    # Alternative: if model uses fc1, fc2, fc3 directly
    for fc_name in ['fc1', 'fc2', 'fc3']:
        if hasattr(model, fc_name):
            layer = getattr(model, fc_name)
            if isinstance(layer, (nn.Linear, curves.Linear)):
                name = fc_name
                num_params = sum(p.numel() for p in layer.parameters())
                layer_info.append((name, num_params, 'fc', None, None, layer))

    return layer_info


def compute_layer_distance(weights_current, weights_reference):
    """Compute normalized L2 distance between two sets of weights.

    Args:
        weights_current: Current weights (tensor or dict of tensors)
        weights_reference: Reference weights (tensor or dict of tensors)

    Returns:
        dict with various distance metrics
    """
    # Handle both single tensors and state dicts
    if isinstance(weights_current, dict):
        # Concatenate all parameters
        current_flat = torch.cat([p.flatten() for p in weights_current.values()])
        ref_flat = torch.cat([p.flatten() for p in weights_reference.values()])
    else:
        current_flat = weights_current.flatten()
        ref_flat = weights_reference.flatten()

    # Compute distances
    diff = current_flat - ref_flat
    n_params = current_flat.numel()

    raw_l2 = torch.norm(diff, p=2).item()
    normalized_l2 = raw_l2 / np.sqrt(n_params)

    # Relative distance (percentage change)
    ref_norm = torch.norm(ref_flat, p=2).item()
    relative_distance = raw_l2 / ref_norm if ref_norm > 0 else 0

    return {
        'raw_l2': raw_l2,
        'normalized_l2': normalized_l2,
        'relative_distance': relative_distance,
        'n_params': n_params
    }


def load_curve_model_at_t(curve_model, curve_checkpoint, t_value):
    """Load curve model and extract weights at specific t value.

    Args:
        curve_model: CurveNet model
        curve_checkpoint: Path to curve checkpoint
        t_value: Value along curve [0, 1]

    Returns:
        State dict at t_value
    """
    # Load checkpoint
    checkpoint = torch.load(curve_checkpoint, map_location='cpu')
    curve_model.load_state_dict(checkpoint['model_state'])
    curve_model.eval()

    # Get coefficients for this t value
    coeffs_t = curve_model.curve.coeffs_t(torch.tensor([t_value]))

    # Extract weights at this t
    state_dict_t = {}

    # Process each curve parameter
    for name, module in curve_model.named_modules():
        if isinstance(module, (curves.Conv2d, curves.Linear)):
            # Compute weighted combination of bend points
            weight_t = sum(coeffs_t[0, i].item() * getattr(module, f'weight_{i}').data
                          for i in range(module.num_bends))
            state_dict_t[f"{name}.weight"] = weight_t

            if module.bias_0 is not None:
                bias_t = sum(coeffs_t[0, i].item() * getattr(module, f'bias_{i}').data
                            for i in range(module.num_bends))
                state_dict_t[f"{name}.bias"] = bias_t

    return state_dict_t


def analyze_curve(curve_checkpoint, original_checkpoint, swap_metadata_path,
                 output_dir, num_points=61):
    """Analyze layer-wise distances along the curve.

    Args:
        curve_checkpoint: Path to trained curve checkpoint
        original_checkpoint: Path to original (seed0) checkpoint
        swap_metadata_path: Path to swap metadata JSON
        output_dir: Directory to save results
        num_points: Number of points to evaluate along curve

    Returns:
        Dictionary with analysis results
    """
    print("="*70)
    print("ANALYZING NEURON SWAP CURVE")
    print("="*70)

    # Load metadata
    with open(swap_metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nSwap configuration:")
    print(f"  Layer depth: {metadata['layer_depth']}")
    print(f"  Layer: {metadata['layer_description']}")
    print(f"  Block {metadata['block_idx']}, Layer {metadata['layer_idx']}")
    print(f"  Number of swaps: {metadata['num_swaps']}")
    print(f"  Swapped neuron pairs: {metadata['neuron_pairs']}")

    # Load original model
    print(f"\nLoading original model...")
    architecture = getattr(models, 'VGG16')
    original_model = architecture.base(num_classes=10, **architecture.kwargs)
    orig_checkpoint = torch.load(original_checkpoint, map_location='cpu')
    original_model.load_state_dict(orig_checkpoint['model_state'])
    original_model.eval()

    # Get layer structure
    layer_info = get_layer_names_and_params(original_model)
    print(f"\nFound {len(layer_info)} layers to analyze")

    # Create curve model
    print(f"\nLoading curve model...")
    curve = getattr(curves, 'Bezier')
    curve_model = curves.CurveNet(
        num_classes=10,
        curve=curve,
        architecture=architecture.curve,
        num_bends=3,
        fix_start=True,
        fix_end=True,
        architecture_kwargs=architecture.kwargs
    )

    # Evaluate at multiple points
    print(f"\nEvaluating curve at {num_points} points...")
    t_values = np.linspace(0, 1, num_points)

    # Storage for results
    results = {
        'layer_names': [info[0] for info in layer_info],
        'layer_types': [info[2] for info in layer_info],
        'layer_params': [info[1] for info in layer_info],
        't_values': t_values.tolist(),
        'distances': {
            'normalized_l2': np.zeros((num_points, len(layer_info))),
            'relative': np.zeros((num_points, len(layer_info))),
            'raw_l2': np.zeros((num_points, len(layer_info)))
        },
        'metadata': metadata
    }

    # Get original layer states
    original_layer_states = {}
    for name, _, _, block_idx, layer_idx, layer in layer_info:
        layer_params = {
            'weight': layer.weight.data.clone(),
        }
        if layer.bias is not None:
            layer_params['bias'] = layer.bias.data.clone()
        original_layer_states[name] = layer_params

    # Analyze each t value
    for t_idx, t in enumerate(tqdm(t_values, desc="Analyzing curve")):
        # Load curve state at t
        state_dict_t = load_curve_model_at_t(curve_model, curve_checkpoint, t)

        # Compute distance for each layer
        for layer_idx, (name, n_params, layer_type, block_idx, idx_in_block, layer) in enumerate(layer_info):
            # Find corresponding weights in curve state dict
            # Match by layer structure
            layer_weights_t = {}

            if layer_type == 'conv':
                # Match convolutional layer
                key_prefix = f"net.layer_blocks.{block_idx}.{idx_in_block}"
            else:
                # Match FC layer
                if name.startswith('fc') and name in ['fc1', 'fc2', 'fc3']:
                    key_prefix = f"net.{name}"
                else:
                    # Classifier layers
                    key_prefix = f"net.classifier.{idx_in_block}"

            # Extract weights for this layer
            for key in state_dict_t.keys():
                if key.startswith(key_prefix):
                    param_name = key.split('.')[-1]  # 'weight' or 'bias'
                    layer_weights_t[param_name] = state_dict_t[key]

            # Compute distance from original
            if layer_weights_t:
                dist = compute_layer_distance(layer_weights_t, original_layer_states[name])
                results['distances']['normalized_l2'][t_idx, layer_idx] = dist['normalized_l2']
                results['distances']['relative'][t_idx, layer_idx] = dist['relative_distance']
                results['distances']['raw_l2'][t_idx, layer_idx] = dist['raw_l2']

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'layer_distances_along_curve.npz')

    np.savez(output_path,
             layer_names=results['layer_names'],
             layer_types=results['layer_types'],
             layer_params=results['layer_params'],
             t_values=results['t_values'],
             normalized_l2=results['distances']['normalized_l2'],
             relative=results['distances']['relative'],
             raw_l2=results['distances']['raw_l2'],
             swapped_block=metadata['block_idx'],
             swapped_layer=metadata['layer_idx'])

    # Also save as readable JSON (excluding large arrays)
    summary = {
        'layer_names': results['layer_names'],
        'layer_types': results['layer_types'],
        'layer_params': results['layer_params'],
        'num_t_points': num_points,
        'swapped_layer': metadata['layer_description'],
        'swapped_block': metadata['block_idx'],
        'swapped_layer_idx': metadata['layer_idx'],
        'neuron_pairs': metadata['neuron_pairs']
    }

    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Summary saved to: {os.path.join(output_dir, 'analysis_summary.json')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze layer-wise weight distances along mode connectivity curve'
    )
    parser.add_argument('--curve-checkpoint', type=str, required=True,
                        help='Path to trained curve checkpoint')
    parser.add_argument('--original-checkpoint', type=str, required=True,
                        help='Path to original endpoint checkpoint')
    parser.add_argument('--swap-metadata', type=str, required=True,
                        help='Path to swap metadata JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save analysis results')
    parser.add_argument('--num-points', type=int, default=61,
                        help='Number of points to evaluate along curve (default: 61)')

    args = parser.parse_args()

    results = analyze_curve(
        args.curve_checkpoint,
        args.original_checkpoint,
        args.swap_metadata,
        args.output_dir,
        args.num_points
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAnalyzed {len(results['layer_names'])} layers across {args.num_points} points")
    print(f"Results saved to: {args.output_dir}")
    print("\nNext step: Create visualization with:")
    print(f"  python scripts/plotting/plot_layer_distance_animation.py \\")
    print(f"    --data {os.path.join(args.output_dir, 'layer_distances_along_curve.npz')} \\")
    print(f"    --output {os.path.join(args.output_dir, 'layer_distances_evolution.gif')}")
    print("="*70)


if __name__ == "__main__":
    main()
