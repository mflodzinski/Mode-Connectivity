"""
Neuron swapping utilities for creating minimally perturbed networks.

Provides utilities for swapping specific neurons to test local vs global
mode connectivity.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Optional

from ..transform import permutation
from ..core import data as lib_data
from ..evaluation import evaluate as evaluation, metrics


class NeuronSwapper:
    """Swap specific neurons to create minimally perturbed networks."""

    # VGG16 layer information for depth selection
    VGG16_LAYER_INFO = {
        'early': {
            'block_idx': 0,
            'layer_idx': 0,
            'description': 'Block 0, Conv 0 (64 filters, early features)'
        },
        'mid': {
            'block_idx': 2,
            'layer_idx': 1,
            'description': 'Block 2, Conv 1 (256 filters, mid-level features)'
        },
        'late': {
            'block_idx': 4,
            'layer_idx': 1,
            'description': 'Block 4, Conv 1 (512 filters, late features)'
        }
    }

    def __init__(self, model: nn.Module, architecture: str = 'VGG16'):
        """Initialize NeuronSwapper.

        Args:
            model: Model to swap neurons in
            architecture: Architecture name (default: 'VGG16')
        """
        self.original = model
        self.architecture = architecture
        self.swapped = None
        self.swap_metadata = {}

    def get_layer_by_depth(self, depth: str) -> Tuple[nn.Module, int, int, str]:
        """Get a convolutional layer based on depth specification.

        Args:
            depth: 'early', 'mid', or 'late'

        Returns:
            Tuple of (layer, block_idx, layer_idx, description)
        """
        if depth not in self.VGG16_LAYER_INFO:
            raise ValueError(f"Depth must be one of {list(self.VGG16_LAYER_INFO.keys())}")

        info = self.VGG16_LAYER_INFO[depth]
        block_idx = info['block_idx']
        layer_idx = info['layer_idx']

        # Navigate to the layer
        layer = self.original.layer_blocks[block_idx][layer_idx]

        if not isinstance(layer, nn.Conv2d):
            raise ValueError(f"Expected Conv2d layer but got {type(layer)}")

        return layer, block_idx, layer_idx, info['description']

    def swap_filters(self,
                    block_idx: int,
                    layer_idx: int,
                    neuron_pairs: List[Tuple[int, int]],
                    verbose: bool = True) -> nn.Module:
        """Swap filters in a convolutional layer maintaining equivalence.

        Args:
            block_idx: Block index in layer_blocks
            layer_idx: Layer index within the block
            neuron_pairs: List of (idx1, idx2) tuples to swap
            verbose: Whether to print swap information

        Returns:
            Swapped model
        """
        # Create swapped model if not already created
        if self.swapped is None:
            self.swapped = deepcopy(self.original)

        # Use permutation utils to perform the swap
        permutation.PermutationUtils.swap_conv_filters(
            self.swapped,
            block_idx,
            layer_idx,
            neuron_pairs,
            verbose=verbose
        )

        # Store metadata
        self.swap_metadata = {
            'block_idx': block_idx,
            'layer_idx': layer_idx,
            'neuron_pairs': neuron_pairs,
            'num_swaps': len(neuron_pairs)
        }

        return self.swapped

    def swap_by_depth(self,
                     depth: str,
                     num_swaps: int = 1,
                     seed: int = 42,
                     verbose: bool = True) -> nn.Module:
        """Swap neurons at a specific depth level.

        Args:
            depth: 'early', 'mid', or 'late'
            num_swaps: Number of neuron pairs to swap
            seed: Random seed for reproducibility
            verbose: Whether to print swap information

        Returns:
            Swapped model
        """
        # Get layer information
        layer, block_idx, layer_idx, description = self.get_layer_by_depth(depth)

        if verbose:
            print(f"\n{'='*70}")
            print(f"SWAPPING NEURONS AT {depth.upper()} DEPTH")
            print(f"{'='*70}")
            print(f"\nTarget layer: {description}")
            print(f"Number of filters: {layer.out_channels}")
            print(f"Number of swaps: {num_swaps}")

        # Generate random neuron pairs
        neuron_pairs = permutation.PermutationUtils.get_random_neuron_pairs(
            num_filters=layer.out_channels,
            num_pairs=num_swaps,
            seed=seed
        )

        # Store depth info in metadata
        self.swap_metadata['depth'] = depth
        self.swap_metadata['description'] = description

        # Perform swap
        return self.swap_filters(block_idx, layer_idx, neuron_pairs, verbose=verbose)

    def calculate_l2_distance(self, verbose: bool = True) -> Dict[str, Any]:
        """Calculate L2 distance between original and swapped models.

        Args:
            verbose: Whether to print distance information

        Returns:
            Dictionary with L2 distance statistics
        """
        if self.swapped is None:
            raise ValueError("Must create swapped model before calculating distance")

        # Use metrics module
        original_state = self.original.state_dict()
        swapped_state = self.swapped.state_dict()

        distance_stats = metrics.l2_distance(
            original_state,
            swapped_state,
            compute_per_layer=True
        )

        if verbose:
            print(f"\n{'='*70}")
            print("L2 DISTANCE BETWEEN ORIGINAL AND SWAPPED MODELS")
            print(f"{'='*70}")
            print(f"\nTotal L2 distance:      {distance_stats['total_l2']:.6f}")
            print(f"Normalized L2 distance: {distance_stats['normalized_total_l2']:.6f}")
            print(f"Total parameters:       {distance_stats['total_params']:,}")

            # Show top affected layers
            layer_dists = sorted(
                distance_stats['layer_distances'].items(),
                key=lambda x: x[1]['normalized_l2'],
                reverse=True
            )

            print(f"\nTop 5 most affected layers:")
            print("-" * 70)
            for i, (layer_name, stats) in enumerate(layer_dists[:5]):
                print(f"{i+1}. {layer_name[:50]:<50} {stats['normalized_l2']:.6f}")
            print("=" * 70)

        return distance_stats

    def verify_equivalence_on_dataset(self,
                                     dataset_name: str = 'CIFAR10',
                                     data_path: str = './data',
                                     batch_size: int = 128,
                                     num_workers: int = 4,
                                     verbose: bool = True) -> Dict[str, Any]:
        """Verify that swapped network produces same accuracy on dataset.

        Args:
            dataset_name: Name of dataset
            data_path: Path to dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            verbose: Whether to print verification details

        Returns:
            Dictionary with accuracy statistics
        """
        if self.swapped is None:
            raise ValueError("Must create swapped model before verifying equivalence")

        if verbose:
            print(f"\n{'='*70}")
            print("VERIFYING EQUIVALENCE ON DATASET")
            print(f"{'='*70}")

        # Load dataset
        loaders, num_classes = lib_data.get_loaders(
            dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_name='VGG',
            use_test=True
        )

        self.original.eval()
        self.swapped.eval()

        correct_original = 0
        correct_swapped = 0
        total = 0
        max_output_diff = 0.0
        predictions_match = 0

        with torch.no_grad():
            for inputs, targets in loaders['test']:
                # Get predictions
                out_orig = self.original(inputs)
                out_swap = self.swapped(inputs)

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

        if verbose:
            print(f"\nOriginal model accuracy: {acc_original:.2f}%")
            print(f"Swapped model accuracy:  {acc_swapped:.2f}%")
            print(f"Prediction match rate:   {pred_match_rate:.2f}%")
            print(f"Max output difference:   {max_output_diff:.2e}")

        tolerance = 1e-5
        is_equivalent = (max_output_diff < tolerance) and (pred_match_rate > 99.9)

        if verbose:
            if is_equivalent:
                print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
            else:
                print(f"\n✗ WARNING: Models may not be perfectly equivalent")
                print(f"  Max diff {max_output_diff:.2e}, Match rate {pred_match_rate:.2f}%")
            print("=" * 70)

        return {
            'acc_original': acc_original,
            'acc_swapped': acc_swapped,
            'pred_match_rate': pred_match_rate,
            'max_output_diff': max_output_diff,
            'is_equivalent': is_equivalent
        }

    def save(self, output_path: str, include_original: bool = False):
        """Save swapped model with metadata.

        Args:
            output_path: Path to save swapped checkpoint
            include_original: Whether to include original model for comparison
        """
        if self.swapped is None:
            raise ValueError("Must create swapped model before saving")

        checkpoint = {
            'model_state': self.swapped.state_dict(),
            'swap_metadata': self.swap_metadata,
            'architecture': self.architecture
        }

        if include_original:
            checkpoint['original_state'] = self.original.state_dict()

        torch.save(checkpoint, output_path)
