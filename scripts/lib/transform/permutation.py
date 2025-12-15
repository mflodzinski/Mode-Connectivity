"""
Neuron permutation utilities for creating functionally equivalent networks.

Provides utilities for permuting neurons/filters in neural networks while
maintaining functional equivalence.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class PermutationUtils:
    """Utilities for neuron permutation operations."""

    @staticmethod
    def get_reverse_permutation(n: int) -> np.ndarray:
        """Generate reverse-order permutation indices.

        Args:
            n: Number of elements

        Returns:
            numpy array of indices in reverse order [n-1, n-2, ..., 1, 0]
        """
        return np.arange(n)[::-1].copy()  # .copy() to avoid negative stride issues

    @staticmethod
    def permute_conv_layer(conv_layer: nn.Conv2d,
                          perm_in: np.ndarray,
                          perm_out: np.ndarray) -> nn.Conv2d:
        """Permute a convolutional layer's weights and biases.

        Args:
            conv_layer: nn.Conv2d layer
            perm_in: Permutation for input channels (from previous layer)
            perm_out: Permutation for output channels (filters)

        Returns:
            Permuted conv layer (modified in-place)
        """
        # Permute weights: [out_channels, in_channels, H, W]
        # First permute output channels, then input channels
        conv_layer.weight.data = conv_layer.weight.data[perm_out, :, :, :]
        conv_layer.weight.data = conv_layer.weight.data[:, perm_in, :, :]

        # Permute bias if it exists
        if conv_layer.bias is not None:
            conv_layer.bias.data = conv_layer.bias.data[perm_out]

        return conv_layer

    @staticmethod
    def permute_bn_layer(bn_layer: nn.BatchNorm2d, perm: np.ndarray) -> nn.BatchNorm2d:
        """Permute a batch normalization layer's parameters.

        Args:
            bn_layer: nn.BatchNorm2d layer
            perm: Permutation for channels

        Returns:
            Permuted BN layer (modified in-place)
        """
        # Permute scale (γ) and shift (β)
        if bn_layer.weight is not None:
            bn_layer.weight.data = bn_layer.weight.data[perm]
        if bn_layer.bias is not None:
            bn_layer.bias.data = bn_layer.bias.data[perm]

        # Permute running statistics
        if bn_layer.running_mean is not None:
            bn_layer.running_mean = bn_layer.running_mean[perm]
        if bn_layer.running_var is not None:
            bn_layer.running_var = bn_layer.running_var[perm]

        return bn_layer

    @staticmethod
    def permute_fc_layer(fc_layer: nn.Linear,
                        perm_in: np.ndarray,
                        perm_out: Optional[np.ndarray] = None) -> nn.Linear:
        """Permute a fully connected layer's weights and biases.

        Args:
            fc_layer: nn.Linear layer
            perm_in: Permutation for input neurons
            perm_out: Permutation for output neurons (None for final layer)

        Returns:
            Permuted FC layer (modified in-place)
        """
        # Permute weights: [out_features, in_features]
        fc_layer.weight.data = fc_layer.weight.data[:, perm_in]

        # Only permute output if not the final classification layer
        if perm_out is not None:
            fc_layer.weight.data = fc_layer.weight.data[perm_out, :]
            if fc_layer.bias is not None:
                fc_layer.bias.data = fc_layer.bias.data[perm_out]

        return fc_layer

    @staticmethod
    def get_random_neuron_pairs(num_filters: int,
                               num_pairs: int,
                               seed: int = 42) -> List[Tuple[int, int]]:
        """Generate random pairs of neuron indices to swap.

        Args:
            num_filters: Total number of filters/neurons in the layer
            num_pairs: Number of pairs to generate
            seed: Random seed for reproducibility

        Returns:
            List of (idx1, idx2) tuples representing neuron pairs to swap
        """
        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(num_filters)

        pairs = []
        for i in range(min(num_pairs, num_filters // 2)):
            pairs.append((int(all_indices[2*i]), int(all_indices[2*i + 1])))

        return pairs

    @staticmethod
    def swap_conv_filters(model: nn.Module,
                         block_idx: int,
                         layer_idx: int,
                         neuron_pairs: List[Tuple[int, int]],
                         verbose: bool = True) -> nn.Module:
        """Swap filters in a convolutional layer maintaining equivalence.

        This swaps:
        1. The filters themselves (outgoing weights)
        2. The biases
        3. The corresponding inputs to the next layer (incoming weights)

        Args:
            model: VGG model (will be modified in-place)
            block_idx: Block index in layer_blocks
            layer_idx: Layer index within the block
            neuron_pairs: List of (idx1, idx2) tuples to swap
            verbose: Whether to print swap information

        Returns:
            Modified model
        """
        # Get the layer to swap
        conv_layer = model.layer_blocks[block_idx][layer_idx]

        if not isinstance(conv_layer, nn.Conv2d):
            raise ValueError(f"Expected Conv2d layer but got {type(conv_layer)}")

        if verbose:
            print(f"\n--- Swapping filters in Block {block_idx}, Layer {layer_idx} ---")
            print(f"Layer shape: {list(conv_layer.weight.shape)}")
            print(f"Number of swaps: {len(neuron_pairs)}")

        for idx1, idx2 in neuron_pairs:
            if verbose:
                print(f"  Swapping neurons {idx1} ↔ {idx2}")

            # 1. Swap filters (outgoing weights): [out_channels, in_channels, H, W]
            conv_layer.weight.data[[idx1, idx2]] = conv_layer.weight.data[[idx2, idx1]]

            # 2. Swap biases
            if conv_layer.bias is not None:
                conv_layer.bias.data[[idx1, idx2]] = conv_layer.bias.data[[idx2, idx1]]

        # 3. Find and swap inputs to next layer
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
            if verbose:
                print(f"  Swapping inputs to Block {next_block_idx}, Layer {next_layer_idx}")
            for idx1, idx2 in neuron_pairs:
                # Swap input channels: [out_channels, in_channels, H, W]
                next_layer.weight.data[:, [idx1, idx2]] = next_layer.weight.data[:, [idx2, idx1]]
        else:
            # We're at the last conv layer before classifier
            if verbose:
                print(f"  Swapping inputs to classifier (with flattening)")

            # Get first FC layer
            fc_layer = None
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    fc_layer = layer
                    break

            if fc_layer is not None:
                in_features = fc_layer.in_features
                out_channels = conv_layer.out_channels
                spatial_size = in_features // out_channels

                if verbose:
                    print(f"  FC input features: {in_features}, conv out: {out_channels}, spatial: {spatial_size}")

                for idx1, idx2 in neuron_pairs:
                    # Swap the flattened channels
                    for spatial_idx in range(spatial_size):
                        fc_idx1 = idx1 * spatial_size + spatial_idx
                        fc_idx2 = idx2 * spatial_size + spatial_idx
                        fc_layer.weight.data[:, [fc_idx1, fc_idx2]] = fc_layer.weight.data[:, [fc_idx2, fc_idx1]]

        return model
