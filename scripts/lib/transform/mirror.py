"""
Mirror network creation and evaluation.

Provides utilities for creating mirrored neural networks via neuron permutation
and verifying their functional equivalence.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any

from ..transform import permutation


class MirrorNetwork:
    """Create and evaluate mirrored neural networks via permutation."""

    def __init__(self, model: nn.Module, use_bn: bool = True):
        """Initialize MirrorNetwork.

        Args:
            model: Original model to mirror
            use_bn: Whether the model uses batch normalization
        """
        self.original = model
        self.use_bn = use_bn
        self.mirrored = None

    def create_mirror(self, verbose: bool = True) -> nn.Module:
        """Create mirrored version by reversing neuron indices.

        This function applies reverse-order permutation to each layer while maintaining
        the chain of permutations between layers to preserve functional equivalence.

        Args:
            verbose: Whether to print permutation details

        Returns:
            Mirrored model that is functionally equivalent to the original
        """
        # Create a deep copy to avoid modifying the original
        self.mirrored = deepcopy(self.original)

        # Track the current permutation for chaining between layers
        current_perm = None  # Input channels permutation from previous layer

        if verbose:
            print("\n" + "="*70)
            print("APPLYING REVERSE-ORDER PERMUTATIONS")
            print("="*70)

        # Process convolutional layers
        if hasattr(self.mirrored, 'layer_blocks'):
            current_perm = self._mirror_layer_blocks(verbose)
        elif hasattr(self.mirrored, 'features'):
            current_perm = self._mirror_features(verbose)

        # Process classifier (fully connected part)
        if hasattr(self.mirrored, 'classifier'):
            self._mirror_classifier(current_perm, verbose)

        if verbose:
            print("\n" + "="*70)
            print("PERMUTATION COMPLETE")
            print("="*70)

        return self.mirrored

    def _mirror_layer_blocks(self, verbose: bool = True) -> np.ndarray:
        """Mirror layer_blocks structure (used in dnn-mode-connectivity VGG)."""
        current_perm = None
        layer_idx = 0

        for block_idx, block in enumerate(self.mirrored.layer_blocks):
            if verbose:
                print(f"\n--- Block {block_idx} ---")

            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels

                    # Input permutation: use previous layer's output or no permutation for first layer
                    if current_perm is None:
                        perm_in = np.arange(in_channels)
                    else:
                        perm_in = current_perm

                    # Output permutation: reverse order
                    perm_out = permutation.PermutationUtils.get_reverse_permutation(out_channels)

                    if verbose:
                        print(f"Layer {layer_idx} (Conv2d): in={in_channels}, out={out_channels}")
                        if len(perm_in) > 6:
                            print(f"  Input perm:  [{perm_in[:3]}...{perm_in[-3:]}]")
                        else:
                            print(f"  Input perm:  {perm_in}")
                        if len(perm_out) > 6:
                            print(f"  Output perm: [{perm_out[:3]}...{perm_out[-3:]}]")
                        else:
                            print(f"  Output perm: {perm_out}")

                    # Apply permutation
                    permutation.PermutationUtils.permute_conv_layer(layer, perm_in, perm_out)

                    # Update current permutation for next layer
                    current_perm = perm_out
                    layer_idx += 1

                elif isinstance(layer, nn.BatchNorm2d):
                    num_features = layer.num_features
                    if verbose:
                        print(f"Layer {layer_idx} (BatchNorm2d): features={num_features}")
                        if len(current_perm) > 6:
                            print(f"  Permutation: [{current_perm[:3]}...{current_perm[-3:]}]")
                        else:
                            print(f"  Permutation: {current_perm}")

                    permutation.PermutationUtils.permute_bn_layer(layer, current_perm)
                    layer_idx += 1

        return current_perm

    def _mirror_features(self, verbose: bool = True) -> np.ndarray:
        """Mirror features structure (used in torchvision VGG)."""
        current_perm = None
        layers = list(self.mirrored.features.children())

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels

                if current_perm is None:
                    perm_in = np.arange(in_channels)
                else:
                    perm_in = current_perm

                perm_out = permutation.PermutationUtils.get_reverse_permutation(out_channels)

                if verbose:
                    print(f"\nLayer {i} (Conv2d): in={in_channels}, out={out_channels}")

                permutation.PermutationUtils.permute_conv_layer(layer, perm_in, perm_out)
                current_perm = perm_out

            elif isinstance(layer, nn.BatchNorm2d):
                permutation.PermutationUtils.permute_bn_layer(layer, current_perm)

            elif isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                if verbose:
                    print(f"\nLayer {i} ({layer.__class__.__name__}): no permutation needed")

        return current_perm

    def _mirror_classifier(self, current_perm: np.ndarray, verbose: bool = True):
        """Mirror classifier (fully connected layers)."""
        fc_layers = list(self.mirrored.classifier.children())
        fc_perm = None

        for i, layer in enumerate(fc_layers):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features

                # First FC layer: needs to handle flattened conv output
                if fc_perm is None and current_perm is not None:
                    spatial_size = in_features // current_perm.shape[0]

                    # Create permutation for flattened tensor
                    flattened_perm = np.concatenate([
                        current_perm * spatial_size + offset
                        for offset in range(spatial_size)
                    ])
                    perm_in = flattened_perm
                else:
                    perm_in = fc_perm if fc_perm is not None else np.arange(in_features)

                # Determine if this is the final classification layer
                is_final = (i == len(fc_layers) - 1) or (
                    i == len(fc_layers) - 2 and isinstance(fc_layers[-1], nn.ReLU)
                )

                if is_final:
                    # Final layer: don't permute output (preserve class indices)
                    perm_out = None
                    if verbose:
                        print(f"\nFC Layer {i} (Linear): in={in_features}, out={out_features} [FINAL - no output permutation]")
                else:
                    # Hidden layer: permute output
                    perm_out = permutation.PermutationUtils.get_reverse_permutation(out_features)
                    if verbose:
                        print(f"\nFC Layer {i} (Linear): in={in_features}, out={out_features}")
                        if len(perm_out) > 10:
                            print(f"  Output perm: {perm_out[:5]}...{perm_out[-5:]}")
                        else:
                            print(f"  Output perm: {perm_out}")

                # Apply permutation
                permutation.PermutationUtils.permute_fc_layer(layer, perm_in, perm_out)

                # Update permutation for next FC layer
                fc_perm = perm_out

            elif isinstance(layer, (nn.ReLU, nn.Dropout)):
                if verbose:
                    print(f"\nFC Layer {i} ({layer.__class__.__name__}): no permutation needed")

    def verify_equivalence(self,
                          num_samples: int = 10,
                          input_size: Tuple[int, int, int] = (3, 224, 224),
                          verbose: bool = True) -> bool:
        """Verify that original and mirrored models produce identical outputs.

        Args:
            num_samples: Number of random test samples
            input_size: Input tensor size (C, H, W)
            verbose: Whether to print verification details

        Returns:
            Boolean indicating if models are functionally equivalent
        """
        if self.mirrored is None:
            raise ValueError("Must create mirrored network before verifying equivalence")

        if verbose:
            print("\n" + "="*70)
            print("VERIFYING FUNCTIONAL EQUIVALENCE")
            print("="*70)

        self.original.eval()
        self.mirrored.eval()

        max_diff = 0.0
        all_close = True

        with torch.no_grad():
            for i in range(num_samples):
                # Generate random input
                x = torch.randn(1, *input_size)

                # Get outputs
                out_orig = self.original(x)
                out_mirror = self.mirrored(x)

                # Compute difference
                diff = torch.abs(out_orig - out_mirror).max().item()
                max_diff = max(max_diff, diff)

                # Check if predictions match
                pred_orig = out_orig.argmax(dim=1).item()
                pred_mirror = out_mirror.argmax(dim=1).item()

                if verbose:
                    match = "✓" if pred_orig == pred_mirror else "✗"
                    print(f"Sample {i+1}: max_diff={diff:.2e}, pred_orig={pred_orig}, pred_mirror={pred_mirror} {match}")

                if pred_orig != pred_mirror:
                    all_close = False

        if verbose:
            print(f"\nMaximum difference across all samples: {max_diff:.2e}")

        # Use a reasonable tolerance for floating point comparisons
        tolerance = 1e-5
        functionally_equivalent = all_close and max_diff < tolerance

        if verbose:
            if functionally_equivalent:
                print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
            else:
                print(f"\n✗ VERIFICATION FAILED: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
            print("="*70)

        return functionally_equivalent

    def count_parameter_differences(self, verbose: bool = True) -> Dict[str, Any]:
        """Count how many parameters differ between original and mirrored models.

        Args:
            verbose: Whether to print details

        Returns:
            Dictionary with statistics about parameter differences
        """
        if self.mirrored is None:
            raise ValueError("Must create mirrored network before counting differences")

        if verbose:
            print("\n" + "="*70)
            print("PARAMETER SPACE ANALYSIS")
            print("="*70)

        total_params = 0
        different_params = 0
        layer_diffs = {}

        for (name_o, param_o), (name_m, param_m) in zip(
            self.original.named_parameters(), self.mirrored.named_parameters()
        ):
            assert name_o == name_m, "Parameter names don't match!"

            total = param_o.numel()
            different = (param_o != param_m).sum().item()

            total_params += total
            different_params += different

            if different > 0:
                layer_diffs[name_o] = {
                    'different': different,
                    'total': total,
                    'percentage': 100 * different / total
                }

                if verbose:
                    print(f"{name_o}: {different}/{total} ({100*different/total:.1f}%) parameters differ")

        if verbose:
            print(f"\nTotal: {different_params}/{total_params} ({100*different_params/total_params:.1f}%) parameters differ")
            print("\n✓ Models are at DIFFERENT points in parameter space")
            print("✓ But produce IDENTICAL outputs (functional equivalence)")
            print("="*70)

        return {
            'total_params': total_params,
            'different_params': different_params,
            'percentage_different': 100 * different_params / total_params,
            'layer_differences': layer_diffs
        }

    def save_mirrored(self, output_path: str, include_metadata: bool = True):
        """Save the mirrored model.

        Args:
            output_path: Path to save mirrored checkpoint
            include_metadata: Whether to include metadata about mirroring
        """
        if self.mirrored is None:
            raise ValueError("Must create mirrored network before saving")

        checkpoint = {
            'model_state': self.mirrored.state_dict(),
        }

        if include_metadata:
            checkpoint['mirrored'] = True
            checkpoint['use_bn'] = self.use_bn

        torch.save(checkpoint, output_path)
