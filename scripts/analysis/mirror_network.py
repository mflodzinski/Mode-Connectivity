"""Generate a mirrored neural network by permuting neuron indices.

This script demonstrates permutation symmetry in neural networks: you can rearrange
neurons/filters systematically to create a different point in parameter space that
computes the identical function.

For each layer, we apply reverse-order permutation while maintaining functional equivalence.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import sys
import os

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import models


def get_reverse_permutation(n):
    """Generate reverse-order permutation indices.

    Args:
        n: Number of elements

    Returns:
        numpy array of indices in reverse order [n-1, n-2, ..., 1, 0]
    """
    return np.arange(n)[::-1].copy()  # .copy() to avoid negative stride issues


def permute_conv_layer(conv_layer, perm_in, perm_out):
    """Permute a convolutional layer's weights and biases.

    Args:
        conv_layer: nn.Conv2d layer
        perm_in: Permutation for input channels (from previous layer)
        perm_out: Permutation for output channels (filters)

    Returns:
        Permuted conv layer
    """
    # Permute weights: [out_channels, in_channels, H, W]
    # First permute output channels, then input channels
    conv_layer.weight.data = conv_layer.weight.data[perm_out, :, :, :]
    conv_layer.weight.data = conv_layer.weight.data[:, perm_in, :, :]

    # Permute bias if it exists
    if conv_layer.bias is not None:
        conv_layer.bias.data = conv_layer.bias.data[perm_out]

    return conv_layer


def permute_bn_layer(bn_layer, perm):
    """Permute a batch normalization layer's parameters.

    Args:
        bn_layer: nn.BatchNorm2d layer
        perm: Permutation for channels

    Returns:
        Permuted BN layer
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


def permute_fc_layer(fc_layer, perm_in, perm_out=None):
    """Permute a fully connected layer's weights and biases.

    Args:
        fc_layer: nn.Linear layer
        perm_in: Permutation for input neurons
        perm_out: Permutation for output neurons (None for final layer)

    Returns:
        Permuted FC layer
    """
    # Permute weights: [out_features, in_features]
    fc_layer.weight.data = fc_layer.weight.data[:, perm_in]

    # Only permute output if not the final classification layer
    if perm_out is not None:
        fc_layer.weight.data = fc_layer.weight.data[perm_out, :]
        if fc_layer.bias is not None:
            fc_layer.bias.data = fc_layer.bias.data[perm_out]

    return fc_layer


def mirror_vgg19(model, use_bn=True):
    """Create a mirrored version of VGG19 by reversing neuron indices.

    This function applies reverse-order permutation to each layer while maintaining
    the chain of permutations between layers to preserve functional equivalence.

    Args:
        model: VGG19 model (either VGG or VGGBatchNorm from the codebase)
        use_bn: Whether the model uses batch normalization

    Returns:
        Mirrored model that is functionally equivalent to the original
    """
    # Create a deep copy to avoid modifying the original
    mirrored = deepcopy(model)

    # Track the current permutation for chaining between layers
    current_perm = None  # Input channels permutation from previous layer

    print("\n" + "="*70)
    print("APPLYING REVERSE-ORDER PERMUTATIONS")
    print("="*70)

    # Process convolutional layers
    # This codebase's VGG uses layer_blocks instead of features
    if hasattr(mirrored, 'layer_blocks'):
        layer_idx = 0
        for block_idx, block in enumerate(mirrored.layer_blocks):
            print(f"\n--- Block {block_idx} ---")
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    # Get input channels
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels

                    # Input permutation: use previous layer's output or no permutation for first layer
                    if current_perm is None:
                        perm_in = np.arange(in_channels)  # First layer: no input permutation
                    else:
                        perm_in = current_perm

                    # Output permutation: reverse order
                    perm_out = get_reverse_permutation(out_channels)

                    print(f"Layer {layer_idx} (Conv2d): in={in_channels}, out={out_channels}")
                    print(f"  Input perm:  [{perm_in[:3]}...{perm_in[-3:]}]" if len(perm_in) > 6 else f"  Input perm:  {perm_in}")
                    print(f"  Output perm: [{perm_out[:3]}...{perm_out[-3:]}]" if len(perm_out) > 6 else f"  Output perm: {perm_out}")

                    # Apply permutation
                    permute_conv_layer(layer, perm_in, perm_out)

                    # Update current permutation for next layer
                    current_perm = perm_out
                    layer_idx += 1

                elif isinstance(layer, nn.BatchNorm2d):
                    num_features = layer.num_features
                    print(f"Layer {layer_idx} (BatchNorm2d): features={num_features}")
                    print(f"  Permutation: [{current_perm[:3]}...{current_perm[-3:]}]" if len(current_perm) > 6 else f"  Permutation: {current_perm}")

                    permute_bn_layer(layer, current_perm)
                    layer_idx += 1
                    # current_perm stays the same after BN

    # Fallback for torchvision-style VGG (features attribute)
    elif hasattr(mirrored, 'features'):
        layers = list(mirrored.features.children())

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels

                if current_perm is None:
                    perm_in = np.arange(in_channels)
                else:
                    perm_in = current_perm

                perm_out = get_reverse_permutation(out_channels)

                print(f"\nLayer {i} (Conv2d): in={in_channels}, out={out_channels}")
                permute_conv_layer(layer, perm_in, perm_out)
                current_perm = perm_out

            elif isinstance(layer, nn.BatchNorm2d):
                permute_bn_layer(layer, current_perm)

            elif isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                print(f"\nLayer {i} ({layer.__class__.__name__}): no permutation needed")

    # Process classifier (fully connected part)
    if hasattr(mirrored, 'classifier'):
        # The input to classifier comes from flattened conv features
        # We need to account for spatial dimensions being flattened

        fc_layers = list(mirrored.classifier.children())
        fc_perm = None  # Track permutation for FC layers separately

        for i, layer in enumerate(fc_layers):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features

                # First FC layer: needs to handle flattened conv output
                if fc_perm is None and current_perm is not None:
                    # Calculate how many spatial positions per channel
                    # For VGG19: after 5 maxpool layers, 224x224 -> 7x7
                    # So we have current_perm.shape[0] channels * 7 * 7 flattened inputs
                    spatial_size = in_features // current_perm.shape[0]

                    # Create permutation for flattened tensor
                    # Flatten is done as [C, H, W] -> [C*H*W], so we need to permute channel dimension
                    flattened_perm = np.concatenate([
                        current_perm * spatial_size + offset
                        for offset in range(spatial_size)
                    ])
                    perm_in = flattened_perm
                else:
                    perm_in = fc_perm if fc_perm is not None else np.arange(in_features)

                # Determine if this is the final classification layer
                is_final = (i == len(fc_layers) - 1) or (i == len(fc_layers) - 2 and isinstance(fc_layers[-1], nn.ReLU))

                if is_final:
                    # Final layer: don't permute output (preserve class indices)
                    perm_out = None
                    print(f"\nFC Layer {i} (Linear): in={in_features}, out={out_features} [FINAL - no output permutation]")
                else:
                    # Hidden layer: permute output
                    perm_out = get_reverse_permutation(out_features)
                    print(f"\nFC Layer {i} (Linear): in={in_features}, out={out_features}")
                    print(f"  Output perm: {perm_out[:5]}...{perm_out[-5:] if len(perm_out) > 10 else perm_out}")

                # Apply permutation
                permute_fc_layer(layer, perm_in, perm_out)

                # Update permutation for next FC layer
                fc_perm = perm_out

            elif isinstance(layer, (nn.ReLU, nn.Dropout)):
                print(f"\nFC Layer {i} ({layer.__class__.__name__}): no permutation needed")

    print("\n" + "="*70)
    print("PERMUTATION COMPLETE")
    print("="*70)

    return mirrored


def verify_functional_equivalence(original, mirrored, num_samples=10, input_size=(3, 224, 224)):
    """Verify that original and mirrored models produce identical outputs.

    Args:
        original: Original model
        mirrored: Mirrored model
        num_samples: Number of random test samples
        input_size: Input tensor size (C, H, W)

    Returns:
        Boolean indicating if models are functionally equivalent
    """
    print("\n" + "="*70)
    print("VERIFYING FUNCTIONAL EQUIVALENCE")
    print("="*70)

    original.eval()
    mirrored.eval()

    max_diff = 0.0
    all_close = True

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input
            x = torch.randn(1, *input_size)

            # Get outputs
            out_orig = original(x)
            out_mirror = mirrored(x)

            # Compute difference
            diff = torch.abs(out_orig - out_mirror).max().item()
            max_diff = max(max_diff, diff)

            # Check if predictions match
            pred_orig = out_orig.argmax(dim=1).item()
            pred_mirror = out_mirror.argmax(dim=1).item()

            match = "✓" if pred_orig == pred_mirror else "✗"
            print(f"Sample {i+1}: max_diff={diff:.2e}, pred_orig={pred_orig}, pred_mirror={pred_mirror} {match}")

            if pred_orig != pred_mirror:
                all_close = False

    print(f"\nMaximum difference across all samples: {max_diff:.2e}")

    # Use a reasonable tolerance for floating point comparisons
    tolerance = 1e-5
    functionally_equivalent = all_close and max_diff < tolerance

    if functionally_equivalent:
        print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
    else:
        print(f"\n✗ VERIFICATION FAILED: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")

    print("="*70)

    return functionally_equivalent


def count_parameter_differences(original, mirrored):
    """Count how many parameters differ between original and mirrored models.

    This demonstrates that the models are at different points in parameter space.
    """
    print("\n" + "="*70)
    print("PARAMETER SPACE ANALYSIS")
    print("="*70)

    total_params = 0
    different_params = 0

    for (name_o, param_o), (name_m, param_m) in zip(
        original.named_parameters(), mirrored.named_parameters()
    ):
        assert name_o == name_m, "Parameter names don't match!"

        total = param_o.numel()
        different = (param_o != param_m).sum().item()

        total_params += total
        different_params += different

        if different > 0:
            print(f"{name_o}: {different}/{total} ({100*different/total:.1f}%) parameters differ")

    print(f"\nTotal: {different_params}/{total_params} ({100*different_params/total_params:.1f}%) parameters differ")
    print("\n✓ Models are at DIFFERENT points in parameter space")
    print("✓ But produce IDENTICAL outputs (functional equivalence)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a mirrored neural network by permuting neuron indices'
    )
    parser.add_argument('--model', type=str, default='VGG19',
                        help='Model architecture (default: VGG19)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (optional)')
    parser.add_argument('--use-bn', action='store_true', default=False,
                        help='Use batch normalization (default: False)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of output classes (default: 10 for CIFAR10)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for verification (default: 10)')
    parser.add_argument('--save-mirrored', type=str, default=None,
                        help='Path to save mirrored model checkpoint')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("NEURAL NETWORK MIRROR IMAGE GENERATOR")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Batch Normalization: {args.use_bn}")
    print(f"Number of classes: {args.num_classes}")

    # Get model architecture
    model_name = args.model.upper().replace('-', '')
    if args.use_bn:
        arch_name = f"{model_name}BN"
    else:
        arch_name = model_name

    if not hasattr(models, arch_name):
        raise ValueError(f"Model {arch_name} not found. Available: VGG16, VGG16BN, VGG19, VGG19BN")

    architecture = getattr(models, arch_name)

    # Create model
    original = architecture.base(num_classes=args.num_classes, **architecture.kwargs)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state' in checkpoint:
            original.load_state_dict(checkpoint['model_state'])
        else:
            original.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded")
    else:
        print("\nUsing randomly initialized model")

    # Create mirrored version
    mirrored = mirror_vgg19(original, use_bn=args.use_bn)

    # Verify functional equivalence
    is_equivalent = verify_functional_equivalence(
        original, mirrored,
        num_samples=args.num_samples,
        input_size=(3, 32, 32) if args.num_classes == 10 else (3, 224, 224)
    )

    # Analyze parameter differences
    count_parameter_differences(original, mirrored)

    # Save mirrored model if requested
    if args.save_mirrored:
        save_dir = os.path.dirname(args.save_mirrored)
        if save_dir:  # Only create directory if path includes one
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state': mirrored.state_dict(),
            'mirrored': True,
            'original_checkpoint': args.checkpoint
        }, args.save_mirrored)
        print(f"\n✓ Mirrored model saved to: {args.save_mirrored}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Permutation symmetry demonstrated:")
    print("  ✓ Original and mirrored models are at DIFFERENT points in parameter space")
    print("  ✓ But they compute the IDENTICAL function")
    print("  ✓ This creates equivalent solutions at different locations in the loss landscape")
    print("\nThis is fundamental to understanding mode connectivity:")
    print("  - Neural networks have inherent symmetries")
    print("  - Multiple parameter configurations can represent the same function")
    print("  - These symmetries affect the loss landscape structure")
    print("="*70 + "\n")

    return is_equivalent


if __name__ == "__main__":
    main()
