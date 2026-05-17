"""Transform neural networks via mirroring or random permutation.

This consolidated script replaces:
- mirror_network.py

Modes:
  mirror: Create mirrored network via reverse neuron permutation
  random: Create functionally equivalent random full-network permutation (VGG16)
"""

import argparse
import os
import torch

import sys
from mode_connectivity.core import models, checkpoint, data
from mode_connectivity.transform import mirror, random_permutation
from mode_connectivity.utils.args import ArgumentParserBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Transform neural networks via mirroring or random permutation'
    )

    # Custom arguments specific to this script
    parser.add_argument('--mode', required=True, choices=['mirror', 'random'],
                       help='Transformation mode')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save transformed checkpoint')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_checkpoint_args(parser, single=True)
    ArgumentParserBuilder.add_model_args(parser)

    # Verification arguments
    parser.add_argument('--verify', action='store_true',
                       help='Verify functional equivalence')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of random samples for verification (default: 10)')
    parser.add_argument('--input-size', type=int, nargs=3, default=[3, 32, 32],
                       help='Input size for verification (C H W) (default: 3 32 32)')
    parser.add_argument('--full-dataset-verify', action='store_true',
                       help='[mirror/random] Verify on full test dataset instead of random samples')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='[mirror/random] Batch size for full dataset verification (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='[mirror/random] Number of data loading workers (default: 4)')
    parser.add_argument('--perm-seed', type=int, default=42,
                       help='[random] Random seed for full-network permutation (default: 42)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       help='Dataset for verification/evaluation (default: CIFAR10)')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to dataset (default: ./data)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"NEURAL NETWORK TRANSFORMATION: {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"\nInput:  {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model...")
    architecture = models.get_architecture(args.model, args.use_bn)
    model = checkpoint.load_model(
        args.checkpoint,
        architecture,
        num_classes=args.num_classes
    )
    print(f"✓ Model loaded")

    if args.mode == 'mirror':
        # ============================================================
        # MIRROR MODE
        # ============================================================
        print(f"\n{'='*70}")
        print("CREATING MIRRORED NETWORK")
        print(f"{'='*70}")

        # Create mirror using MirrorNetwork class
        mirror_net = mirror.MirrorNetwork(model, use_bn=args.use_bn)
        mirrored_model = mirror_net.create_mirror(verbose=True)

        # Verify equivalence if requested
        if args.verify:
            print(f"\n{'='*70}")
            print("VERIFICATION")
            print(f"{'='*70}")

            if args.full_dataset_verify:
                # Full dataset verification
                print("\nVerifying on full test dataset...")

                # Load dataset
                loaders, num_classes = data.get_loaders(
                    args.dataset,
                    data_path=args.data_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    transform_name='VGG',
                    use_test=True
                )

                model.eval()
                mirrored_model.eval()

                correct_original = 0
                correct_mirrored = 0
                total = 0
                max_output_diff = 0.0
                predictions_match = 0

                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(loaders['test']):
                        # Get predictions
                        out_orig = model(inputs)
                        out_mirror = mirrored_model(inputs)

                        # Check predictions
                        pred_orig = out_orig.argmax(dim=1)
                        pred_mirror = out_mirror.argmax(dim=1)

                        correct_original += pred_orig.eq(targets).sum().item()
                        correct_mirrored += pred_mirror.eq(targets).sum().item()
                        predictions_match += pred_orig.eq(pred_mirror).sum().item()
                        total += targets.size(0)

                        # Track maximum output difference
                        diff = torch.abs(out_orig - out_mirror).max().item()
                        max_output_diff = max(max_output_diff, diff)

                        if (batch_idx + 1) % 20 == 0:
                            print(f"  Processed {total} samples...")

                acc_original = 100.0 * correct_original / total
                acc_mirrored = 100.0 * correct_mirrored / total
                pred_match_rate = 100.0 * predictions_match / total

                print(f"\n{'='*70}")
                print("FULL DATASET VERIFICATION RESULTS")
                print(f"{'='*70}")
                print(f"Total samples:           {total}")
                print(f"Original accuracy:       {acc_original:.2f}%")
                print(f"Mirrored accuracy:       {acc_mirrored:.2f}%")
                print(f"Prediction match rate:   {pred_match_rate:.2f}%")
                print(f"Max output difference:   {max_output_diff:.2e}")

                tolerance = 5e-5
                is_equivalent = (max_output_diff < tolerance) and (pred_match_rate > 99.9)

                if is_equivalent:
                    print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
                else:
                    print(f"\n✗ WARNING: Models may not be perfectly equivalent")
                    print(f"  Max diff {max_output_diff:.2e}, Match rate {pred_match_rate:.2f}%")
                print("=" * 70)

            else:
                # Quick verification with random inputs
                is_equivalent = mirror_net.verify_equivalence(
                    num_samples=args.num_samples,
                    input_size=tuple(args.input_size),
                    verbose=True
                )

                # Count parameter differences
                mirror_net.count_parameter_differences(verbose=True)

                if not is_equivalent:
                    print("\n⚠ WARNING: Verification failed!")
                    print("The mirrored network may not be functionally equivalent.")

        # Save mirrored model
        mirror_net.save_mirrored(args.output, include_metadata=True)

    elif args.mode == 'random':
        # ============================================================
        # RANDOM FULL-NETWORK PERMUTATION MODE
        # ============================================================
        print(f"\n{'='*70}")
        print("CREATING RANDOM FULL-NETWORK PERMUTATION")
        print(f"{'='*70}")

        if args.model.upper() != 'VGG16':
            raise ValueError("Random full-network permutation currently supports only VGG16")
        if args.use_bn:
            raise ValueError("Random full-network permutation currently supports only non-BN VGG16")

        perm_gen = random_permutation.VGG16RandomPermutation()
        perm = perm_gen.generate(seed=args.perm_seed)
        permuted_state = perm_gen.apply_to_state_dict(model.state_dict(), perm)

        permuted_model = checkpoint.load_model(
            args.checkpoint,
            architecture,
            num_classes=args.num_classes
        )
        permuted_model.load_state_dict(permuted_state)

        print(f"✓ Applied random full-network permutation (seed={args.perm_seed})")

        # Verify equivalence if requested
        if args.verify:
            print(f"\n{'='*70}")
            print("VERIFICATION")
            print(f"{'='*70}")

            if args.full_dataset_verify:
                # Full dataset verification
                print("\nVerifying on full test dataset...")

                loaders, _ = data.get_loaders(
                    args.dataset,
                    data_path=args.data_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    transform_name='VGG',
                    use_test=True
                )

                model.eval()
                permuted_model.eval()

                correct_original = 0
                correct_permuted = 0
                total = 0
                max_output_diff = 0.0
                predictions_match = 0

                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(loaders['test']):
                        out_orig = model(inputs)
                        out_perm = permuted_model(inputs)

                        pred_orig = out_orig.argmax(dim=1)
                        pred_perm = out_perm.argmax(dim=1)

                        correct_original += pred_orig.eq(targets).sum().item()
                        correct_permuted += pred_perm.eq(targets).sum().item()
                        predictions_match += pred_orig.eq(pred_perm).sum().item()
                        total += targets.size(0)

                        diff = torch.abs(out_orig - out_perm).max().item()
                        max_output_diff = max(max_output_diff, diff)

                        if (batch_idx + 1) % 20 == 0:
                            print(f"  Processed {total} samples...")

                acc_original = 100.0 * correct_original / total
                acc_permuted = 100.0 * correct_permuted / total
                pred_match_rate = 100.0 * predictions_match / total

                print(f"\n{'='*70}")
                print("FULL DATASET VERIFICATION RESULTS")
                print(f"{'='*70}")
                print(f"Total samples:           {total}")
                print(f"Original accuracy:       {acc_original:.2f}%")
                print(f"Permuted accuracy:       {acc_permuted:.2f}%")
                print(f"Prediction match rate:   {pred_match_rate:.2f}%")
                print(f"Max output difference:   {max_output_diff:.2e}")

                tolerance = 5e-5
                is_equivalent = (max_output_diff < tolerance) and (pred_match_rate > 99.9)

                if is_equivalent:
                    print("\n✓ VERIFICATION PASSED: Models are functionally equivalent!")
                else:
                    print(f"\n✗ WARNING: Models may not be perfectly equivalent")
                    print(f"  Max diff {max_output_diff:.2e}, Match rate {pred_match_rate:.2f}%")
                print("=" * 70)
            else:
                # Quick verification with random inputs
                print("\nVerifying on random inputs...")
                model.eval()
                permuted_model.eval()
                max_diff = 0.0

                with torch.no_grad():
                    for i in range(args.num_samples):
                        x = torch.randn(1, *args.input_size)
                        out_orig = model(x)
                        out_perm = permuted_model(x)

                        diff = torch.abs(out_orig - out_perm).max().item()
                        max_diff = max(max_diff, diff)
                        pred_orig = out_orig.argmax(dim=1).item()
                        pred_perm = out_perm.argmax(dim=1).item()
                        match = "✓" if pred_orig == pred_perm else "✗"
                        print(f"Sample {i+1}: max_diff={diff:.2e}, "
                              f"pred_orig={pred_orig}, pred_perm={pred_perm} {match}")

                if max_diff < 1e-5:
                    print(f"\n✓ Quick verification passed (max diff: {max_diff:.2e})")
                else:
                    print(f"\n✗ Quick verification failed (max diff: {max_diff:.2e})")

        # Save random-permuted model with weights-only-safe metadata
        # (PyTorch 2.6 defaults torch.load(..., weights_only=True)).
        permutation_serialized = {k: v.tolist() for k, v in perm.items()}
        torch.save({
            'model_state': permuted_model.state_dict(),
            'random_permuted': True,
            'perm_seed': args.perm_seed,
            'permutation': permutation_serialized,
        }, args.output)

    print(f"\n{'='*70}")
    print(f"✓ TRANSFORMATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
