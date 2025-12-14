"""Transform neural networks via mirroring or neuron swapping.

This consolidated script replaces:
- mirror_network.py
- swap_neurons.py

Modes:
  mirror: Create mirrored network via reverse neuron permutation
  swap:   Swap specific neurons to create minimally perturbed network
"""

import argparse
import torch

import sys
sys.path.insert(0, '../lib')

from lib.core import models, checkpoint, data
from lib.transform import mirror, neuron_swap
from lib.evaluation import evaluate as evaluation
from lib.utils.args import ArgumentParserBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Transform neural networks via mirroring or neuron swapping'
    )

    # Custom arguments specific to this script
    parser.add_argument('--mode', required=True, choices=['mirror', 'swap'],
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
                       help='[mirror] Verify on full test dataset instead of random samples')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='[mirror] Batch size for full dataset verification (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='[mirror] Number of data loading workers (default: 4)')

    # Swap-specific arguments
    parser.add_argument('--layer-depth', type=str, choices=['early', 'mid', 'late'],
                       default='mid',
                       help='[swap] Layer depth to swap (default: mid)')
    parser.add_argument('--num-swaps', type=int, default=1,
                       help='[swap] Number of neuron pairs to swap (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='[swap] Random seed for neuron selection (default: 42)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       help='[swap] Dataset for verification (default: CIFAR10)')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='[swap] Path to dataset (default: ./data)')
    parser.add_argument('--quick-verify', action='store_true',
                       help='[swap] Quick verification with random inputs only')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"NEURAL NETWORK TRANSFORMATION: {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"\nInput:  {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")

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

                tolerance = 1e-5
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

    elif args.mode == 'swap':
        # ============================================================
        # SWAP MODE
        # ============================================================
        print(f"\n{'='*70}")
        print("SWAPPING NEURONS")
        print(f"{'='*70}")

        # Create swapper using NeuronSwapper class
        swapper = neuron_swap.NeuronSwapper(model, architecture=args.model)

        # Perform swap
        swapped_model = swapper.swap_by_depth(
            depth=args.layer_depth,
            num_swaps=args.num_swaps,
            seed=args.seed,
            verbose=True
        )

        # Calculate L2 distance
        print(f"\n{'='*70}")
        print("DISTANCE ANALYSIS")
        print(f"{'='*70}")
        distance_stats = swapper.calculate_l2_distance(verbose=True)

        # Verify equivalence if requested
        if args.verify:
            print(f"\n{'='*70}")
            print("VERIFICATION")
            print(f"{'='*70}")

            if args.quick_verify:
                # Quick verification with random inputs
                print("\nPerforming quick verification with random inputs...")

                model.eval()
                swapped_model.eval()

                max_diff = 0.0
                all_match = True

                with torch.no_grad():
                    for i in range(args.num_samples):
                        x = torch.randn(1, *args.input_size)
                        out_orig = model(x)
                        out_swap = swapped_model(x)

                        diff = torch.abs(out_orig - out_swap).max().item()
                        max_diff = max(max_diff, diff)

                        pred_orig = out_orig.argmax(dim=1).item()
                        pred_swap = out_swap.argmax(dim=1).item()

                        match = "✓" if pred_orig == pred_swap else "✗"
                        print(f"Sample {i+1}: max_diff={diff:.2e}, "
                              f"pred_orig={pred_orig}, pred_swap={pred_swap} {match}")

                        if pred_orig != pred_swap:
                            all_match = False

                tolerance = 1e-5
                is_equivalent = all_match and max_diff < tolerance

                if is_equivalent:
                    print(f"\n✓ Quick verification passed (max diff: {max_diff:.2e})")
                else:
                    print(f"\n✗ Quick verification failed (max diff: {max_diff:.2e})")
            else:
                # Full dataset verification
                equiv_stats = swapper.verify_equivalence_on_dataset(
                    dataset_name=args.dataset,
                    data_path=args.data_path,
                    verbose=True
                )

        # Save swapped model
        swapper.save(args.output, include_original=False)

    print(f"\n{'='*70}")
    print(f"✓ TRANSFORMATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
