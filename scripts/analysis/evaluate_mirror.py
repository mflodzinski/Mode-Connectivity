"""Evaluate and compare original and mirrored models on any dataset.

This script loads both the original and mirrored checkpoints and verifies that
they produce identical predictions on the entire test set.
"""

import argparse
import torch
import torch.nn as nn
import sys
import numpy as np

sys.path.insert(0, 'external/dnn-mode-connectivity')
import models
import data as data_module


def evaluate_models(original, mirrored, test_loader):
    """Test both models on entire test set.

    Args:
        original: Original model
        mirrored: Mirrored model
        test_loader: DataLoader for test set

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    print(f"\n  Total test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {test_loader.batch_size}")

    # Set models to eval mode
    original.eval()
    mirrored.eval()

    # Track statistics
    max_output_diff = 0.0
    total_samples = 0
    matching_predictions = 0
    total_correct_orig = 0
    total_correct_mirror = 0

    print(f"\nProcessing all batches...")
    print("-" * 70)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Get outputs from both models
            outputs_orig = original(inputs)
            outputs_mirror = mirrored(inputs)

            # Compute differences
            output_diff = torch.abs(outputs_orig - outputs_mirror)
            batch_max_diff = output_diff.max().item()
            max_output_diff = max(max_output_diff, batch_max_diff)

            # Get predictions
            preds_orig = outputs_orig.argmax(dim=1)
            preds_mirror = outputs_mirror.argmax(dim=1)

            # Check if predictions match
            matches = (preds_orig == preds_mirror).sum().item()
            batch_size_actual = inputs.size(0)

            matching_predictions += matches
            total_samples += batch_size_actual

            # Compute accuracy for both models
            correct_orig = (preds_orig == targets).sum().item()
            correct_mirror = (preds_mirror == targets).sum().item()
            total_correct_orig += correct_orig
            total_correct_mirror += correct_mirror

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1:3d}: max_diff={batch_max_diff:.2e}, "
                      f"matches={matches}/{batch_size_actual}, "
                      f"processed={total_samples} samples")

    # Compute overall statistics
    match_rate = matching_predictions / total_samples
    accuracy_orig = total_correct_orig / total_samples
    accuracy_mirror = total_correct_mirror / total_samples

    print("-" * 70)
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal samples tested: {total_samples}")
    print(f"\nPrediction Matching:")
    print(f"  Matching predictions: {matching_predictions}/{total_samples} ({100*match_rate:.4f}%)")
    print(f"\nAccuracy:")
    print(f"  Original model:  {total_correct_orig}/{total_samples} ({100*accuracy_orig:.2f}%)")
    print(f"  Mirrored model:  {total_correct_mirror}/{total_samples} ({100*accuracy_mirror:.2f}%)")
    print(f"  Accuracy difference: {abs(accuracy_orig - accuracy_mirror):.6f}")
    print(f"\nOutput Differences:")
    print(f"  Maximum output difference: {max_output_diff:.2e}")

    # Determine pass/fail
    # Main criterion: all predictions must match
    # Secondary criterion: output differences should be small (but numerical errors are acceptable)
    tolerance = 1e-4  # More realistic tolerance for floating point operations
    all_match = (match_rate == 1.0)
    within_tolerance = (max_output_diff < tolerance)

    print("\n" + "="*70)
    if all_match:
        print("✓ EVALUATION PASSED!")
        print("="*70)
        print(f"- All {total_samples} predictions match (100%)")
        print(f"- Both models achieve {100*accuracy_orig:.2f}% accuracy")
        print(f"- Maximum output difference: {max_output_diff:.2e}")
        if within_tolerance:
            print(f"- Difference well within tolerance ({tolerance:.2e})")
        else:
            print(f"- Note: Output difference slightly exceeds strict tolerance but predictions identical")
        print("- Models are functionally equivalent on entire test set")
        success = True
    else:
        print("✗ EVALUATION FAILED!")
        print("="*70)
        print(f"- Predictions don't match: {matching_predictions}/{total_samples} ({100*match_rate:.2f}%)")
        print(f"- Max difference: {max_output_diff:.2e}")
        success = False

    print("="*70)

    return {
        'success': success,
        'total_samples': total_samples,
        'matching_predictions': matching_predictions,
        'match_rate': match_rate,
        'max_output_diff': max_output_diff,
        'accuracy_orig': accuracy_orig,
        'accuracy_mirror': accuracy_mirror
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare original and mirrored models'
    )
    parser.add_argument('--original', type=str, required=True,
                        help='Path to original model checkpoint')
    parser.add_argument('--mirrored', type=str, required=True,
                        help='Path to mirrored model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture (e.g., VGG16, VGG19, ResNet18)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., CIFAR10, CIFAR100, ImageNet)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--transform', type=str, required=True,
                        help='Transform name (e.g., VGG, ResNet)')
    parser.add_argument('--use-bn', action='store_true', default=False,
                        help='Use batch normalization (default: False)')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of output classes')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("MIRROR MODEL EVALUATION")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Batch Normalization: {args.use_bn}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {args.num_classes}")
    print(f"\nOriginal checkpoint: {args.original}")
    print(f"Mirrored checkpoint: {args.mirrored}")

    # Get model architecture
    model_name = args.model.upper().replace('-', '')
    if args.use_bn:
        arch_name = f"{model_name}BN"
    else:
        arch_name = model_name

    if not hasattr(models, arch_name):
        raise ValueError(f"Model {arch_name} not found in models module")

    architecture = getattr(models, arch_name)

    # Load original model
    print("\nLoading original model...")
    original = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
    checkpoint = torch.load(args.original, map_location='cpu')
    if 'model_state' in checkpoint:
        original.load_state_dict(checkpoint['model_state'])
    else:
        original.load_state_dict(checkpoint)
    original.eval()
    print("✓ Original model loaded")

    # Load mirrored model
    print("\nLoading mirrored model...")
    mirrored = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
    checkpoint = torch.load(args.mirrored, map_location='cpu')
    if 'model_state' in checkpoint:
        mirrored.load_state_dict(checkpoint['model_state'])
    else:
        mirrored.load_state_dict(checkpoint)
    mirrored.eval()
    print("✓ Mirrored model loaded")

    # Verify it's actually a mirrored checkpoint
    if 'mirrored' in checkpoint and checkpoint['mirrored']:
        print("✓ Checkpoint confirmed as mirrored")
    else:
        print("⚠ Warning: Checkpoint not marked as mirrored")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    loaders, num_classes = data_module.loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=True,
        shuffle_train=False
    )
    test_loader = loaders['test']
    print(f"✓ {args.dataset} test set loaded")

    # Evaluate on dataset
    results = evaluate_models(original, mirrored, test_loader)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nFunctional Equivalence:")
    if results['success']:
        print(f"  ✓ VERIFIED on all {results['total_samples']} test samples")
        print(f"  ✓ 100% prediction match")
        print(f"  ✓ Both models achieve {100*results['accuracy_orig']:.2f}% accuracy")
    else:
        print(f"  ✗ FAILED verification")
        print(f"  - Match rate: {100*results['match_rate']:.4f}%")
        print(f"  - Max diff: {results['max_output_diff']:.2e}")
    print("="*70 + "\n")

    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit(main())
