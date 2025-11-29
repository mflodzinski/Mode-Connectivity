"""Compare per-class accuracy between seed0 and seed1 endpoints.

Analyzes how the two independently trained models differ in their
per-class performance and identifies which classes show the most divergence.
"""
import argparse
import numpy as np
from pathlib import Path


def compute_per_class_metrics(predictions, targets):
    """Compute per-class accuracy and confusion patterns.

    Args:
        predictions: [num_samples] - predicted classes
        targets: [num_samples] - true labels

    Returns:
        Dictionary with per-class metrics
    """
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    results = {
        'class_names': class_names,
        'accuracy': np.zeros(num_classes),
        'correct': np.zeros(num_classes, dtype=int),
        'total': np.zeros(num_classes, dtype=int),
    }

    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        n_class = np.sum(class_mask)
        results['total'][class_idx] = n_class

        if n_class > 0:
            class_preds = predictions[class_mask]
            class_targets = targets[class_mask]
            n_correct = np.sum(class_preds == class_targets)
            results['correct'][class_idx] = n_correct
            results['accuracy'][class_idx] = n_correct / n_class * 100

    return results


def compare_endpoints(seed0_results, seed1_results):
    """Compare two endpoint results and compute differences.

    Args:
        seed0_results: metrics from seed0
        seed1_results: metrics from seed1

    Returns:
        Dictionary with comparison metrics
    """
    diff = seed1_results['accuracy'] - seed0_results['accuracy']
    abs_diff = np.abs(diff)

    # Find agreement/disagreement patterns
    seed0_preds = None  # Will be set from predictions
    seed1_preds = None

    comparison = {
        'accuracy_diff': diff,
        'abs_diff': abs_diff,
        'seed0_better': diff < 0,
        'seed1_better': diff > 0,
        'equal': diff == 0,
    }

    return comparison


def save_comparison_report(seed0_results, seed1_results, comparison, output_path, targets,
                           seed0_preds, seed1_preds):
    """Save detailed comparison report to text file."""

    class_names = seed0_results['class_names']

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ENDPOINT ACCURACY COMPARISON: SEED0 vs SEED1\n")
        f.write("=" * 80 + "\n\n")

        # Overall accuracy
        overall_acc_0 = np.sum(seed0_results['correct']) / np.sum(seed0_results['total']) * 100
        overall_acc_1 = np.sum(seed1_results['correct']) / np.sum(seed1_results['total']) * 100

        f.write("OVERALL ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Seed0: {overall_acc_0:.2f}%\n")
        f.write(f"Seed1: {overall_acc_1:.2f}%\n")
        f.write(f"Difference (Seed1 - Seed0): {overall_acc_1 - overall_acc_0:+.2f}%\n")
        f.write("\n")

        # Per-class comparison
        f.write("PER-CLASS ACCURACY COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<12} {'Seed0':>8} {'Seed1':>8} {'Diff':>8} {'Better':>10} {'n':<6}\n")
        f.write("-" * 80 + "\n")

        for i in range(10):
            acc0 = seed0_results['accuracy'][i]
            acc1 = seed1_results['accuracy'][i]
            diff = comparison['accuracy_diff'][i]
            n = seed0_results['total'][i]

            if diff > 0:
                better = "Seed1"
            elif diff < 0:
                better = "Seed0"
            else:
                better = "Equal"

            f.write(f"{class_names[i]:<12} {acc0:7.2f}% {acc1:7.2f}% {diff:+7.2f}% {better:>10} {n:<6}\n")

        f.write("\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean absolute difference: {comparison['abs_diff'].mean():.2f}%\n")
        f.write(f"Max absolute difference: {comparison['abs_diff'].max():.2f}%\n")
        f.write(f"Std of differences: {comparison['accuracy_diff'].std():.2f}%\n")
        f.write(f"\nClasses where Seed0 better: {np.sum(comparison['seed0_better'])}\n")
        f.write(f"Classes where Seed1 better: {np.sum(comparison['seed1_better'])}\n")
        f.write(f"Classes with equal accuracy: {np.sum(comparison['equal'])}\n")
        f.write("\n")

        # Classes with largest differences
        sorted_idx = np.argsort(comparison['abs_diff'])[::-1]

        f.write("CLASSES WITH LARGEST DIFFERENCES\n")
        f.write("-" * 80 + "\n")
        for rank, idx in enumerate(sorted_idx[:5], 1):
            diff = comparison['accuracy_diff'][idx]
            better = "Seed1" if diff > 0 else "Seed0"
            f.write(f"{rank}. {class_names[idx]:<12} {abs(diff):6.2f}% difference ({better} better)\n")
        f.write("\n")

        # Agreement analysis
        f.write("PREDICTION AGREEMENT ANALYSIS\n")
        f.write("-" * 80 + "\n")

        # Overall agreement
        agreement = (seed0_preds == seed1_preds)
        agreement_rate = np.mean(agreement) * 100
        f.write(f"Overall agreement: {agreement_rate:.2f}% ({np.sum(agreement)}/{len(targets)} samples)\n")
        f.write(f"Overall disagreement: {100-agreement_rate:.2f}% ({np.sum(~agreement)}/{len(targets)} samples)\n")
        f.write("\n")

        # Per-class agreement
        f.write("PER-CLASS AGREEMENT\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<12} {'Agree':>10} {'Disagree':>10} {'Rate':>12} {'Pattern'}\n")
        f.write("-" * 80 + "\n")

        for class_idx in range(10):
            class_mask = targets == class_idx
            class_agreement = agreement[class_mask]
            n_agree = np.sum(class_agreement)
            n_disagree = np.sum(~class_agreement)
            rate = np.mean(class_agreement) * 100

            # Find most common disagreement pattern
            if n_disagree > 0:
                disagree_mask = class_mask & (~agreement)
                seed0_wrong = seed0_preds[disagree_mask]
                seed1_wrong = seed1_preds[disagree_mask]

                # Most common predictions when they disagree
                from collections import Counter
                pattern = f"Most common: {Counter(seed0_wrong).most_common(1)[0][0]} vs {Counter(seed1_wrong).most_common(1)[0][0]}"
            else:
                pattern = "Perfect agreement"

            f.write(f"{class_names[class_idx]:<12} {n_agree:>10} {n_disagree:>10} {rate:>11.1f}% {pattern}\n")

        f.write("\n")

        # Both correct vs both wrong vs one correct
        both_correct = (seed0_preds == targets) & (seed1_preds == targets)
        both_wrong = (seed0_preds != targets) & (seed1_preds != targets)
        seed0_only_correct = (seed0_preds == targets) & (seed1_preds != targets)
        seed1_only_correct = (seed0_preds != targets) & (seed1_preds == targets)

        f.write("CORRECTNESS PATTERNS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Both correct:       {np.sum(both_correct):5} ({np.mean(both_correct)*100:5.2f}%)\n")
        f.write(f"Both wrong:         {np.sum(both_wrong):5} ({np.mean(both_wrong)*100:5.2f}%)\n")
        f.write(f"Seed0 only correct: {np.sum(seed0_only_correct):5} ({np.mean(seed0_only_correct)*100:5.2f}%)\n")
        f.write(f"Seed1 only correct: {np.sum(seed1_only_correct):5} ({np.mean(seed1_only_correct)*100:5.2f}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare endpoint accuracies')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions_detailed.npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for comparison report')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    data = np.load(args.predictions)

    predictions = data['predictions']  # [num_points, num_samples]
    targets = data['targets']
    ts = data['ts']

    # Extract endpoint predictions (t=0 and t=1)
    seed0_preds = predictions[0]   # t=0
    seed1_preds = predictions[-1]  # t=1

    print(f"\nLoaded data:")
    print(f"  Number of samples: {len(targets)}")
    print(f"  Number of t points: {len(ts)}")

    # Compute per-class metrics
    print("\nComputing per-class metrics...")
    seed0_results = compute_per_class_metrics(seed0_preds, targets)
    seed1_results = compute_per_class_metrics(seed1_preds, targets)

    # Compare endpoints
    print("Comparing endpoints...")
    comparison = compare_endpoints(seed0_results, seed1_results)

    # Save report
    output_path = output_dir / 'endpoint_comparison.txt'
    print(f"\nSaving comparison report to {output_path}...")
    save_comparison_report(seed0_results, seed1_results, comparison, output_path,
                          targets, seed0_preds, seed1_preds)

    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    overall_acc_0 = np.sum(seed0_results['correct']) / np.sum(seed0_results['total']) * 100
    overall_acc_1 = np.sum(seed1_results['correct']) / np.sum(seed1_results['total']) * 100

    print(f"\nOverall accuracy:")
    print(f"  Seed0: {overall_acc_0:.2f}%")
    print(f"  Seed1: {overall_acc_1:.2f}%")
    print(f"  Difference: {overall_acc_1 - overall_acc_0:+.2f}%")

    print(f"\nMean absolute per-class difference: {comparison['abs_diff'].mean():.2f}%")
    print(f"Max absolute per-class difference: {comparison['abs_diff'].max():.2f}%")

    agreement = (seed0_preds == seed1_preds)
    agreement_rate = np.mean(agreement) * 100
    print(f"\nPrediction agreement: {agreement_rate:.2f}%")

    print(f"\nFull report saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
