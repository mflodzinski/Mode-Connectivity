"""Analyze prediction changes along the curve.

Identifies samples that change predictions and computes statistics.
"""
import argparse
import numpy as np
import os
from pathlib import Path


def analyze_prediction_changes(predictions, targets, features_t0, features_t1, images, ts):
    """Analyze which samples change predictions along the curve.

    Args:
        predictions: [num_points, num_samples] - predictions at each t
        targets: [num_samples] - ground truth labels
        features_t0: [num_samples, feature_dim] - features at t=0
        features_t1: [num_samples, feature_dim] - features at t=1
        images: [num_samples, 3, 32, 32] - CIFAR-10 images
        ts: [num_points] - t values

    Returns:
        Dictionary with analysis results
    """
    num_points, num_samples = predictions.shape

    print(f"\nAnalyzing predictions for {num_samples} samples across {num_points} points...")

    # Use endpoint 0 features (simplest and safest - no averaging artifacts)
    features_for_visualization = features_t0
    print(f"Using endpoint 0 features for visualization: {features_for_visualization.shape}")

    # Find samples that change predictions
    # A sample changes if its prediction differs at any two points
    initial_pred = predictions[0]  # Predictions at t=0
    changes_mask = np.zeros(num_samples, dtype=bool)

    for i in range(num_samples):
        # Check if prediction ever differs from initial
        if not np.all(predictions[:, i] == initial_pred[i]):
            changes_mask[i] = True

    changing_indices = np.where(changes_mask)[0]
    stable_indices = np.where(~changes_mask)[0]

    num_changing = len(changing_indices)
    num_stable = len(stable_indices)

    print(f"\nPrediction stability:")
    print(f"  Samples that change: {num_changing} ({num_changing/num_samples*100:.2f}%)")
    print(f"  Samples that stay stable: {num_stable} ({num_stable/num_samples*100:.2f}%)")

    # Count prediction changes for each sample
    change_counts = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        pred_trajectory = predictions[:, i]
        # Count transitions
        changes = np.sum(pred_trajectory[:-1] != pred_trajectory[1:])
        change_counts[i] = changes

    # Statistics on changing samples
    if num_changing > 0:
        changing_counts = change_counts[changing_indices]
        print(f"\nChange count statistics (for changing samples):")
        print(f"  Min changes: {changing_counts.min()}")
        print(f"  Max changes: {changing_counts.max()}")
        print(f"  Mean changes: {changing_counts.mean():.2f}")
        print(f"  Median changes: {np.median(changing_counts):.1f}")

        # Find most unstable samples
        most_unstable_idx = changing_indices[np.argmax(changing_counts)]
        print(f"\nMost unstable sample:")
        print(f"  Index: {most_unstable_idx}")
        print(f"  Number of changes: {change_counts[most_unstable_idx]}")
        print(f"  True label: {targets[most_unstable_idx]}")
        print(f"  Prediction trajectory: {predictions[:, most_unstable_idx]}")

    # Per-class analysis
    print(f"\nPer-class stability:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for class_idx in range(10):
        class_mask = targets == class_idx
        class_samples = np.sum(class_mask)
        class_changing = np.sum(changes_mask & class_mask)

        if class_samples > 0:
            pct_changing = class_changing / class_samples * 100
            print(f"  {class_names[class_idx]:<12}: {class_changing:4d}/{class_samples:4d} "
                  f"({pct_changing:5.2f}%) change predictions")

    # Correctness analysis
    initial_correct = predictions[0] == targets
    final_correct = predictions[-1] == targets

    both_correct = np.sum(initial_correct & final_correct)
    both_wrong = np.sum(~initial_correct & ~final_correct)
    correct_to_wrong = np.sum(initial_correct & ~final_correct)
    wrong_to_correct = np.sum(~initial_correct & final_correct)

    print(f"\nEndpoint correctness:")
    print(f"  Both endpoints correct: {both_correct} ({both_correct/num_samples*100:.2f}%)")
    print(f"  Both endpoints wrong: {both_wrong} ({both_wrong/num_samples*100:.2f}%)")
    print(f"  Correct→Wrong: {correct_to_wrong} ({correct_to_wrong/num_samples*100:.2f}%)")
    print(f"  Wrong→Correct: {wrong_to_correct} ({wrong_to_correct/num_samples*100:.2f}%)")

    return {
        'num_samples': num_samples,
        'num_changing': num_changing,
        'num_stable': num_stable,
        'changing_indices': changing_indices,
        'stable_indices': stable_indices,
        'change_counts': change_counts,
        'predictions_changing': predictions[:, changing_indices],
        'targets_changing': targets[changing_indices],
        'features_for_visualization': features_for_visualization,
        'features_changing': features_for_visualization[changing_indices],
        'features_t0_changing': features_t0[changing_indices],
        'features_t1_changing': features_t1[changing_indices],
        'images_changing': images[changing_indices],
        'changing_counts': change_counts[changing_indices],
        'initial_correct': initial_correct,
        'final_correct': final_correct,
    }


def save_analysis(results, output_dir, class_names=None):
    """Save analysis results to files."""
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # Save NPZ with detailed data
    npz_path = output_dir / 'changing_samples_info.npz'
    np.savez_compressed(
        npz_path,
        indices=results['changing_indices'],
        change_counts=results['change_counts'],
        changing_counts=results['changing_counts'],
        predictions_changing=results['predictions_changing'],
        targets_changing=results['targets_changing'],
        features_for_visualization=results['features_for_visualization'],
        features_changing=results['features_changing'],
        features_t0_changing=results['features_t0_changing'],
        features_t1_changing=results['features_t1_changing'],
        images_changing=results['images_changing'],
    )
    print(f"\nSaved detailed data to: {npz_path}")

    # Save text summary
    txt_path = output_dir / 'analysis_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PREDICTION CHANGE ANALYSIS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total samples: {results['num_samples']}\n")
        f.write(f"Samples that change: {results['num_changing']} "
                f"({results['num_changing']/results['num_samples']*100:.2f}%)\n")
        f.write(f"Samples that stay stable: {results['num_stable']} "
                f"({results['num_stable']/results['num_samples']*100:.2f}%)\n\n")

        if results['num_changing'] > 0:
            changing_counts = results['changing_counts']
            f.write("Change count statistics (for changing samples):\n")
            f.write(f"  Min changes: {changing_counts.min()}\n")
            f.write(f"  Max changes: {changing_counts.max()}\n")
            f.write(f"  Mean changes: {changing_counts.mean():.2f}\n")
            f.write(f"  Median changes: {np.median(changing_counts):.1f}\n\n")

        f.write("Per-class stability:\n")
        for class_idx in range(10):
            class_mask = results['change_counts'] > 0
            targets = results['targets_changing']
            # Need to reconstruct per-class from changing samples
            # This is approximate - we'd need original targets for exact numbers
            f.write(f"  Class {class_idx} ({class_names[class_idx]})\n")

        f.write("\nEndpoint correctness:\n")
        both_correct = np.sum(results['initial_correct'] & results['final_correct'])
        both_wrong = np.sum(~results['initial_correct'] & ~results['final_correct'])
        correct_to_wrong = np.sum(results['initial_correct'] & ~results['final_correct'])
        wrong_to_correct = np.sum(~results['initial_correct'] & results['final_correct'])

        f.write(f"  Both endpoints correct: {both_correct} "
                f"({both_correct/results['num_samples']*100:.2f}%)\n")
        f.write(f"  Both endpoints wrong: {both_wrong} "
                f"({both_wrong/results['num_samples']*100:.2f}%)\n")
        f.write(f"  Correct→Wrong: {correct_to_wrong} "
                f"({correct_to_wrong/results['num_samples']*100:.2f}%)\n")
        f.write(f"  Wrong→Correct: {wrong_to_correct} "
                f"({wrong_to_correct/results['num_samples']*100:.2f}%)\n")

        f.write("="*70 + "\n")

    print(f"Saved summary to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze prediction changes along curve')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions_detailed.npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions data
    print(f"Loading predictions from {args.predictions}...")
    data = np.load(args.predictions)

    predictions = data['predictions']
    targets = data['targets']
    features_t0 = data['features_t0']
    features_t1 = data['features_t1']
    images = data['images']
    ts = data['ts']

    print(f"Loaded data:")
    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  features_t0: {features_t0.shape}")
    print(f"  features_t1: {features_t1.shape}")
    print(f"  images: {images.shape}")
    print(f"  ts: {ts.shape}")

    # Analyze changes
    results = analyze_prediction_changes(
        predictions, targets, features_t0, features_t1, images, ts
    )

    # Save results
    save_analysis(results, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
