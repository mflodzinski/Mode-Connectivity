"""Plot per-class accuracy along the Bezier curve.

Creates a multi-line plot showing how accuracy changes for each CIFAR-10 class
as we move from t=0 to t=1 along the curve.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_per_class_accuracy(predictions, targets, ts):
    """Compute per-class accuracy at each t value.

    Args:
        predictions: [num_points, num_samples] - predictions at each t
        targets: [num_samples] - true labels
        ts: [num_points] - t values

    Returns:
        accuracies: [num_classes, num_points] - accuracy for each class at each t
        class_counts: [num_classes] - number of samples per class
    """
    num_points = len(ts)
    num_classes = 10

    accuracies = np.zeros((num_classes, num_points))
    class_counts = np.zeros(num_classes, dtype=int)

    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        class_counts[class_idx] = np.sum(class_mask)

        if class_counts[class_idx] == 0:
            continue

        class_targets = targets[class_mask]
        class_predictions = predictions[:, class_mask]  # [num_points, num_class_samples]

        # Compute accuracy at each t
        for t_idx in range(num_points):
            correct = (class_predictions[t_idx] == class_targets).sum()
            accuracies[class_idx, t_idx] = correct / class_counts[class_idx] * 100

    return accuracies, class_counts


def plot_class_accuracy_curves(accuracies, class_counts, ts, output_path):
    """Plot per-class accuracy vs t.

    Args:
        accuracies: [num_classes, num_points] - accuracy for each class
        class_counts: [num_classes] - samples per class
        ts: [num_points] - t values
        output_path: where to save figure
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 5 colors + 2 line styles for 10 classes
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = {
        0: '-', 1: '-', 2: '-', 3: '-', 4: '-',
        5: '--', 6: '--', 7: '--', 8: '--', 9: '--'
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each class
    for class_idx in range(10):
        if class_counts[class_idx] == 0:
            continue

        color = color_palette[class_idx % 5]
        linestyle = line_styles[class_idx]

        ax.plot(
            ts,
            accuracies[class_idx],
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker='o' if class_idx < 5 else '^',
            markersize=4,
            markevery=5,  # Show marker every 5 points
            label=f'{class_names[class_idx]} ({class_counts[class_idx]})',
            alpha=0.9
        )

    ax.set_xlabel('t (position along curve)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Accuracy Along Bezier Curve\n(Seed0 → Seed1)',
                 fontsize=15, fontweight='bold', pad=15)

    ax.legend(fontsize=11, loc='best', ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis limits
    ax.set_xlim(ts[0], ts[-1])

    # Add vertical lines at endpoints
    ax.axvline(x=0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add text annotations for endpoints
    ax.text(0.02, 0.98, 'Endpoint 0\n(seed0)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))
    ax.text(0.98, 0.98, 'Endpoint 1\n(seed1)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()


def print_statistics(accuracies, class_counts, ts):
    """Print statistics about accuracy drops."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n" + "="*70)
    print("PER-CLASS ACCURACY STATISTICS")
    print("="*70)

    for class_idx in range(10):
        if class_counts[class_idx] == 0:
            continue

        acc_start = accuracies[class_idx, 0]
        acc_end = accuracies[class_idx, -1]
        acc_min = accuracies[class_idx].min()
        acc_max = accuracies[class_idx].max()
        acc_mean = accuracies[class_idx].mean()

        # Find where minimum occurs
        min_idx = accuracies[class_idx].argmin()
        t_min = ts[min_idx]

        # Maximum drop from endpoints
        drop_from_start = acc_start - acc_min
        drop_from_end = acc_end - acc_min
        max_drop = max(drop_from_start, drop_from_end)

        print(f"\n{class_names[class_idx].upper():<12} (n={class_counts[class_idx]})")
        print(f"  Start (t=0.0): {acc_start:5.2f}%")
        print(f"  End   (t=1.0): {acc_end:5.2f}%")
        print(f"  Minimum:       {acc_min:5.2f}% at t={t_min:.3f}")
        print(f"  Mean:          {acc_mean:5.2f}%")
        print(f"  Max drop:      {max_drop:5.2f}%")

    print("\n" + "="*70)

    # Overall statistics
    overall_start = (predictions[0] == targets).sum() / len(targets) * 100
    overall_end = (predictions[-1] == targets).sum() / len(targets) * 100

    # Compute overall accuracy at each t
    overall_acc = np.array([(predictions[i] == targets).sum() / len(targets) * 100
                            for i in range(len(ts))])
    overall_min = overall_acc.min()
    overall_mean = overall_acc.mean()
    min_idx = overall_acc.argmin()
    t_min_overall = ts[min_idx]

    print("\nOVERALL ACCURACY")
    print(f"  Start (t=0.0): {overall_start:.2f}%")
    print(f"  End   (t=1.0): {overall_end:.2f}%")
    print(f"  Minimum:       {overall_min:.2f}% at t={t_min_overall:.3f}")
    print(f"  Mean:          {overall_mean:.2f}%")
    print(f"  Max drop:      {max(overall_start - overall_min, overall_end - overall_min):.2f}%")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot per-class accuracy vs t')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions_detailed.npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for figure')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions data
    print(f"Loading predictions from {args.predictions}...")
    data = np.load(args.predictions)

    global predictions, targets  # For statistics function
    predictions = data['predictions']  # [num_points, num_samples]
    targets = data['targets']
    ts = data['ts']

    print(f"\nLoaded data:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Number of samples: {len(targets)}")
    print(f"  Number of t points: {len(ts)}")
    print(f"  t range: {ts[0]:.3f} to {ts[-1]:.3f}")

    # Compute per-class accuracies
    print("\nComputing per-class accuracies...")
    accuracies, class_counts = compute_per_class_accuracy(predictions, targets, ts)

    # Print statistics
    print_statistics(accuracies, class_counts, ts)

    # Create plot
    print("Creating plot...")
    output_path = output_dir / 'class_accuracy_vs_t.png'
    plot_class_accuracy_curves(accuracies, class_counts, ts, output_path)

    print(f"\nVisualization complete! Figure saved to {output_dir}")


if __name__ == "__main__":
    main()
