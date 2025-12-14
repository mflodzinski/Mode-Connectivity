"""Visualize prediction changes in 2D using UMAP.

Creates scatter plots of samples that change predictions, using UMAP
for dimensionality reduction of averaged endpoint features.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add lib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from lib.analysis import dim_reduction, plotting
from lib.utils.args import ArgumentParserBuilder


def plot_changing_samples(embeddings, targets, change_counts, output_path, title=None):
    """Plot 2D scatter of changing samples.

    Args:
        embeddings: [n_samples, 2] - 2D coordinates
        targets: [n_samples] - true class labels
        change_counts: [n_samples] - number of prediction changes
        output_path: path to save figure
        title: optional title
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 5 colors shared across 10 classes
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Shapes: circles for 0-4, triangles for 5-9
    markers = {
        0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o',
        5: '^', 6: '^', 7: '^', 8: '^', 9: '^'
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each class separately
    for class_idx in range(10):
        mask = targets == class_idx
        if np.sum(mask) == 0:
            continue

        class_embeddings = embeddings[mask]
        class_changes = change_counts[mask]

        # Size based on number of changes (larger = more unstable)
        sizes = 30 + class_changes * 20

        # Color: class_idx % 5 gives us 5 colors
        color = color_palette[class_idx % 5]
        marker = markers[class_idx]

        scatter = ax.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=color,
            s=sizes,
            marker=marker,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.8,
            label=f'{class_names[class_idx]} ({np.sum(mask)})'
        )

    ax.set_xlabel('UMAP Dimension 1', fontsize=13)
    ax.set_ylabel('UMAP Dimension 2', fontsize=13)

    if title is None:
        title = 'Samples that Change Predictions Along Curve'
    ax.set_title(title + '\n(Size ∝ Number of Prediction Changes)',
                 fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add size legend
    max_changes = change_counts.max()
    if max_changes > 1:
        # Create dummy points for size legend
        legend_changes = [1, max_changes//2, max_changes] if max_changes > 2 else [1, max_changes]
        legend_sizes = [20 + c * 15 for c in legend_changes]

        for i, (changes, size) in enumerate(zip(legend_changes, legend_sizes)):
            ax.scatter([], [], c='gray', s=size, alpha=0.7,
                      edgecolors='black', linewidths=0.5,
                      label=f'{changes} change{"s" if changes > 1 else ""}')

    plt.tight_layout()
    plotting.save_figure(fig, output_path)
    plt.close()


def plot_all_samples_comparison(embeddings_all, embeddings_changing, targets_all,
                                  changing_indices, output_path):
    """Plot all samples with changing samples highlighted.

    Args:
        embeddings_all: [n_total, 2] - all sample embeddings
        embeddings_changing: [n_changing, 2] - changing sample embeddings
        targets_all: [n_total] - all true labels
        changing_indices: [n_changing] - indices of changing samples
        output_path: path to save figure
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.tab10(np.arange(10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: All samples
    for class_idx in range(10):
        mask = targets_all == class_idx
        if np.sum(mask) == 0:
            continue

        class_embeddings = embeddings_all[mask]

        ax1.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=[colors[class_idx]],
            s=10,
            alpha=0.5,
            label=f'{class_names[class_idx]}'
        )

    ax1.set_xlabel('UMAP Dimension 1', fontsize=13)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=13)
    ax1.set_title('All Samples', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right plot: Stable vs Changing
    stable_mask = np.ones(len(targets_all), dtype=bool)
    stable_mask[changing_indices] = False
    stable_indices = np.where(stable_mask)[0]

    # Plot stable samples in gray
    ax2.scatter(
        embeddings_all[stable_indices, 0],
        embeddings_all[stable_indices, 1],
        c='lightgray',
        s=5,
        alpha=0.3,
        label=f'Stable ({len(stable_indices)})'
    )

    # Plot changing samples by class
    targets_changing = targets_all[changing_indices]
    for class_idx in range(10):
        mask = targets_changing == class_idx
        if np.sum(mask) == 0:
            continue

        class_embeddings = embeddings_changing[mask]

        ax2.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=[colors[class_idx]],
            s=30,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            label=f'{class_names[class_idx]} ({np.sum(mask)})'
        )

    ax2.set_xlabel('UMAP Dimension 1', fontsize=13)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=13)
    ax2.set_title('Stable vs Changing Samples', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plotting.save_figure(fig, output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize prediction changes in 2D')

    # Custom arguments
    parser.add_argument('--analysis', type=str, required=True,
                        help='Path to changing_samples_info.npz')
    parser.add_argument('--plot-all', action='store_true',
                        help='Also create plot showing all samples (stable + changing)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_plot_output_args(parser)
    ArgumentParserBuilder.add_dimred_args(parser)

    args = parser.parse_args()

    # Create output directory
    output_dir = plotting.create_output_dir(args.output)

    # Load analysis data
    print(f"Loading analysis from {args.analysis}...")
    data = np.load(args.analysis)

    indices = data['indices']
    change_counts = data['change_counts']
    changing_counts = data['changing_counts']
    targets_changing = data['targets_changing']
    features_for_visualization = data['features_for_visualization']
    features_changing = data['features_changing']

    print(f"\nLoaded data:")
    print(f"  Total samples: {len(features_for_visualization)}")
    print(f"  Changing samples: {len(indices)}")
    print(f"  Feature dimension: {features_for_visualization.shape[1]}")
    print(f"  Changing samples: {len(features_changing)} "
          f"({len(features_changing)/len(features_for_visualization)*100:.2f}%)")

    # STEP 1: Reduce dimensions for ALL changing samples first
    print("\nReducing dimensions for all changing samples...")
    embeddings_changing_full = dim_reduction.reduce_dimensions(
        features_changing,
        method=args.method,
        random_state=args.random_state
    )

    # STEP 2: Subsample 10% from each class for cleaner visualization
    subsample_idx = dim_reduction.subsample_per_class(
        targets_changing,
        num_classes=10,
        fraction=0.1,
        random_state=args.random_state
    )

    # Extract subsampled data
    embeddings_subsample = embeddings_changing_full[subsample_idx]
    targets_subsample = targets_changing[subsample_idx]
    counts_subsample = changing_counts[subsample_idx]

    # Plot changing samples
    plot_changing_samples(
        embeddings_subsample,
        targets_subsample,
        counts_subsample,
        output_dir / 'prediction_changes_2d.png',
        title='Samples that Change Predictions Along Bezier Curve\n(Using Endpoint 0 Features, 10% Per-Class Subsample)'
    )

    # Optionally plot all samples
    if args.plot_all:
        print("\nReducing dimensions for all samples...")
        embeddings_all = dim_reduction.reduce_dimensions(
            features_for_visualization,
            method=args.method,
            random_state=args.random_state
        )

        # Reconstruct targets for all samples (need to get from somewhere)
        # For now, we'll skip this plot unless we have all targets
        print("Warning: --plot_all requires full targets array, skipping...")

    print(f"\nVisualization complete! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
