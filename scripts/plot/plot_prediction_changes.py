"""Visualize prediction changes in 2D using UMAP.

Creates scatter plots of samples that change predictions, using UMAP
for dimensionality reduction of averaged endpoint features.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not available, will use PCA instead")
    from sklearn.decomposition import PCA


def reduce_dimensions(features, method='umap', n_components=2, random_state=42):
    """Reduce feature dimensions to 2D.

    Args:
        features: [n_samples, feature_dim] array
        method: 'umap' or 'pca'
        n_components: number of dimensions (default: 2)
        random_state: random seed

    Returns:
        [n_samples, n_components] reduced features
    """
    print(f"\nReducing dimensions using {method.upper()}...")
    print(f"  Input shape: {features.shape}")

    if method == 'umap' and UMAP_AVAILABLE:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=random_state,
            verbose=True
        )
        reduced = reducer.fit_transform(features)
    else:
        if method == 'umap' and not UMAP_AVAILABLE:
            print("  UMAP not available, falling back to PCA")
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        print(f"  Explained variance: {explained_var}")
        print(f"  Total explained variance: {explained_var.sum():.3f}")

    print(f"  Output shape: {reduced.shape}")
    return reduced


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

    # Color scheme for CIFAR-10 classes
    colors = plt.cm.tab10(np.arange(10))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each class separately
    for class_idx in range(10):
        mask = targets == class_idx
        if np.sum(mask) == 0:
            continue

        class_embeddings = embeddings[mask]
        class_changes = change_counts[mask]

        # Size based on number of changes (larger = more unstable)
        sizes = 20 + class_changes * 15

        scatter = ax.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=[colors[class_idx]],
            s=sizes,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize prediction changes in 2D')
    parser.add_argument('--analysis', type=str, required=True,
                        help='Path to changing_samples_info.npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for figures')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'pca'],
                        help='Dimensionality reduction method')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--plot_all', action='store_true',
                        help='Also create plot showing all samples (stable + changing)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Reduce dimensions for changing samples
    embeddings_changing = reduce_dimensions(
        features_changing,
        method=args.method,
        random_state=args.random_state
    )

    # Plot changing samples
    plot_changing_samples(
        embeddings_changing,
        targets_changing,
        changing_counts,
        output_dir / 'prediction_changes_2d.png',
        title='Samples that Change Predictions Along Bezier Curve\n(Using Endpoint 0 Features)'
    )

    # Optionally plot all samples
    if args.plot_all:
        print("\nReducing dimensions for all samples...")
        embeddings_all = reduce_dimensions(
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
