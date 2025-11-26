"""Create animated GIF showing prediction changes along the curve.

Shows how predictions evolve as we move along the Bezier curve,
with samples colored by their current prediction at each t value.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from tqdm import tqdm

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    from sklearn.decomposition import PCA


def reduce_dimensions(features, method='umap', n_components=2, random_state=42):
    """Reduce feature dimensions to 2D."""
    print(f"\nReducing dimensions using {method.upper()}...")

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
            print("  UMAP not available, using PCA")
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(features)

    return reduced


def create_frame(embeddings, current_preds, true_labels, prev_preds, t_value,
                 t_index, num_points, output_path):
    """Create a single frame showing predictions at time t.

    Args:
        embeddings: [n_samples, 2] - fixed 2D positions
        current_preds: [n_samples] - predictions at current t
        true_labels: [n_samples] - ground truth labels
        prev_preds: [n_samples] - predictions at previous t (or None)
        t_value: current t value
        t_index: current index in t array
        num_points: total number of t points
        output_path: where to save frame
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

    # Identify samples that just changed (if we have previous predictions)
    just_changed = np.zeros(len(current_preds), dtype=bool)
    if prev_preds is not None:
        just_changed = current_preds != prev_preds

    # Plot samples grouped by current prediction
    for class_idx in range(10):
        mask = current_preds == class_idx
        if np.sum(mask) == 0:
            continue

        class_embeddings = embeddings[mask]
        class_true_labels = true_labels[mask]
        class_just_changed = just_changed[mask]

        # Determine edge colors: green if correct, red if incorrect
        edge_colors = ['green' if true_labels[i] == class_idx else 'red'
                       for i in np.where(mask)[0]]

        # Make recently changed samples larger and with thicker edges
        sizes = np.where(class_just_changed, 150, 50)
        linewidths = np.where(class_just_changed, 3.0, 1.0)

        # Color: class_idx % 5 gives us 5 colors
        color = color_palette[class_idx % 5]
        marker = markers[class_idx]

        ax.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=color,
            s=sizes,
            marker=marker,
            alpha=0.8,
            edgecolors=edge_colors,
            linewidths=linewidths,
            label=f'{class_names[class_idx]} ({np.sum(mask)})'
        )

    ax.set_xlabel('UMAP Dimension 1', fontsize=13)
    ax.set_ylabel('UMAP Dimension 2', fontsize=13)

    # Title with progress bar
    progress = t_index / (num_points - 1) * 100
    ax.set_title(
        f'Predictions Along Bezier Curve at t={t_value:.3f}\n'
        f'Progress: {progress:.1f}% | '
        f'Green edge = Correct | Red edge = Incorrect | Large = Just Changed',
        fontsize=13, fontweight='bold'
    )

    ax.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set consistent axis limits across all frames
    # We'll set these based on the full embedding range
    ax.set_xlim(embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1)
    ax.set_ylim(embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Lower DPI for GIF
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create prediction change animation')
    parser.add_argument('--analysis', type=str, required=True,
                        help='Path to changing_samples_info.npz')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions_detailed.npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for GIF file')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'pca'],
                        help='Dimensionality reduction method')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for GIF')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='Only render every Nth frame (for faster generation)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.parent / 'animation_frames'
    frames_dir.mkdir(exist_ok=True)

    # Load analysis data
    print(f"Loading analysis from {args.analysis}...")
    analysis = np.load(args.analysis)

    indices = analysis['indices']
    targets_changing = analysis['targets_changing']
    features_changing = analysis['features_changing']

    # Load predictions data
    print(f"Loading predictions from {args.predictions}...")
    pred_data = np.load(args.predictions)

    predictions_all = pred_data['predictions']  # [num_points, num_samples]
    ts = pred_data['ts']

    # Extract predictions for changing samples only
    predictions_changing = predictions_all[:, indices]  # [num_points, num_changing]

    num_points = len(ts)
    num_changing = len(indices)

    print(f"\nData loaded:")
    print(f"  Changing samples: {num_changing}")
    print(f"  Number of time points: {num_points}")
    print(f"  t range: {ts[0]:.3f} to {ts[-1]:.3f}")

    # STEP 1: Reduce dimensions for ALL changing samples first
    print("\nReducing dimensions for all changing samples...")
    embeddings_full = reduce_dimensions(
        features_changing,
        method=args.method,
        random_state=args.random_state
    )

    # STEP 2: Subsample 10% from each class for cleaner visualization
    np.random.seed(args.random_state)
    subsample_idx = []

    print("\nSubsampling 10% from each class:")
    for class_idx in range(10):
        class_mask = targets_changing == class_idx
        class_indices = np.where(class_mask)[0]
        n_class = len(class_indices)

        if n_class > 0:
            n_subsample_class = max(1, int(n_class * 0.1))
            sampled = np.random.choice(class_indices, size=n_subsample_class, replace=False)
            subsample_idx.extend(sampled)
            print(f"  Class {class_idx}: {n_class} → {n_subsample_class} samples")

    subsample_idx = np.array(subsample_idx)

    # Extract subsampled data
    embeddings = embeddings_full[subsample_idx]
    targets_subsample = targets_changing[subsample_idx]
    predictions_subsample = predictions_changing[:, subsample_idx]  # [num_points, n_subsample]

    print(f"\nTotal subsampled: {len(subsample_idx)} (from {num_changing})")

    # Generate frames
    print(f"\nGenerating frames...")
    frame_paths = []

    indices_to_render = range(0, num_points, args.skip_frames)

    prev_preds = None
    for i in tqdm(indices_to_render, desc="Creating frames"):
        t_value = ts[i]
        current_preds = predictions_subsample[i]

        frame_path = frames_dir / f'frame_{i:03d}.png'
        create_frame(
            embeddings,
            current_preds,
            targets_subsample,
            prev_preds,
            t_value,
            i,
            num_points,
            frame_path
        )
        frame_paths.append(frame_path)
        prev_preds = current_preds

    # Create GIF
    print(f"\nCreating GIF animation...")
    images = []
    for frame_path in tqdm(frame_paths, desc="Loading frames"):
        images.append(imageio.imread(frame_path))

    # Add pause at the end
    for _ in range(args.fps * 2):  # 2 second pause
        images.append(images[-1])

    imageio.mimsave(
        args.output,
        images,
        fps=args.fps,
        loop=0  # Infinite loop
    )

    print(f"\nGIF saved to: {args.output}")
    print(f"  Total frames: {len(frame_paths)}")
    print(f"  Frame rate: {args.fps} fps")
    print(f"  Duration: ~{len(images) / args.fps:.1f} seconds")

    # Optionally clean up frames
    cleanup = input("\nDelete individual frames? (y/N): ")
    if cleanup.lower() == 'y':
        for frame_path in frame_paths:
            frame_path.unlink()
        frames_dir.rmdir()
        print("Frames deleted.")
    else:
        print(f"Frames kept in: {frames_dir}")


if __name__ == "__main__":
    main()
