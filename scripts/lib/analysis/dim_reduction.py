"""Dimensionality reduction utilities for visualization.

Extracted from plot scripts to avoid duplication.
"""
import numpy as np

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
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


def subsample_per_class(targets, num_classes=10, fraction=0.1, random_state=42):
    """Subsample indices from each class separately.

    Args:
        targets: [n_samples] array of class labels
        num_classes: number of classes (default: 10)
        fraction: fraction to sample from each class (default: 0.1)
        random_state: random seed

    Returns:
        Array of subsampled indices
    """
    np.random.seed(random_state)
    subsample_idx = []

    print(f"\nSubsampling {fraction*100:.0f}% from each class:")
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        class_indices = np.where(class_mask)[0]
        n_class = len(class_indices)

        if n_class > 0:
            n_subsample_class = max(1, int(n_class * fraction))
            sampled = np.random.choice(class_indices, size=n_subsample_class, replace=False)
            subsample_idx.extend(sampled)
            print(f"  Class {class_idx}: {n_class} → {n_subsample_class} samples")

    subsample_idx = np.array(subsample_idx)
    print(f"\nTotal subsampled: {len(subsample_idx)}")
    return subsample_idx
