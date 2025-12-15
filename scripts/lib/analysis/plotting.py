"""
Plotting utilities.

Provides common plotting functions for analysis scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List


def setup_plot(figsize: Tuple[int, int] = (12, 6),
               title: Optional[str] = None,
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Setup a basic plot with common styling.

    Args:
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, alpha=0.3)

    return fig, ax


def save_plot(fig: plt.Figure,
              path: Union[str, Path],
              dpi: int = 300,
              bbox_inches: str = 'tight') -> None:
    """
    Save plot to file.

    Args:
        fig: Figure to save
        path: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box mode
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def plot_per_class_comparison(accuracies1: np.ndarray,
                              accuracies2: np.ndarray,
                              class_names: List[str],
                              model1_name: str = 'Model 1',
                              model2_name: str = 'Model 2',
                              output_path: Optional[Union[str, Path]] = None,
                              title: Optional[str] = None) -> plt.Figure:
    """
    Plot per-class accuracy comparison between two models.

    Args:
        accuracies1: Per-class accuracies for model 1
        accuracies2: Per-class accuracies for model 2
        class_names: List of class names
        model1_name: Name for model 1 in legend
        model2_name: Name for model 2 in legend
        output_path: Optional path to save plot
        title: Optional plot title

    Returns:
        Figure object
    """
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, accuracies1, width, label=model1_name, alpha=0.8)
    ax.bar(x + width/2, accuracies2, width, label=model2_name, alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          ax: Optional[plt.Axes] = None,
                          title: str = 'Confusion Matrix',
                          cmap: str = 'Blues',
                          normalize: bool = False) -> plt.Axes:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        ax: Optional axes to plot on (creates new if None)
        title: Plot title
        cmap: Colormap name
        normalize: Whether matrix is normalized (affects formatting)

    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)

    return ax


def plot_line_with_markers(x: np.ndarray,
                           y: np.ndarray,
                           label: str,
                           ax: Optional[plt.Axes] = None,
                           marker: str = 'o',
                           **kwargs) -> plt.Axes:
    """
    Plot line with markers.

    Args:
        x: X values
        y: Y values
        label: Line label
        ax: Optional axes to plot on
        marker: Marker style
        **kwargs: Additional kwargs for plot

    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, marker=marker, label=label, **kwargs)

    return ax


def plot_comparison_grid(data: dict,
                         class_names: List[str],
                         output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a grid of comparison plots.

    Args:
        data: Dictionary with plot data
        class_names: List of class names
        output_path: Optional path to save plot

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(16, 10))

    # This is a template - customize based on specific needs
    # Example: 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplots would be added here based on data
    # This is a placeholder for specific implementations

    if output_path:
        save_plot(fig, output_path)

    return fig


# Additional utility functions for plot scripts

def save_figure(fig: plt.Figure,
                path: Union[str, Path],
                dpi: int = 300,
                **kwargs) -> None:
    """
    Save figure with consistent defaults and directory creation.

    Args:
        fig: Figure to save
        path: Output file path
        dpi: Resolution in dots per inch (default: 300)
        **kwargs: Additional kwargs for savefig (e.g., bbox_inches='tight')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Set default bbox_inches if not provided
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'

    fig.savefig(path, dpi=dpi, **kwargs)
    print(f"\n✓ Figure saved to {path}")


def create_output_dir(path: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        path: Directory path or file path (will use parent directory)

    Returns:
        Path object for the directory
    """
    path = Path(path)

    # If path has a suffix, use parent directory
    if path.suffix:
        output_dir = path.parent
    else:
        output_dir = path

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_grid(ax: plt.Axes,
               alpha: float = 0.3,
               linestyle: str = '--',
               **kwargs) -> None:
    """
    Add grid to axes with consistent styling.

    Args:
        ax: Axes to add grid to
        alpha: Grid transparency (default: 0.3)
        linestyle: Grid line style (default: '--')
        **kwargs: Additional kwargs for grid()
    """
    ax.grid(True, alpha=alpha, linestyle=linestyle, **kwargs)


def save_summary_text(lines: List[str],
                      output_path: Union[str, Path]) -> None:
    """
    Save summary text to file.

    Args:
        lines: List of text lines to save
        output_path: Path to save text file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Summary saved to {output_path}")
