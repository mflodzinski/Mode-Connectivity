"""Create animated GIF showing layer-wise distance evolution along the curve.

This visualization shows how each layer changes as we move along the connectivity
curve from the original network to the neuron-swapped version.

The animation reveals whether corrections are:
- Local: Only the swapped layer shows significant changes
- Global: All layers adjust to accommodate the swap
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os
import sys

# Add lib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from lib.analysis import plotting
from lib.utils.args import ArgumentParserBuilder


def load_analysis_data(data_path):
    """Load analysis results from NPZ file.

    Returns:
        Dictionary with analysis data
    """
    data = np.load(data_path, allow_pickle=True)

    return {
        'layer_names': data['layer_names'].tolist(),
        'layer_types': data['layer_types'].tolist(),
        'layer_params': data['layer_params'].tolist(),
        't_values': data['t_values'],
        'normalized_l2': data['normalized_l2'],
        'relative': data['relative'],
        'raw_l2': data['raw_l2'],
        'swapped_block': int(data['swapped_block']),
        'swapped_layer': int(data['swapped_layer'])
    }


def identify_swapped_layer_idx(layer_names, layer_types, swapped_block, swapped_layer_idx):
    """Find the index of the swapped layer in the layer list.

    Args:
        layer_names: List of layer names
        layer_types: List of layer types
        swapped_block: Block index of swapped layer
        swapped_layer_idx: Layer index within block

    Returns:
        Index in layer_names list, or None if not found
    """
    target_name = f"block{swapped_block}_conv{swapped_layer_idx}"

    for idx, name in enumerate(layer_names):
        if name == target_name:
            return idx

    return None


def create_animation(data, output_path, fps=10, distance_metric='normalized_l2'):
    """Create animated GIF of layer distances.

    Args:
        data: Dictionary with analysis data
        output_path: Path to save GIF
        fps: Frames per second
        distance_metric: Which distance metric to plot
    """
    print("="*70)
    print("CREATING ANIMATION")
    print("="*70)

    # Extract data
    layer_names = data['layer_names']
    t_values = data['t_values']
    distances = data[distance_metric]  # Shape: (num_t, num_layers)
    num_t, num_layers = distances.shape

    # Find swapped layer
    swapped_idx = identify_swapped_layer_idx(
        layer_names,
        data['layer_types'],
        data['swapped_block'],
        data['swapped_layer']
    )

    print(f"\nAnimation settings:")
    print(f"  Number of frames: {num_t}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Distance metric: {distance_metric}")
    print(f"  Swapped layer: {layer_names[swapped_idx] if swapped_idx is not None else 'Unknown'}")
    print(f"  FPS: {fps}")

    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare layer indices for x-axis
    layer_indices = np.arange(num_layers)

    # Determine y-axis limits (use max across all t values)
    y_max = distances.max() * 1.1
    y_min = 0

    # Color scheme
    color_normal = '#3498db'  # Blue
    color_swapped = '#e74c3c'  # Red
    color_conv = '#2ecc71'  # Green
    color_fc = '#9b59b6'  # Purple

    def update(frame):
        """Update function for animation."""
        ax.clear()

        t = t_values[frame]
        dist_at_t = distances[frame]

        # Create bar colors (highlight swapped layer)
        colors = []
        for idx, layer_type in enumerate(data['layer_types']):
            if idx == swapped_idx:
                colors.append(color_swapped)
            elif layer_type == 'conv':
                colors.append(color_conv)
            else:
                colors.append(color_fc)

        # Plot bars
        bars = ax.bar(layer_indices, dist_at_t, color=colors, alpha=0.7, edgecolor='black')

        # Highlight swapped layer with border
        if swapped_idx is not None:
            bars[swapped_idx].set_linewidth(3)
            bars[swapped_idx].set_edgecolor('red')

        # Formatting
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{distance_metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer-wise Distance Evolution Along Curve (t = {t:.3f})',
                    fontsize=14, fontweight='bold', pad=20)

        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.set_ylim(y_min, y_max)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add layer names as x-tick labels (rotated)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=90, ha='right', fontsize=8)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_swapped, edgecolor='black', label='Swapped Layer'),
            Patch(facecolor=color_conv, edgecolor='black', label='Conv Layer', alpha=0.7),
            Patch(facecolor=color_fc, edgecolor='black', label='FC Layer', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # Add progress bar at top
        progress_ax = fig.add_axes([0.15, 0.95, 0.7, 0.02])
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_ax.axis('off')

        # Progress bar background
        progress_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black'))
        # Progress bar fill
        progress_ax.add_patch(Rectangle((0, 0), t, 1, facecolor='#3498db', edgecolor='black'))
        # Add text
        progress_ax.text(0.5, 0.5, f'Progress: {100*t:.1f}%',
                        ha='center', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Create animation
    print("\nGenerating frames...")
    anim = animation.FuncAnimation(fig, update, frames=num_t, interval=1000/fps, repeat=True)

    # Save as GIF
    print(f"Saving animation to: {output_path}")
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close(fig)

    print(f"✓ Animation saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("="*70)


def create_static_heatmap(data, output_path, distance_metric='normalized_l2'):
    """Create static heatmap showing all layers across all t values.

    Args:
        data: Dictionary with analysis data
        output_path: Path to save figure
        distance_metric: Which distance metric to plot
    """
    print("\nCreating static heatmap...")

    layer_names = data['layer_names']
    t_values = data['t_values']
    distances = data[distance_metric]  # Shape: (num_t, num_layers)

    # Find swapped layer
    swapped_idx = identify_swapped_layer_idx(
        layer_names,
        data['layer_types'],
        data['swapped_block'],
        data['swapped_layer']
    )

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create heatmap
    im = ax.imshow(distances.T, aspect='auto', cmap='viridis', origin='lower')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=f'{distance_metric.replace("_", " ").title()}')

    # Formatting
    ax.set_xlabel('Position along curve (t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Distance Evolution: Heatmap View',
                fontsize=14, fontweight='bold', pad=20)

    # Set ticks
    num_t = len(t_values)
    num_layers = len(layer_names)

    # X-axis: show t values at regular intervals
    x_tick_indices = np.linspace(0, num_t-1, min(11, num_t), dtype=int)
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([f'{t_values[i]:.2f}' for i in x_tick_indices])

    # Y-axis: show layer names
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(layer_names, fontsize=8)

    # Highlight swapped layer
    if swapped_idx is not None:
        ax.axhline(y=swapped_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(num_t * 1.02, swapped_idx, 'Swapped', color='red', fontsize=10,
                fontweight='bold', va='center')

    plt.tight_layout()
    plotting.save_figure(fig, output_path)

    print(f"  Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create animated visualization of layer distance evolution'
    )

    # Custom arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to layer distances NPZ file')
    parser.add_argument('--metric', type=str, default='normalized_l2',
                        choices=['normalized_l2', 'relative', 'raw_l2'],
                        help='Distance metric to plot (default: normalized_l2)')
    parser.add_argument('--heatmap', action='store_true',
                        help='Also create static heatmap')

    # Standard arguments
    ArgumentParserBuilder.add_plot_output_args(parser, required=True)
    ArgumentParserBuilder.add_animation_args(parser)

    args = parser.parse_args()

    # Load data
    print("Loading analysis data...")
    data = load_analysis_data(args.data)
    print(f"✓ Loaded data with {len(data['layer_names'])} layers and {len(data['t_values'])} time points")

    # Create animation
    create_animation(data, args.output, args.fps, args.metric)

    # Create heatmap if requested
    if args.heatmap:
        heatmap_path = args.output.replace('.gif', '_heatmap.png')
        create_static_heatmap(data, heatmap_path, args.metric)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - Animation: {args.output}")
    if args.heatmap:
        print(f"  - Heatmap: {heatmap_path}")
    print("\nView the animation to see how each layer changes along the curve.")
    print("Red bars indicate the swapped layer.")
    print("="*70)


if __name__ == "__main__":
    main()
