"""
Plot comparison between polygon chain (unconstrained), symmetry plane (constrained), and linear interpolation.

Shows that constraining to symmetry plane doesn't degrade performance compared to unconstrained polygon chain.
All curves are optional - you can plot any combination.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add lib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from lib.analysis import plotting
from lib.utils.args import ArgumentParserBuilder


def plot_comparison(args):
    """Create comparison plot for polygon chain vs symmetry plane vs linear."""

    # Load data (all optional)
    print("Loading evaluation data...")
    polygon_data = np.load(args.polygon_file) if args.polygon_file else None
    symplane_data = np.load(args.symplane_file) if args.symplane_file else None
    linear_data = np.load(args.linear_file) if args.linear_file else None

    # Check that at least one curve is provided
    if not any([polygon_data is not None, symplane_data is not None, linear_data is not None]):
        raise ValueError("At least one curve file must be provided!")

    # Extract t values (use first available)
    t_polygon = polygon_data['ts'] if polygon_data is not None else None
    t_symplane = symplane_data['ts'] if symplane_data is not None else None
    t_linear = linear_data['ts'] if linear_data is not None else None

    # Build title based on what's being plotted
    curves_plotted = []
    if polygon_data is not None:
        curves_plotted.append("Polygon Chain")
    if symplane_data is not None:
        curves_plotted.append("Symmetry Plane")
    if linear_data is not None:
        curves_plotted.append("Linear")
    title = ' vs '.join(curves_plotted) + ' Comparison'

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot styles
    styles = {
        'linear': {'color': '#d62728', 'linestyle': '--', 'label': 'Linear Interpolation',
                   'alpha': 0.7, 'linewidth': 2, 'marker': 'o', 'markersize': 3, 'markevery': 5},
        'polygon': {'color': '#2ca02c', 'linestyle': '-', 'label': 'Polygon Chain (unconstrained)',
                    'alpha': 0.9, 'linewidth': 2.5, 'marker': 's', 'markersize': 3, 'markevery': 5},
        'symplane': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Symmetry Plane (constrained)',
                     'alpha': 0.9, 'linewidth': 2.5, 'marker': '^', 'markersize': 4, 'markevery': 5},
    }

    # Panel 1: Test Error
    ax = axes[0, 0]
    if linear_data is not None:
        ax.plot(t_linear, linear_data['te_err'], **styles['linear'])
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['te_err'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['te_err'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Test Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 2: Test Loss
    ax = axes[0, 1]
    if linear_data is not None:
        ax.plot(t_linear, linear_data['te_loss'], **styles['linear'])
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['te_loss'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['te_loss'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Train Error
    ax = axes[1, 0]
    if linear_data is not None:
        ax.plot(t_linear, linear_data['tr_err'], **styles['linear'])
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['tr_err'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['tr_err'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Error (%)', fontsize=12)
    ax.set_title('Train Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 4: Train Loss
    ax = axes[1, 1]
    if linear_data is not None:
        ax.plot(t_linear, linear_data['tr_loss'], **styles['linear'])
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['tr_loss'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['tr_loss'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Train Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Add vertical line at t=0.5 (middle bend location)
    for ax in axes.flat:
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Print summary statistics
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

    # Calculate metrics for each curve
    metrics = {}
    if linear_data is not None:
        linear_max_err = np.max(linear_data['te_err'])
        linear_endpoint_err = (linear_data['te_err'][0] + linear_data['te_err'][-1]) / 2
        linear_barrier = linear_max_err - linear_endpoint_err
        metrics['linear'] = {'max_err': linear_max_err, 'barrier': linear_barrier, 't': t_linear}

    if polygon_data is not None:
        polygon_max_err = np.max(polygon_data['te_err'])
        polygon_endpoint_err = (polygon_data['te_err'][0] + polygon_data['te_err'][-1]) / 2
        polygon_barrier = polygon_max_err - polygon_endpoint_err
        metrics['polygon'] = {'max_err': polygon_max_err, 'barrier': polygon_barrier, 't': t_polygon}

    if symplane_data is not None:
        symplane_max_err = np.max(symplane_data['te_err'])
        symplane_endpoint_err = (symplane_data['te_err'][0] + symplane_data['te_err'][-1]) / 2
        symplane_barrier = symplane_max_err - symplane_endpoint_err
        metrics['symplane'] = {'max_err': symplane_max_err, 'barrier': symplane_barrier, 't': t_symplane}

    print("\nMaximum Test Error:")
    if 'linear' in metrics:
        print(f"  Linear:         {metrics['linear']['max_err']:.2f}%")
    if 'polygon' in metrics:
        print(f"  Polygon Chain:  {metrics['polygon']['max_err']:.2f}%")
    if 'symplane' in metrics:
        print(f"  Symmetry Plane: {metrics['symplane']['max_err']:.2f}%")

    print("\nBarrier Height (max - endpoint avg):")
    if 'linear' in metrics:
        print(f"  Linear:         {metrics['linear']['barrier']:.2f}%")
    if 'polygon' in metrics:
        print(f"  Polygon Chain:  {metrics['polygon']['barrier']:.2f}%")
    if 'symplane' in metrics:
        print(f"  Symmetry Plane: {metrics['symplane']['barrier']:.2f}%")

    # Comparison stats (only if relevant curves present)
    if 'symplane' in metrics and 'polygon' in metrics:
        print("\nDifference (Symmetry Plane - Polygon Chain):")
        err_diff = metrics['symplane']['max_err'] - metrics['polygon']['max_err']
        barrier_diff = metrics['symplane']['barrier'] - metrics['polygon']['barrier']
        print(f"  Max Test Error: {err_diff:+.3f}%")
        print(f"  Barrier Height: {barrier_diff:+.3f}%")

        if abs(barrier_diff) < 0.5:
            print("\n✓ Symmetry plane constraint does NOT degrade performance!")
            print(f"  Difference is only {abs(barrier_diff):.3f}% (negligible)")

    if 'linear' in metrics:
        print("\nBarrier Reduction vs Linear:")
        if 'polygon' in metrics:
            reduction = (metrics['linear']['barrier'] - metrics['polygon']['barrier']) / metrics['linear']['barrier'] * 100
            print(f"  Polygon Chain:  {reduction:.1f}%")
        if 'symplane' in metrics:
            reduction = (metrics['linear']['barrier'] - metrics['symplane']['barrier']) / metrics['linear']['barrier'] * 100
            print(f"  Symmetry Plane: {reduction:.1f}%")

    print("=" * 80)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Determine output path (use first available file's directory)
    if args.output:
        output_path = args.output
    else:
        # Find first available file to use its directory
        first_file = args.polygon_file or args.symplane_file or args.linear_file
        output_path = os.path.join(
            os.path.dirname(first_file), '../figures/polygon_symmetry_comparison.png'
        )
    # Save figure
    plotting.save_figure(fig, output_path)

    # Save summary to text file
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(title.upper())
    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Write metrics for available curves
    summary_lines.append("Maximum Test Error:")
    if 'linear' in metrics:
        summary_lines.append(f"  Linear:         {metrics['linear']['max_err']:.2f}%")
    if 'polygon' in metrics:
        summary_lines.append(f"  Polygon Chain:  {metrics['polygon']['max_err']:.2f}%")
    if 'symplane' in metrics:
        summary_lines.append(f"  Symmetry Plane: {metrics['symplane']['max_err']:.2f}%")
    summary_lines.append("")

    summary_lines.append("Barrier Height:")
    if 'linear' in metrics:
        summary_lines.append(f"  Linear:         {metrics['linear']['barrier']:.2f}%")
    if 'polygon' in metrics:
        summary_lines.append(f"  Polygon Chain:  {metrics['polygon']['barrier']:.2f}%")
    if 'symplane' in metrics:
        summary_lines.append(f"  Symmetry Plane: {metrics['symplane']['barrier']:.2f}%")
    summary_lines.append("")

    # Write comparison stats only if relevant curves present
    if 'symplane' in metrics and 'polygon' in metrics:
        err_diff = metrics['symplane']['max_err'] - metrics['polygon']['max_err']
        barrier_diff = metrics['symplane']['barrier'] - metrics['polygon']['barrier']
        summary_lines.append(f"Difference (Symmetry - Polygon): {barrier_diff:+.3f}%")
        summary_lines.append("")

    if 'linear' in metrics:
        summary_lines.append("Barrier Reduction vs Linear:")
        if 'polygon' in metrics:
            reduction = (metrics['linear']['barrier'] - metrics['polygon']['barrier']) / metrics['linear']['barrier'] * 100
            summary_lines.append(f"  Polygon Chain:  {reduction:.1f}%")
        if 'symplane' in metrics:
            reduction = (metrics['linear']['barrier'] - metrics['symplane']['barrier']) / metrics['linear']['barrier'] * 100
            summary_lines.append(f"  Symmetry Plane: {reduction:.1f}%")
        summary_lines.append("")

    summary_lines.append("=" * 80)

    summary_path = output_path.replace('.png', '_summary.txt')
    plotting.save_summary_text(summary_lines, summary_path)

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Polygon Chain vs Symmetry Plane Comparison'
    )

    # Custom arguments
    parser.add_argument('--polygon-file', type=str, default=None,
                       help='Path to polygon chain curve.npz file (optional)')
    parser.add_argument('--symplane-file', type=str, default=None,
                       help='Path to symmetry plane curve.npz file (optional)')
    parser.add_argument('--linear-file', type=str, default=None,
                       help='Path to linear.npz file (optional)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_plot_output_args(parser, required=False)

    args = parser.parse_args()
    plot_comparison(args)
