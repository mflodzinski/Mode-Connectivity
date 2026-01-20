"""
Plot comparison between unconstrained polygon, symmetry plane, random plane (midpoint), and random plane (random anchor).

Shows whether constraining to arbitrary planes enables low-loss mode connectivity.
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
    """Create comparison plot for different plane constraints."""

    # Load data (all optional)
    print("Loading evaluation data...")
    polygon_data = np.load(args.polygon_file) if args.polygon_file else None
    symplane_data = np.load(args.symplane_file) if args.symplane_file else None
    random_midpoint_data = np.load(args.random_midpoint_file) if args.random_midpoint_file else None
    random_random_data = np.load(args.random_random_file) if args.random_random_file else None

    # Check that at least one curve is provided
    if not any([polygon_data is not None, symplane_data is not None, random_midpoint_data is not None, random_random_data is not None]):
        raise ValueError("At least one curve file must be provided!")

    # Extract t values
    t_polygon = polygon_data['ts'] if polygon_data is not None else None
    t_symplane = symplane_data['ts'] if symplane_data is not None else None
    t_random_midpoint = random_midpoint_data['ts'] if random_midpoint_data is not None else None
    t_random_random = random_random_data['ts'] if random_random_data is not None else None

    # Build title based on what's being plotted
    curves_plotted = []
    if polygon_data is not None:
        curves_plotted.append("Polygon")
    if symplane_data is not None:
        curves_plotted.append("Symmetry Plane")
    if random_midpoint_data is not None:
        curves_plotted.append("Random Plane (Midpoint)")
    if random_random_data is not None:
        curves_plotted.append("Random Plane (Random)")
    title = ' vs '.join(curves_plotted) + ' Comparison'

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot styles
    styles = {
        'polygon': {'color': '#9467bd', 'linestyle': '-', 'label': 'Polygon Chain (unconstrained)',
                    'alpha': 0.9, 'linewidth': 2.5, 'marker': 'o', 'markersize': 3, 'markevery': 5},
        'symplane': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Symmetry Plane (perpendicular bisector)',
                     'alpha': 0.9, 'linewidth': 2.5, 'marker': '^', 'markersize': 4, 'markevery': 5},
        'random_midpoint': {'color': '#2ca02c', 'linestyle': '-', 'label': 'Random Plane (through midpoint)',
                    'alpha': 0.9, 'linewidth': 2.5, 'marker': 's', 'markersize': 3, 'markevery': 5},
        'random_random': {'color': '#ff7f0e', 'linestyle': '-', 'label': 'Random Plane (random anchor)',
                   'alpha': 0.9, 'linewidth': 2.5, 'marker': 'D', 'markersize': 3, 'markevery': 5},
    }

    # Panel 1: Test Error
    ax = axes[0, 0]
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['te_err'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['te_err'], **styles['symplane'])
    if random_midpoint_data is not None:
        ax.plot(t_random_midpoint, random_midpoint_data['te_err'], **styles['random_midpoint'])
    if random_random_data is not None:
        ax.plot(t_random_random, random_random_data['te_err'], **styles['random_random'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Test Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 2: Test Loss
    ax = axes[0, 1]
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['te_loss'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['te_loss'], **styles['symplane'])
    if random_midpoint_data is not None:
        ax.plot(t_random_midpoint, random_midpoint_data['te_loss'], **styles['random_midpoint'])
    if random_random_data is not None:
        ax.plot(t_random_random, random_random_data['te_loss'], **styles['random_random'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Train Error
    ax = axes[1, 0]
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['tr_err'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['tr_err'], **styles['symplane'])
    if random_midpoint_data is not None:
        ax.plot(t_random_midpoint, random_midpoint_data['tr_err'], **styles['random_midpoint'])
    if random_random_data is not None:
        ax.plot(t_random_random, random_random_data['tr_err'], **styles['random_random'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Error (%)', fontsize=12)
    ax.set_title('Train Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 4: Train Loss
    ax = axes[1, 1]
    if polygon_data is not None:
        ax.plot(t_polygon, polygon_data['tr_loss'], **styles['polygon'])
    if symplane_data is not None:
        ax.plot(t_symplane, symplane_data['tr_loss'], **styles['symplane'])
    if random_midpoint_data is not None:
        ax.plot(t_random_midpoint, random_midpoint_data['tr_loss'], **styles['random_midpoint'])
    if random_random_data is not None:
        ax.plot(t_random_random, random_random_data['tr_loss'], **styles['random_random'])
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

    if random_midpoint_data is not None:
        random_midpoint_max_err = np.max(random_midpoint_data['te_err'])
        random_midpoint_endpoint_err = (random_midpoint_data['te_err'][0] + random_midpoint_data['te_err'][-1]) / 2
        random_midpoint_barrier = random_midpoint_max_err - random_midpoint_endpoint_err
        metrics['random_midpoint'] = {'max_err': random_midpoint_max_err, 'barrier': random_midpoint_barrier, 't': t_random_midpoint}

    if random_random_data is not None:
        random_random_max_err = np.max(random_random_data['te_err'])
        random_random_endpoint_err = (random_random_data['te_err'][0] + random_random_data['te_err'][-1]) / 2
        random_random_barrier = random_random_max_err - random_random_endpoint_err
        metrics['random_random'] = {'max_err': random_random_max_err, 'barrier': random_random_barrier, 't': t_random_random}

    print("\nMaximum Test Error:")
    if 'polygon' in metrics:
        print(f"  Polygon Chain:               {metrics['polygon']['max_err']:.2f}%")
    if 'symplane' in metrics:
        print(f"  Symmetry Plane:              {metrics['symplane']['max_err']:.2f}%")
    if 'random_midpoint' in metrics:
        print(f"  Random Plane (midpoint):     {metrics['random_midpoint']['max_err']:.2f}%")
    if 'random_random' in metrics:
        print(f"  Random Plane (random):       {metrics['random_random']['max_err']:.2f}%")

    print("\nBarrier Height (max - endpoint avg):")
    if 'polygon' in metrics:
        print(f"  Polygon Chain:               {metrics['polygon']['barrier']:.2f}%")
    if 'symplane' in metrics:
        print(f"  Symmetry Plane:              {metrics['symplane']['barrier']:.2f}%")
    if 'random_midpoint' in metrics:
        print(f"  Random Plane (midpoint):     {metrics['random_midpoint']['barrier']:.2f}%")
    if 'random_random' in metrics:
        print(f"  Random Plane (random):       {metrics['random_random']['barrier']:.2f}%")

    # Comparison stats
    if 'symplane' in metrics and 'random_midpoint' in metrics:
        print("\nDifference (Random Midpoint - Symmetry Plane):")
        err_diff = metrics['random_midpoint']['max_err'] - metrics['symplane']['max_err']
        barrier_diff = metrics['random_midpoint']['barrier'] - metrics['symplane']['barrier']
        print(f"  Max Test Error: {err_diff:+.3f}%")
        print(f"  Barrier Height: {barrier_diff:+.3f}%")

        if abs(barrier_diff) < 0.5:
            print("\n✓ Random plane (midpoint) performs similarly to symmetry plane!")
            print(f"  Difference is only {abs(barrier_diff):.3f}% (negligible)")

    if 'symplane' in metrics and 'random_random' in metrics:
        print("\nDifference (Random Random - Symmetry Plane):")
        err_diff = metrics['random_random']['max_err'] - metrics['symplane']['max_err']
        barrier_diff = metrics['random_random']['barrier'] - metrics['symplane']['barrier']
        print(f"  Max Test Error: {err_diff:+.3f}%")
        print(f"  Barrier Height: {barrier_diff:+.3f}%")

        if abs(barrier_diff) < 0.5:
            print("\n✓ Random plane (random anchor) performs similarly to symmetry plane!")
            print(f"  Difference is only {abs(barrier_diff):.3f}% (negligible)")

    if 'random_midpoint' in metrics and 'random_random' in metrics:
        print("\nDifference (Random Random - Random Midpoint):")
        err_diff = metrics['random_random']['max_err'] - metrics['random_midpoint']['max_err']
        barrier_diff = metrics['random_random']['barrier'] - metrics['random_midpoint']['barrier']
        print(f"  Max Test Error: {err_diff:+.3f}%")
        print(f"  Barrier Height: {barrier_diff:+.3f}%")

    print("=" * 80)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Find first available file to use its directory
        first_file = args.polygon_file or args.symplane_file or args.random_midpoint_file or args.random_random_file
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(first_file)), 'figures/random_planes_comparison.png'
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
    if 'polygon' in metrics:
        summary_lines.append(f"  Polygon Chain:               {metrics['polygon']['max_err']:.2f}%")
    if 'symplane' in metrics:
        summary_lines.append(f"  Symmetry Plane:              {metrics['symplane']['max_err']:.2f}%")
    if 'random_midpoint' in metrics:
        summary_lines.append(f"  Random Plane (midpoint):     {metrics['random_midpoint']['max_err']:.2f}%")
    if 'random_random' in metrics:
        summary_lines.append(f"  Random Plane (random):       {metrics['random_random']['max_err']:.2f}%")
    summary_lines.append("")

    summary_lines.append("Barrier Height:")
    if 'polygon' in metrics:
        summary_lines.append(f"  Polygon Chain:               {metrics['polygon']['barrier']:.2f}%")
    if 'symplane' in metrics:
        summary_lines.append(f"  Symmetry Plane:              {metrics['symplane']['barrier']:.2f}%")
    if 'random_midpoint' in metrics:
        summary_lines.append(f"  Random Plane (midpoint):     {metrics['random_midpoint']['barrier']:.2f}%")
    if 'random_random' in metrics:
        summary_lines.append(f"  Random Plane (random):       {metrics['random_random']['barrier']:.2f}%")
    summary_lines.append("")

    # Write comparison stats
    if 'symplane' in metrics and 'random_midpoint' in metrics:
        barrier_diff = metrics['random_midpoint']['barrier'] - metrics['symplane']['barrier']
        summary_lines.append(f"Difference (Random Midpoint - Symmetry): {barrier_diff:+.3f}%")

    if 'symplane' in metrics and 'random_random' in metrics:
        barrier_diff = metrics['random_random']['barrier'] - metrics['symplane']['barrier']
        summary_lines.append(f"Difference (Random Random - Symmetry): {barrier_diff:+.3f}%")

    if 'random_midpoint' in metrics and 'random_random' in metrics:
        barrier_diff = metrics['random_random']['barrier'] - metrics['random_midpoint']['barrier']
        summary_lines.append(f"Difference (Random Random - Random Midpoint): {barrier_diff:+.3f}%")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    summary_path = output_path.replace('.png', '_summary.txt')
    plotting.save_summary_text(summary_lines, summary_path)

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Random Plane Comparison'
    )

    # Custom arguments
    parser.add_argument('--polygon-file', type=str, default=None,
                       help='Path to unconstrained polygon curve.npz file (optional)')
    parser.add_argument('--symplane-file', type=str, default=None,
                       help='Path to symmetry plane curve.npz file (optional)')
    parser.add_argument('--random-midpoint-file', type=str, default=None,
                       help='Path to random plane (midpoint) curve.npz file (optional)')
    parser.add_argument('--random-random-file', type=str, default=None,
                       help='Path to random plane (random anchor) curve.npz file (optional)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_plot_output_args(parser, required=False)

    args = parser.parse_args()
    plot_comparison(args)
