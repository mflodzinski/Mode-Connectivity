"""
Plot comparison between biased linear initialization experiments (alpha=0.9 vs alpha=0.75).

Shows whether different biased initialization strategies affect mode connectivity.
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
    """Create comparison plot for different biased initialization strategies."""

    # Load data
    print("Loading evaluation data...")
    alpha09_data = np.load(args.alpha09_file)
    alpha075_data = np.load(args.alpha075_file)

    # Optionally load standard midpoint initialization (alpha=0.5)
    standard_data = np.load(args.standard_file) if args.standard_file else None

    # Extract t values
    t_alpha09 = alpha09_data['ts']
    t_alpha075 = alpha075_data['ts']
    t_standard = standard_data['ts'] if standard_data is not None else None

    # Create title
    title = 'Biased Linear Initialization Comparison'

    # Create figure with 6 subplots (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot styles
    styles = {
        'alpha09': {'color': '#e74c3c', 'linestyle': '-', 'label': r'Biased $\alpha=0.9$ (close to endpoint)',
                    'alpha': 0.9, 'linewidth': 2.5, 'marker': 'o', 'markersize': 3, 'markevery': 5},
        'alpha075': {'color': '#3498db', 'linestyle': '-', 'label': r'Biased $\alpha=0.75$',
                     'alpha': 0.9, 'linewidth': 2.5, 'marker': 's', 'markersize': 3, 'markevery': 5},
        'standard': {'color': '#95a5a6', 'linestyle': '--', 'label': r'Standard $\alpha=0.5$ (midpoint)',
                     'alpha': 0.7, 'linewidth': 2.0, 'marker': '^', 'markersize': 3, 'markevery': 5},
    }

    # Panel 1: Test Error
    ax = axes[0, 0]
    ax.plot(t_alpha09, alpha09_data['te_err'], **styles['alpha09'])
    ax.plot(t_alpha075, alpha075_data['te_err'], **styles['alpha075'])
    if standard_data is not None:
        ax.plot(t_standard, standard_data['te_err'], **styles['standard'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Test Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 2: Test Loss
    ax = axes[0, 1]
    ax.plot(t_alpha09, alpha09_data['te_loss'], **styles['alpha09'])
    ax.plot(t_alpha075, alpha075_data['te_loss'], **styles['alpha075'])
    if standard_data is not None:
        ax.plot(t_standard, standard_data['te_loss'], **styles['standard'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Train Error
    ax = axes[1, 0]
    ax.plot(t_alpha09, alpha09_data['tr_err'], **styles['alpha09'])
    ax.plot(t_alpha075, alpha075_data['tr_err'], **styles['alpha075'])
    if standard_data is not None:
        ax.plot(t_standard, standard_data['tr_err'], **styles['standard'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Error (%)', fontsize=12)
    ax.set_title('Train Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 4: Train Loss
    ax = axes[1, 1]
    ax.plot(t_alpha09, alpha09_data['tr_loss'], **styles['alpha09'])
    ax.plot(t_alpha075, alpha075_data['tr_loss'], **styles['alpha075'])
    if standard_data is not None:
        ax.plot(t_standard, standard_data['tr_loss'], **styles['standard'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Train Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 5: L2 Norm
    ax = axes[2, 0]
    ax.plot(t_alpha09, alpha09_data['l2_norm'], **styles['alpha09'])
    ax.plot(t_alpha075, alpha075_data['l2_norm'], **styles['alpha075'])
    if standard_data is not None:
        ax.plot(t_standard, standard_data['l2_norm'], **styles['standard'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title('L2 Norm Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 6: Blank (reserved for future use)
    ax = axes[2, 1]
    ax.axis('off')  # Hide the axes to make it blank

    # Add vertical line at t=0.5 (middle bend location)
    for ax in axes.flat:
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Print summary statistics
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

    # Calculate metrics
    alpha09_max_err = np.max(alpha09_data['te_err'])
    alpha09_endpoint_err = (alpha09_data['te_err'][0] + alpha09_data['te_err'][-1]) / 2
    alpha09_barrier = alpha09_max_err - alpha09_endpoint_err

    alpha075_max_err = np.max(alpha075_data['te_err'])
    alpha075_endpoint_err = (alpha075_data['te_err'][0] + alpha075_data['te_err'][-1]) / 2
    alpha075_barrier = alpha075_max_err - alpha075_endpoint_err

    print("\nMaximum Test Error:")
    print(f"  α=0.9:  {alpha09_max_err:.2f}%")
    print(f"  α=0.75: {alpha075_max_err:.2f}%")

    if standard_data is not None:
        standard_max_err = np.max(standard_data['te_err'])
        standard_endpoint_err = (standard_data['te_err'][0] + standard_data['te_err'][-1]) / 2
        standard_barrier = standard_max_err - standard_endpoint_err
        print(f"  α=0.5:  {standard_max_err:.2f}% (standard midpoint)")

    print("\nBarrier Height (max - endpoint avg):")
    print(f"  α=0.9:  {alpha09_barrier:.2f}%")
    print(f"  α=0.75: {alpha075_barrier:.2f}%")

    if standard_data is not None:
        print(f"  α=0.5:  {standard_barrier:.2f}% (standard midpoint)")

    # Comparison
    print("\nDifference (α=0.9 - α=0.75):")
    err_diff = alpha09_max_err - alpha075_max_err
    barrier_diff = alpha09_barrier - alpha075_barrier
    print(f"  Max Test Error: {err_diff:+.3f}%")
    print(f"  Barrier Height: {barrier_diff:+.3f}%")

    if abs(barrier_diff) < 0.5:
        print(f"\n✓ Both biased initializations perform similarly!")
        print(f"  Difference is only {abs(barrier_diff):.3f}% (negligible)")

    if standard_data is not None:
        print("\nDifference from standard midpoint (α=0.5):")
        print(f"  α=0.9 - α=0.5:  {(alpha09_barrier - standard_barrier):+.3f}%")
        print(f"  α=0.75 - α=0.5: {(alpha075_barrier - standard_barrier):+.3f}%")

    print("=" * 80)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(args.alpha09_file)), 'figures/biased_init_comparison.png'
        )

    # Save figure
    plotting.save_figure(fig, output_path)

    # Save summary to text file
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(title.upper())
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("Maximum Test Error:")
    summary_lines.append(f"  α=0.9:  {alpha09_max_err:.2f}%")
    summary_lines.append(f"  α=0.75: {alpha075_max_err:.2f}%")
    if standard_data is not None:
        summary_lines.append(f"  α=0.5:  {standard_max_err:.2f}% (standard)")
    summary_lines.append("")
    summary_lines.append("Barrier Height:")
    summary_lines.append(f"  α=0.9:  {alpha09_barrier:.2f}%")
    summary_lines.append(f"  α=0.75: {alpha075_barrier:.2f}%")
    if standard_data is not None:
        summary_lines.append(f"  α=0.5:  {standard_barrier:.2f}% (standard)")
    summary_lines.append("")
    summary_lines.append(f"Difference (α=0.9 - α=0.75): {barrier_diff:+.3f}%")
    if standard_data is not None:
        summary_lines.append(f"Difference (α=0.9 - α=0.5):  {(alpha09_barrier - standard_barrier):+.3f}%")
        summary_lines.append(f"Difference (α=0.75 - α=0.5): {(alpha075_barrier - standard_barrier):+.3f}%")
    summary_lines.append("")
    summary_lines.append("=" * 80)

    summary_path = output_path.replace('.png', '_summary.txt')
    plotting.save_summary_text(summary_lines, summary_path)

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Biased Linear Initialization Comparison'
    )

    # Custom arguments
    parser.add_argument('--alpha09-file', type=str, required=True,
                       help='Path to alpha=0.9 curve.npz file')
    parser.add_argument('--alpha075-file', type=str, required=True,
                       help='Path to alpha=0.75 curve.npz file')
    parser.add_argument('--standard-file', type=str, default=None,
                       help='Path to standard (alpha=0.5) curve.npz file (optional)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_plot_output_args(parser, required=False)

    args = parser.parse_args()
    plot_comparison(args)
