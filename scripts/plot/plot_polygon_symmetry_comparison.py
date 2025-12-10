"""
Plot comparison between polygon chain (unconstrained), symmetry plane (constrained), and linear interpolation.

Shows that constraining to symmetry plane doesn't degrade performance compared to unconstrained polygon chain.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comparison(args):
    """Create comparison plot for polygon chain vs symmetry plane vs linear."""

    # Load data
    print("Loading evaluation data...")
    polygon_data = np.load(args.polygon_file)
    symplane_data = np.load(args.symplane_file)
    linear_data = np.load(args.linear_file)

    # Extract t values (should be same for all)
    t_polygon = polygon_data['ts']
    t_symplane = symplane_data['ts']
    t_linear = linear_data['ts']

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Polygon Chain vs Symmetry Plane vs Linear Interpolation',
                 fontsize=16, fontweight='bold')

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
    ax.plot(t_linear, linear_data['te_err'], **styles['linear'])
    ax.plot(t_polygon, polygon_data['te_err'], **styles['polygon'])
    ax.plot(t_symplane, symplane_data['te_err'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Test Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 2: Test Loss
    ax = axes[0, 1]
    ax.plot(t_linear, linear_data['te_loss'], **styles['linear'])
    ax.plot(t_polygon, polygon_data['te_loss'], **styles['polygon'])
    ax.plot(t_symplane, symplane_data['te_loss'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Train Error
    ax = axes[1, 0]
    ax.plot(t_linear, linear_data['tr_err'], **styles['linear'])
    ax.plot(t_polygon, polygon_data['tr_err'], **styles['polygon'])
    ax.plot(t_symplane, symplane_data['tr_err'], **styles['symplane'])
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Train Error (%)', fontsize=12)
    ax.set_title('Train Error Along Path', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.05, 1.05)

    # Panel 4: Train Loss
    ax = axes[1, 1]
    ax.plot(t_linear, linear_data['tr_loss'], **styles['linear'])
    ax.plot(t_polygon, polygon_data['tr_loss'], **styles['polygon'])
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
    print("POLYGON CHAIN VS SYMMETRY PLANE COMPARISON")
    print("=" * 80)

    print("\nMaximum Test Error:")
    linear_max_err = np.max(linear_data['te_err'])
    polygon_max_err = np.max(polygon_data['te_err'])
    symplane_max_err = np.max(symplane_data['te_err'])

    print(f"  Linear:         {linear_max_err:.2f}% at t={t_linear[np.argmax(linear_data['te_err'])]:.3f}")
    print(f"  Polygon Chain:  {polygon_max_err:.2f}% at t={t_polygon[np.argmax(polygon_data['te_err'])]:.3f}")
    print(f"  Symmetry Plane: {symplane_max_err:.2f}% at t={t_symplane[np.argmax(symplane_data['te_err'])]:.3f}")

    print("\nBarrier Height (max - endpoint avg):")
    linear_endpoint_err = (linear_data['te_err'][0] + linear_data['te_err'][-1]) / 2
    polygon_endpoint_err = (polygon_data['te_err'][0] + polygon_data['te_err'][-1]) / 2
    symplane_endpoint_err = (symplane_data['te_err'][0] + symplane_data['te_err'][-1]) / 2

    linear_barrier = linear_max_err - linear_endpoint_err
    polygon_barrier = polygon_max_err - polygon_endpoint_err
    symplane_barrier = symplane_max_err - symplane_endpoint_err

    print(f"  Linear:         {linear_barrier:.2f}%")
    print(f"  Polygon Chain:  {polygon_barrier:.2f}%")
    print(f"  Symmetry Plane: {symplane_barrier:.2f}%")

    print("\nDifference (Symmetry Plane - Polygon Chain):")
    err_diff = symplane_max_err - polygon_max_err
    barrier_diff = symplane_barrier - polygon_barrier
    print(f"  Max Test Error: {err_diff:+.3f}% ({abs(err_diff)/polygon_max_err*100:.2f}% relative)")
    print(f"  Barrier Height: {barrier_diff:+.3f}%")

    if abs(barrier_diff) < 0.5:
        print("\n✓ Symmetry plane constraint does NOT degrade performance!")
        print(f"  Difference is only {abs(barrier_diff):.3f}% (negligible)")
    else:
        print(f"\n  Performance difference: {barrier_diff:.2f}%")

    print("\nBarrier Reduction vs Linear:")
    linear_reduction_polygon = (linear_barrier - polygon_barrier) / linear_barrier * 100
    linear_reduction_symplane = (linear_barrier - symplane_barrier) / linear_barrier * 100
    print(f"  Polygon Chain:  {linear_reduction_polygon:.1f}%")
    print(f"  Symmetry Plane: {linear_reduction_symplane:.1f}%")

    print("=" * 80)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = args.output if args.output else os.path.join(
        os.path.dirname(args.polygon_file), '../figures/polygon_symmetry_comparison.png'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Save summary to text file
    summary_path = output_path.replace('.png', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POLYGON CHAIN VS SYMMETRY PLANE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Maximum Test Error:\n")
        f.write(f"  Linear:         {linear_max_err:.2f}%\n")
        f.write(f"  Polygon Chain:  {polygon_max_err:.2f}%\n")
        f.write(f"  Symmetry Plane: {symplane_max_err:.2f}%\n\n")
        f.write(f"Barrier Height:\n")
        f.write(f"  Linear:         {linear_barrier:.2f}%\n")
        f.write(f"  Polygon Chain:  {polygon_barrier:.2f}%\n")
        f.write(f"  Symmetry Plane: {symplane_barrier:.2f}%\n\n")
        f.write(f"Difference (Symmetry - Polygon): {barrier_diff:+.3f}%\n")
        f.write(f"Barrier Reduction vs Linear:\n")
        f.write(f"  Polygon Chain:  {linear_reduction_polygon:.1f}%\n")
        f.write(f"  Symmetry Plane: {linear_reduction_symplane:.1f}%\n")
        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {summary_path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Polygon Chain vs Symmetry Plane Comparison'
    )

    parser.add_argument('--polygon-file', type=str, required=True,
                       help='Path to polygon chain curve.npz file')
    parser.add_argument('--symplane-file', type=str, required=True,
                       help='Path to symmetry plane curve.npz file')
    parser.add_argument('--linear-file', type=str, required=True,
                       help='Path to linear.npz file (from either experiment)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: auto-generated)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot interactively')

    args = parser.parse_args()
    plot_comparison(args)
