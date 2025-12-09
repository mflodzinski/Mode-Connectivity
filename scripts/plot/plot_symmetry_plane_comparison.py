"""
Plot comparison between connectivity methods.

Creates a 4-panel plot comparing:
- Direct linear interpolation
- Two-segment symmetry plane path
- (Optional) Bezier curve

Panels show: train loss, test loss, train error, test error
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comparison(args):
    """Create comparison plot."""

    # Load data
    print("Loading comparison data...")
    data = np.load(args.comparison_file)

    t_values = data['t_values']

    # Extract metrics
    metrics = {
        'linear': {
            'train_loss': data['linear_train_loss'],
            'test_loss': data['linear_test_loss'],
            'train_err': data['linear_train_err'],
            'test_err': data['linear_test_err'],
        },
        'symplane': {
            'train_loss': data['symplane_train_loss'],
            'test_loss': data['symplane_test_loss'],
            'train_err': data['symplane_train_err'],
            'test_err': data['symplane_test_err'],
        }
    }

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mode Connectivity: Symmetry Plane vs Linear Interpolation',
                 fontsize=16, fontweight='bold')

    # Plot styles
    styles = {
        'linear': {'color': 'blue', 'linestyle': '--', 'label': 'Linear Interpolation', 'alpha': 0.7},
        'symplane': {'color': 'red', 'linestyle': '-', 'label': 'Symmetry Plane', 'alpha': 0.9, 'linewidth': 2},
    }

    # Panel 1: Train Loss
    ax = axes[0, 0]
    for method in ['linear', 'symplane']:
        ax.plot(t_values, metrics[method]['train_loss'], **styles[method])
    ax.set_xlabel('t (interpolation parameter)', fontsize=11)
    ax.set_ylabel('Train Loss', fontsize=11)
    ax.set_title('Training Loss Along Path', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 2: Test Loss
    ax = axes[0, 1]
    for method in ['linear', 'symplane']:
        ax.plot(t_values, metrics[method]['test_loss'], **styles[method])
    ax.set_xlabel('t (interpolation parameter)', fontsize=11)
    ax.set_ylabel('Test Loss', fontsize=11)
    ax.set_title('Test Loss Along Path', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 3: Train Error
    ax = axes[1, 0]
    for method in ['linear', 'symplane']:
        ax.plot(t_values, metrics[method]['train_err'], **styles[method])
    ax.set_xlabel('t (interpolation parameter)', fontsize=11)
    ax.set_ylabel('Train Error (%)', fontsize=11)
    ax.set_title('Training Error Along Path', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 4: Test Error
    ax = axes[1, 1]
    for method in ['linear', 'symplane']:
        ax.plot(t_values, metrics[method]['test_err'], **styles[method])
    ax.set_xlabel('t (interpolation parameter)', fontsize=11)
    ax.set_ylabel('Test Error (%)', fontsize=11)
    ax.set_title('Test Error Along Path', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add vertical line at t=0.5 (midpoint/theta location)
    for ax in axes.flat:
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1,
                  label='θ* (symmetry plane point)')

    # Print summary statistics
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nMaximum Test Loss:")
    linear_max_loss = np.max(metrics['linear']['test_loss'])
    symplane_max_loss = np.max(metrics['symplane']['test_loss'])
    print(f"  Linear:         {linear_max_loss:.4f}")
    print(f"  Symmetry Plane: {symplane_max_loss:.4f}")
    print(f"  Improvement:    {linear_max_loss - symplane_max_loss:.4f} " +
          f"({(linear_max_loss - symplane_max_loss)/linear_max_loss * 100:.1f}%)")

    print("\nMaximum Test Error:")
    linear_max_err = np.max(metrics['linear']['test_err'])
    symplane_max_err = np.max(metrics['symplane']['test_err'])
    print(f"  Linear:         {linear_max_err:.2f}%")
    print(f"  Symmetry Plane: {symplane_max_err:.2f}%")
    print(f"  Improvement:    {linear_max_err - symplane_max_err:.2f}%")

    print("\nBarrier Height (peak - endpoints average):")
    # Endpoint losses
    linear_endpoint_loss = (metrics['linear']['test_loss'][0] + metrics['linear']['test_loss'][-1]) / 2
    symplane_endpoint_loss = (metrics['symplane']['test_loss'][0] + metrics['symplane']['test_loss'][-1]) / 2

    linear_barrier = linear_max_loss - linear_endpoint_loss
    symplane_barrier = symplane_max_loss - symplane_endpoint_loss

    print(f"  Linear:         {linear_barrier:.4f}")
    print(f"  Symmetry Plane: {symplane_barrier:.4f}")
    print(f"  Reduction:      {linear_barrier - symplane_barrier:.4f} " +
          f"({(linear_barrier - symplane_barrier)/linear_barrier * 100:.1f}%)")

    print("=" * 80)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = args.output if args.output else os.path.join(
        os.path.dirname(args.comparison_file), 'symmetry_plane_comparison.png'
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Symmetry Plane Comparison')

    parser.add_argument('--comparison-file', type=str, required=True,
                       help='Path to comparison.npz file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: same dir as comparison file)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot interactively')

    args = parser.parse_args()
    plot_comparison(args)
