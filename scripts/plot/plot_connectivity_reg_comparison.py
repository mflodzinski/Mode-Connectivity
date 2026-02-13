"""
Plot connectivity comparison for different modes vs permuted modes.

4 subplots showing Bezier curve interpolation only (no linear):
- Test Error (top left)
- Test Loss (top right)
- Train Error (bottom left)
- Train Loss (bottom right)

Solid lines = Different modes (seed pairs)
Dashed lines = Permuted modes (mirrored)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)


def load_curve_data(npz_path):
    """Load curve evaluation data from npz file."""
    data = np.load(npz_path)
    return {
        'ts': data['ts'],
        'tr_loss': data['tr_loss'],
        'te_loss': data['te_loss'],
        'tr_err': data['tr_err'],
        'te_err': data['te_err'],
    }


def main():
    parser = argparse.ArgumentParser(description='Plot connectivity comparison for reg configurations')
    parser.add_argument('--output', type=str, default='plots/connectivity_reg_comparison.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Data paths - seeds and bezier curves
    seed_paths = {
        'seed0-seed1': 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/curve.npz',
        'seed0-seed2': 'results/vgg16/cifar10/curves/standard/seed0-seed2_bezier/evaluations/curve.npz',
        'seed1-seed2': 'results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/evaluations/curve.npz',
    }

    # Data paths - mirrored versions
    mirror_paths = {
        'seed0-mirror': 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz',
        'seed1-mirror': 'results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/curve.npz',
    }

    # Load seed data
    seed_curves = {}
    for name, path in seed_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            seed_curves[name] = load_curve_data(full_path)
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load mirror data
    mirror_curves = {}
    for name, path in mirror_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            mirror_curves[name] = load_curve_data(full_path)
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    if not seed_curves and not mirror_curves:
        print("No data files found!")
        return

    # Colors for seeds (solid lines)
    seed_colors = {
        'seed0-seed1': '#1f77b4',  # Blue
        'seed0-seed2': '#2ca02c',  # Green
        'seed1-seed2': '#9467bd',  # Purple
    }

    # Colors for mirrored (dashed lines)
    mirror_colors = {
        'seed0-mirror': '#ff7f0e',  # Orange
        'seed1-mirror': '#d62728',  # Red
    }

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot configuration: top left to bottom right
    # Test Error, Test Loss, Train Error, Train Loss
    plots = [
        ('te_err', 'Test Error (%)', axes[0, 0]),
        ('te_loss', 'Test Loss', axes[0, 1]),
        ('tr_err', 'Train Error (%)', axes[1, 0]),
        ('tr_loss', 'Train Loss', axes[1, 1]),
    ]

    for idx, (key, title, ax) in enumerate(plots):
        # Plot seed curves (solid lines) - different modes
        for name, data in seed_curves.items():
            ax.plot(data['ts'], data[key], '-', color=seed_colors[name],
                    linewidth=2)

        # Plot mirror curves (dashed lines) - permuted modes
        for name, data in mirror_curves.items():
            ax.plot(data['ts'], data[key], '--', color=mirror_colors[name],
                    linewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t (interpolation parameter)', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # Add custom legend with grey lines to first subplot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='-', linewidth=2, label='Different modes (solid)'),
        Line2D([0], [0], color='grey', linestyle='--', linewidth=2, label='Permuted modes (dashed)'),
    ]
    axes[0, 0].legend(handles=legend_elements, fontsize=9, loc='best')

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    print("\n" + "=" * 75)
    print("CONNECTIVITY SUMMARY")
    print("=" * 75)
    print(f"{'Config':<15} {'Type':<8} {'Train Loss':>12} {'Test Loss':>12} {'Train Err':>12} {'Test Err':>12}")
    print(f"{'':15} {'':8} {'(max)':>12} {'(max)':>12} {'(max)':>12} {'(max)':>12}")
    print("-" * 75)
    for name, data in seed_curves.items():
        print(f"{name:<15} {'seed':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")
    for name, data in mirror_curves.items():
        print(f"{name:<15} {'mirror':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")
    print("-" * 75)


if __name__ == '__main__':
    main()
