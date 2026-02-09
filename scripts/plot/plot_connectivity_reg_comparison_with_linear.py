"""
Plot connectivity comparison for seed0-seed1_reg and seed0-mirror_reg configurations.

4 subplots showing both curve and linear interpolation:
- Train Loss
- Test Loss
- Train Error
- Test Error

Solid lines = Bezier curves (seed pairs)
Dashed lines = Bezier curves (mirrored)
Dotted lines = Linear interpolation
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


def load_linear_data(npz_path):
    """Load linear evaluation data from npz file."""
    data = np.load(npz_path)
    return {
        'ts': data['ts'],
        'tr_loss': data['tr_loss'],
        'te_loss': data['te_loss'],
        'tr_err': data['tr_err'],
        'te_err': data['te_err'],
    }


def main():
    parser = argparse.ArgumentParser(description='Plot connectivity comparison with linear interpolation')
    parser.add_argument('--output', type=str, default='plots/connectivity_reg_comparison_with_linear.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Data paths - seeds and bezier curves
    seed_curve_paths = {
        'seed0-seed1': 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/curve.npz',
        'seed0-seed2': 'results/vgg16/cifar10/curves/standard/seed0-seed2_bezier/evaluations/curve.npz',
        'seed1-seed2': 'results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/evaluations/curve.npz',
    }

    # Data paths - mirrored versions (curves)
    mirror_curve_paths = {
        'seed0-mirror': 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz',
    }

    # Data paths - linear interpolations (only for those with linear.npz)
    seed_linear_paths = {
        'seed0-seed1': 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/linear.npz',
    }

    mirror_linear_paths = {
        'seed0-mirror': 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/linear.npz',
    }

    # Load seed curve data
    seed_curves = {}
    for name, path in seed_curve_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            seed_curves[name] = load_curve_data(full_path)
            print(f"Loaded curve {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load mirror curve data
    mirror_curves = {}
    for name, path in mirror_curve_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            mirror_curves[name] = load_curve_data(full_path)
            print(f"Loaded curve {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load seed linear data
    seed_linears = {}
    for name, path in seed_linear_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            seed_linears[name] = load_linear_data(full_path)
            print(f"Loaded linear {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping linear {name}")

    # Load mirror linear data
    mirror_linears = {}
    for name, path in mirror_linear_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            mirror_linears[name] = load_linear_data(full_path)
            print(f"Loaded linear {name} from {path}")
        else:
            print(f"Warning: {full_path} not found, skipping linear {name}")

    if not seed_curves and not mirror_curves:
        print("No data files found!")
        return

    # Colors for seeds
    seed_colors = {
        'seed0-seed1': '#1f77b4',  # Blue
        'seed0-seed2': '#2ca02c',  # Green
        'seed1-seed2': '#9467bd',  # Purple
    }

    # Colors for mirrored
    mirror_colors = {
        'seed0-mirror': '#ff7f0e',  # Orange
        'seed1-mirror': '#d62728',  # Red
    }

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot configuration
    plots = [
        ('tr_loss', 'Train Loss', axes[0, 0]),
        ('te_loss', 'Test Loss', axes[0, 1]),
        ('tr_err', 'Train Error (%)', axes[1, 0]),
        ('te_err', 'Test Error (%)', axes[1, 1]),
    ]

    for key, ylabel, ax in plots:
        # Plot seed curves (solid lines)
        for name, data in seed_curves.items():
            ax.plot(data['ts'], data[key], '-', color=seed_colors[name],
                    linewidth=2, label=f'{name} (curve)')

        # Plot mirror curves (dashed lines)
        for name, data in mirror_curves.items():
            ax.plot(data['ts'], data[key], '--', color=mirror_colors[name],
                    linewidth=2, label=f'{name} (curve)')

        # Plot seed linear (dotted lines, same color but lighter)
        for name, data in seed_linears.items():
            ax.plot(data['ts'], data[key], ':', color=seed_colors[name],
                    linewidth=2, alpha=0.7, label=f'{name} (linear)')

        # Plot mirror linear (dotted lines, same color but lighter)
        for name, data in mirror_linears.items():
            ax.plot(data['ts'], data[key], ':', color=mirror_colors[name],
                    linewidth=2, alpha=0.7, label=f'{name} (linear)')

        ax.set_xlabel('t (interpolation parameter)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Add title
    fig.suptitle('Bezier Curve vs Linear Interpolation Connectivity\n(solid/dashed = Bezier, dotted = linear)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    print("\n" + "=" * 90)
    print("CONNECTIVITY SUMMARY")
    print("=" * 90)
    print(f"{'Config':<15} {'Type':<8} {'Interp':<8} {'Train Loss':>12} {'Test Loss':>12} {'Train Err':>12} {'Test Err':>12}")
    print(f"{'':15} {'':8} {'':8} {'(max)':>12} {'(max)':>12} {'(max)':>12} {'(max)':>12}")
    print("-" * 90)

    for name, data in seed_curves.items():
        print(f"{name:<15} {'seed':<8} {'curve':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")

    for name, data in seed_linears.items():
        print(f"{name:<15} {'seed':<8} {'linear':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")

    for name, data in mirror_curves.items():
        print(f"{name:<15} {'mirror':<8} {'curve':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")

    for name, data in mirror_linears.items():
        print(f"{name:<15} {'mirror':<8} {'linear':<8} {max(data['tr_loss']):>12.4f} {max(data['te_loss']):>12.4f} "
              f"{max(data['tr_err']):>11.2f}% {max(data['te_err']):>11.2f}%")

    print("-" * 90)


if __name__ == '__main__':
    main()
