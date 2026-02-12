"""
Plot connectivity comparison for different modes vs permuted modes.

4 subplots showing both curve and linear interpolation:
- Test Error (top left)
- Test Loss (top right)
- Train Error (bottom left)
- Train Loss (bottom right)

Dotted lines = Bezier curves
Solid lines = Linear interpolation
Blue = Different modes (seed pairs)
Orange = Permuted modes (mirrored)
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

    # Data paths - different modes (seed pairs)
    diff_modes_curve = 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/curve.npz'
    diff_modes_linear = 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/linear.npz'

    # Data paths - permuted modes (mirrored) - seed0
    perm0_modes_curve = 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz'
    perm0_modes_linear = 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/linear.npz'

    # Data paths - permuted modes (mirrored) - seed1
    perm1_modes_curve = 'results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/curve.npz'
    perm1_modes_linear = 'results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/linear.npz'

    # Load data
    data = {}

    full_path = os.path.join(project_root, diff_modes_curve)
    if os.path.exists(full_path):
        data['diff_curve'] = load_curve_data(full_path)
        print(f"Loaded different modes curve")
    else:
        print(f"Warning: {full_path} not found")

    full_path = os.path.join(project_root, diff_modes_linear)
    if os.path.exists(full_path):
        data['diff_linear'] = load_linear_data(full_path)
        print(f"Loaded different modes linear")
    else:
        print(f"Warning: {full_path} not found")

    full_path = os.path.join(project_root, perm0_modes_curve)
    if os.path.exists(full_path):
        data['perm0_curve'] = load_curve_data(full_path)
        print(f"Loaded permuted modes (seed0) curve")
    else:
        print(f"Warning: {full_path} not found")

    full_path = os.path.join(project_root, perm0_modes_linear)
    if os.path.exists(full_path):
        data['perm0_linear'] = load_linear_data(full_path)
        print(f"Loaded permuted modes (seed0) linear")
    else:
        print(f"Warning: {full_path} not found")

    full_path = os.path.join(project_root, perm1_modes_curve)
    if os.path.exists(full_path):
        data['perm1_curve'] = load_curve_data(full_path)
        print(f"Loaded permuted modes (seed1) curve")
    else:
        print(f"Warning: {full_path} not found")

    full_path = os.path.join(project_root, perm1_modes_linear)
    if os.path.exists(full_path):
        data['perm1_linear'] = load_linear_data(full_path)
        print(f"Loaded permuted modes (seed1) linear")
    else:
        print(f"Warning: {full_path} not found")

    if not data:
        print("No data files found!")
        return

    # Colors
    diff_color = '#1f77b4'  # Blue for different modes
    perm0_color = '#ff7f0e'  # Orange for permuted modes (seed0)
    perm1_color = '#ffbb78'  # Light orange for permuted modes (seed1)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot configuration: top left to bottom right
    # Test Error, Test Loss, Train Error, Train Loss
    plots = [
        ('te_err', 'Test Error (%)', axes[0, 0]),
        ('te_loss', 'Test Loss', axes[0, 1]),
        ('tr_err', 'Train Error (%)', axes[1, 0]),
        ('tr_loss', 'Train Loss', axes[1, 1]),
    ]

    for idx, (key, title, ax) in enumerate(plots):
        # Plot different modes - curve (dotted)
        if 'diff_curve' in data:
            label = 'Different modes (curve)' if idx == 0 else None
            ax.plot(data['diff_curve']['ts'], data['diff_curve'][key], ':',
                    color=diff_color, linewidth=2, label=label)

        # Plot different modes - linear (solid)
        if 'diff_linear' in data:
            label = 'Different modes (linear)' if idx == 0 else None
            ax.plot(data['diff_linear']['ts'], data['diff_linear'][key], '-',
                    color=diff_color, linewidth=2, label=label)

        # Plot permuted modes seed0 - curve (dotted)
        if 'perm0_curve' in data:
            label = 'Permuted modes (curve)' if idx == 0 else None
            ax.plot(data['perm0_curve']['ts'], data['perm0_curve'][key], ':',
                    color=perm0_color, linewidth=2, label=label)

        # Plot permuted modes seed0 - linear (solid)
        if 'perm0_linear' in data:
            label = 'Permuted modes (linear)' if idx == 0 else None
            ax.plot(data['perm0_linear']['ts'], data['perm0_linear'][key], '-',
                    color=perm0_color, linewidth=2, label=label)

        # Plot permuted modes seed1 - curve (dotted, no label - same category)
        if 'perm1_curve' in data:
            ax.plot(data['perm1_curve']['ts'], data['perm1_curve'][key], ':',
                    color=perm1_color, linewidth=2)

        # Plot permuted modes seed1 - linear (solid, no label - same category)
        if 'perm1_linear' in data:
            ax.plot(data['perm1_linear']['ts'], data['perm1_linear'][key], '-',
                    color=perm1_color, linewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t (interpolation parameter)', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # Add single legend to first subplot
    axes[0, 0].legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    print("\n" + "=" * 80)
    print("CONNECTIVITY SUMMARY")
    print("=" * 80)
    print(f"{'Mode Type':<20} {'Interp':<10} {'Train Loss':>12} {'Test Loss':>12} {'Train Err':>12} {'Test Err':>12}")
    print(f"{'':20} {'':10} {'(max)':>12} {'(max)':>12} {'(max)':>12} {'(max)':>12}")
    print("-" * 80)

    if 'diff_curve' in data:
        d = data['diff_curve']
        print(f"{'Different modes':<20} {'curve':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    if 'diff_linear' in data:
        d = data['diff_linear']
        print(f"{'Different modes':<20} {'linear':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    if 'perm0_curve' in data:
        d = data['perm0_curve']
        print(f"{'Permuted seed0':<20} {'curve':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    if 'perm0_linear' in data:
        d = data['perm0_linear']
        print(f"{'Permuted seed0':<20} {'linear':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    if 'perm1_curve' in data:
        d = data['perm1_curve']
        print(f"{'Permuted seed1':<20} {'curve':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    if 'perm1_linear' in data:
        d = data['perm1_linear']
        print(f"{'Permuted seed1':<20} {'linear':<10} {max(d['tr_loss']):>12.4f} {max(d['te_loss']):>12.4f} "
              f"{max(d['tr_err']):>11.2f}% {max(d['te_err']):>11.2f}%")

    print("-" * 80)


if __name__ == '__main__':
    main()
