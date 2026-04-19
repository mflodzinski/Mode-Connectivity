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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def barrier_tick_label(value, key):
    """Format inset barrier tick labels."""
    if key.endswith('err'):
        return f"{value:.1f}"
    return f"{value:.3f}"


def add_curve_only_inset(ax, key, seed_curves, mirror_curves, seed_colors, mirror_colors):
    """Add a large inset with only the barrier y-ticks shown."""
    inset_ax = ax.inset_axes([0.37, 0.20, 0.58, 0.68])

    for name, curve_data in seed_curves.items():
        inset_ax.plot(
            curve_data['ts'],
            curve_data[key],
            '--',
            color='#1f77b4',
            linewidth=1.5,
        )

    for name, curve_data in mirror_curves.items():
        inset_ax.plot(
            curve_data['ts'],
            curve_data[key],
            ':',
            color='#ff7f0e',
            linewidth=1.5,
        )

    seed_peak = max(float(np.max(curve_data[key])) for curve_data in seed_curves.values())
    mirror_peak = max(float(np.max(curve_data[key])) for curve_data in mirror_curves.values())
    y_min = min(float(np.min(curve_data[key])) for curve_data in list(seed_curves.values()) + list(mirror_curves.values()))
    y_max = max(seed_peak, mirror_peak)
    y_pad = max((y_max - y_min) * 0.08, 1e-4)

    inset_ax.set_xlim(0.0, 1.0)
    inset_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    inset_ax.set_xticks([])
    inset_ax.set_yticks(sorted({seed_peak, mirror_peak}))
    inset_ax.set_yticklabels(
        [barrier_tick_label(value, key) for value in sorted({seed_peak, mirror_peak})],
        fontsize=8,
        fontweight='normal',
    )
    inset_ax.tick_params(axis='y', width=1.0, labelsize=8)
    inset_ax.tick_params(axis='x', width=1.0, labelsize=8)
    inset_ax.grid(True, alpha=0.18)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)


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
    perm1_modes_linear = 'results/vgg16/cifar10/endpoints/standard/seed1_mirrored/evaluations/linear.npz'
    seed_paths = {
        'seed0-seed1': 'results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/curve.npz',
        'seed0-seed2': 'results/vgg16/cifar10/curves/standard/seed0-seed2_bezier/evaluations/curve.npz',
        'seed1-seed2': 'results/vgg16/cifar10/curves/standard/seed1-seed2_bezier/evaluations/curve.npz',
    }
    mirror_paths = {
        'seed0-mirror': 'results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz',
        'seed1-mirror': 'results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/curve.npz',
    }

    # Load data
    data = {}
    seed_curves = {}
    mirror_curves = {}

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

    for name, path in seed_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            seed_curves[name] = load_curve_data(full_path)

    for name, path in mirror_paths.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            mirror_curves[name] = load_curve_data(full_path)

    if not data:
        print("No data files found!")
        return

    # Colors
    diff_color = '#1f77b4'  # Blue for different modes
    perm0_color = '#ff7f0e'  # Orange for permuted modes (seed0)
    perm1_color = '#ffbb78'  # Light orange for permuted modes (seed1)
    seed_colors = {
        'seed0-seed1': '#1f77b4',
        'seed0-seed2': '#2ca02c',
        'seed1-seed2': '#9467bd',
    }
    mirror_colors = {
        'seed0-mirror': '#ff7f0e',
        'seed1-mirror': '#d62728',
    }

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
            label = 'Mirrored modes (curve)' if idx == 0 else None
            ax.plot(data['perm0_curve']['ts'], data['perm0_curve'][key], ':',
                    color=perm0_color, linewidth=2, label=label)

        # Plot permuted modes seed0 - linear (solid)
        if 'perm0_linear' in data:
            label = 'Mirrored modes (linear)' if idx == 0 else None
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

        ax.set_xlabel('t (interpolation parameter)', fontsize=11, fontweight='normal')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=10)
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontweight('normal')
        ax.grid(True, alpha=0.3)

        if seed_curves and mirror_curves:
            add_curve_only_inset(ax, key, seed_curves, mirror_curves, seed_colors, mirror_colors)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.515), ncol=2, frameon=True)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.98, wspace=0.18, hspace=0.28)

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
