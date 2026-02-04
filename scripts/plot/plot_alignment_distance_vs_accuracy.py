"""
Plot L2 distance vs minimum accuracy along interpolation path.

Compares different split configurations (150/50, 80/120, 30/170, 8/192)
and independently trained models (seed0-seed1).
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)


def load_benchmark_data(json_path):
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Plot L2 distance vs accuracy')
    parser.add_argument('--output', type=str, default='plots/alignment_distance_vs_accuracy.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Data paths
    data_files = {
        '150/50': 'results/analysis/alignment_benchmark/results.json',
        '80/120': 'results/analysis/alignment_benchmark_80split/results.json',
        '30/170': 'results/analysis/alignment_benchmark_30split/results.json',
        '8/192': 'results/analysis/alignment_benchmark_8split/results.json',
    }
    seed_file = 'results/analysis/alignment_independent/seed0_seed1_results.json'

    # Collect data
    experiments = []

    for name, path in data_files.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            data = load_benchmark_data(full_path)
            l2_dist = data['distances']['w0_w1']['l2_distance']
            train_acc_min = min(data['original_barrier']['train_acc'])
            test_acc_min = min(data['original_barrier']['test_acc'])
            experiments.append({
                'name': name,
                'l2_distance': l2_dist,
                'train_acc_min': train_acc_min,
                'test_acc_min': test_acc_min,
                'is_independent': False
            })
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load seed0-seed1 (independent) data - use AFTER alignment values
    seed_path = os.path.join(project_root, seed_file)
    if os.path.exists(seed_path):
        data = load_benchmark_data(seed_path)
        # For seed0-seed1, use the recovered distance and accuracy (after alignment)
        l2_dist = data['after_alignment']['distance']['l2_distance']
        train_acc_min = min(data['after_alignment']['barrier']['train_acc'])
        test_acc_min = min(data['after_alignment']['barrier']['test_acc'])
        experiments.append({
            'name': 'seed0-seed1',
            'l2_distance': l2_dist,
            'train_acc_min': train_acc_min,
            'test_acc_min': test_acc_min,
            'is_independent': True
        })
    else:
        print(f"Warning: {seed_path} not found, skipping seed0-seed1")

    if not experiments:
        print("No data files found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors and markers
    lmc_train_color = '#1f77b4'
    lmc_test_color = '#ff7f0e'
    ind_train_color = '#d62728'
    ind_test_color = '#9467bd'

    # Plot LMC-connected experiments
    lmc_exps = [e for e in experiments if not e['is_independent']]
    ind_exps = [e for e in experiments if e['is_independent']]

    # LMC train accuracy
    if lmc_exps:
        x_lmc = [e['l2_distance'] for e in lmc_exps]
        y_train_lmc = [e['train_acc_min'] for e in lmc_exps]
        y_test_lmc = [e['test_acc_min'] for e in lmc_exps]

        ax.scatter(x_lmc, y_train_lmc, c=lmc_train_color, marker='o', s=100,
                   label='Train (LMC splits)', zorder=5)
        ax.scatter(x_lmc, y_test_lmc, c=lmc_test_color, marker='s', s=100,
                   label='Test (LMC splits)', zorder=5)

        # Add labels
        for e in lmc_exps:
            ax.annotate(e['name'], (e['l2_distance'], e['train_acc_min']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
            ax.annotate(e['name'], (e['l2_distance'], e['test_acc_min']),
                        textcoords="offset points", xytext=(5, -10), fontsize=9)

    # Independent (seed0-seed1) - highlighted differently
    if ind_exps:
        x_ind = [e['l2_distance'] for e in ind_exps]
        y_train_ind = [e['train_acc_min'] for e in ind_exps]
        y_test_ind = [e['test_acc_min'] for e in ind_exps]

        ax.scatter(x_ind, y_train_ind, c=ind_train_color, marker='D', s=150,
                   edgecolors='black', linewidths=1.5,
                   label='Train (independent)', zorder=6)
        ax.scatter(x_ind, y_test_ind, c=ind_test_color, marker='D', s=150,
                   edgecolors='black', linewidths=1.5,
                   label='Test (independent)', zorder=6)

        # Add labels
        for e in ind_exps:
            ax.annotate(e['name'], (e['l2_distance'], e['train_acc_min']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight='bold')
            ax.annotate(e['name'], (e['l2_distance'], e['test_acc_min']),
                        textcoords="offset points", xytext=(5, -10), fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlabel('L2 Distance (w₀ ↔ w₁)', fontsize=12)
    ax.set_ylabel('Minimum Accuracy Along Interpolation Path (%)', fontsize=12)
    ax.set_title('L2 Distance vs Minimum Accuracy\n(Original Interpolation Before Alignment)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower left')

    # Set y-axis to show full range
    ax.set_ylim(0, 105)

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<15} {'L2 Dist':>10} {'Train Min':>12} {'Test Min':>12}")
    print("-" * 60)
    for e in experiments:
        marker = " *" if e['is_independent'] else ""
        print(f"{e['name']:<15} {e['l2_distance']:>10.2f} {e['train_acc_min']:>11.2f}% {e['test_acc_min']:>11.2f}%{marker}")
    print("-" * 60)
    print("* = independently trained (no shared initialization)")


if __name__ == '__main__':
    main()
