"""
Plot Train Accuracy: Original vs Recovered after alignment.

Compares minimum train accuracy along interpolation path before and after
permutation alignment for different split configurations.
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
    parser = argparse.ArgumentParser(description='Plot train accuracy: original vs recovered')
    parser.add_argument('--output', type=str, default='plots/alignment_train_accuracy.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Data paths for LMC split experiments
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
            experiments.append({
                'name': name,
                'w0_w1': data['distances']['w0_w1']['l2_distance'],
                'w0_w1_recovered': data['distances']['w0_w1_recovered']['l2_distance'],
                'org_train_min': min(data['original_barrier']['train_acc']),
                'rec_train_min': min(data['recovered_barrier_to_w0']['train_acc']),
                'is_independent': False
            })
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load seed0-seed1 (independent) data
    seed_path = os.path.join(project_root, seed_file)
    if os.path.exists(seed_path):
        data = load_benchmark_data(seed_path)
        experiments.append({
            'name': 'seed0-seed1',
            'w0_w1': data['before_alignment']['distance']['l2_distance'],
            'w0_w1_recovered': data['after_alignment']['distance']['l2_distance'],
            'org_train_min': min(data['before_alignment']['barrier']['train_acc']),
            'rec_train_min': min(data['after_alignment']['barrier']['train_acc']),
            'is_independent': True
        })
    else:
        print(f"Warning: {seed_path} not found, skipping seed0-seed1")

    if not experiments:
        print("No data files found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    org_color = '#1f77b4'  # Blue for original
    rec_color = '#2ca02c'  # Green for recovered

    # Separate LMC and independent experiments
    lmc_exps = [e for e in experiments if not e['is_independent']]
    ind_exps = [e for e in experiments if e['is_independent']]

    # Plot LMC experiments - Original
    if lmc_exps:
        x_org = [e['w0_w1'] for e in lmc_exps]
        y_org = [e['org_train_min'] for e in lmc_exps]
        ax.scatter(x_org, y_org, c=org_color, marker='o', s=100,
                   label='Original (LMC)', zorder=5)

        # Plot LMC experiments - Recovered
        x_rec = [e['w0_w1_recovered'] for e in lmc_exps]
        y_rec = [e['rec_train_min'] for e in lmc_exps]
        ax.scatter(x_rec, y_rec, c=rec_color, marker='o', s=100,
                   label='Recovered (LMC)', zorder=5)

        # Add labels and arrows connecting original to recovered
        for e in lmc_exps:
            # Label original point
            ax.annotate(e['name'], (e['w0_w1'], e['org_train_min']),
                        textcoords="offset points", xytext=(-5, 8), fontsize=8,
                        color=org_color)
            # Draw arrow from original to recovered
            ax.annotate('', xy=(e['w0_w1_recovered'], e['rec_train_min']),
                        xytext=(e['w0_w1'], e['org_train_min']),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

    # Plot independent experiment - highlighted differently
    if ind_exps:
        for e in ind_exps:
            # Original
            ax.scatter([e['w0_w1']], [e['org_train_min']], c=org_color, marker='D', s=150,
                       edgecolors='black', linewidths=1.5,
                       label='Original (independent)', zorder=6)
            # Recovered
            ax.scatter([e['w0_w1_recovered']], [e['rec_train_min']], c=rec_color, marker='D', s=150,
                       edgecolors='black', linewidths=1.5,
                       label='Recovered (independent)', zorder=6)
            # Label
            ax.annotate(e['name'], (e['w0_w1'], e['org_train_min']),
                        textcoords="offset points", xytext=(-5, 8), fontsize=8,
                        fontweight='bold', color=org_color)
            # Arrow
            ax.annotate('', xy=(e['w0_w1_recovered'], e['rec_train_min']),
                        xytext=(e['w0_w1'], e['org_train_min']),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=1.5))

    # Styling
    ax.set_xlabel('L2 Distance', fontsize=12)
    ax.set_ylabel('Minimum Train Accuracy (%)', fontsize=12)
    ax.set_title('Train Accuracy: Original vs Recovered\n(Before and After Permutation Alignment)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
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
    print("\n" + "=" * 80)
    print("TRAIN ACCURACY SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<15} {'w0_w1':>10} {'Org Train':>12} {'w0_w1_rec':>10} {'Rec Train':>12} {'Δ Acc':>10}")
    print("-" * 80)
    for e in experiments:
        marker = " *" if e['is_independent'] else ""
        delta = e['rec_train_min'] - e['org_train_min']
        print(f"{e['name']:<15} {e['w0_w1']:>10.2f} {e['org_train_min']:>11.2f}% {e['w0_w1_recovered']:>10.2f} {e['rec_train_min']:>11.2f}% {delta:>+9.2f}%{marker}")
    print("-" * 80)
    print("* = independently trained (no shared initialization)")
    print("Δ Acc = recovered - original (positive = improvement)")


if __name__ == '__main__':
    main()
