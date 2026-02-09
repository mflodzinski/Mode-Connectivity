"""
Plot Train Loss: Original vs Recovered after alignment.

Compares maximum train loss along interpolation path before and after
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
    parser = argparse.ArgumentParser(description='Plot train loss: original vs recovered')
    parser.add_argument('--output', type=str, default='plots/alignment_train_loss.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Data paths for LMC split experiments
    data_files = {
        '150/50': 'results/analysis/alignment_benchmark/results.json',
        '80/120': 'results/analysis/alignment_benchmark_80split/results.json',
        '30/170': 'results/analysis/alignment_benchmark_30split/results.json',
        '8/192': 'results/analysis/alignment_benchmark_8split/results.json',
        '0/200': 'results/analysis/alignment_benchmark_0split/results.json',
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
                'org_train_loss_max': max(data['original_barrier']['train_loss']),
                'rec_train_loss_max': max(data['recovered_barrier_to_w0']['train_loss']),
                'is_independent': False
            })
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    # Load seed0-seed1 (independent - different random inits) data
    seed_path = os.path.join(project_root, seed_file)
    if os.path.exists(seed_path):
        data = load_benchmark_data(seed_path)
        experiments.append({
            'name': 'seed0-seed1',
            'w0_w1': data['before_alignment']['distance']['l2_distance'],
            'w0_w1_recovered': data['after_alignment']['distance']['l2_distance'],
            'org_train_loss_max': max(data['before_alignment']['barrier']['train_loss']),
            'rec_train_loss_max': max(data['after_alignment']['barrier']['train_loss']),
            'is_independent': True,
            'is_same_init': False
        })
    else:
        print(f"Warning: {seed_path} not found, skipping seed0-seed1")

    # Add 0/200 independent experiment (same random init, different training seeds)
    # Load from file if available
    ind_0split_path = os.path.join(project_root, 'results/analysis/alignment_0split_independent/results.json')
    if os.path.exists(ind_0split_path):
        data = load_benchmark_data(ind_0split_path)
        experiments.append({
            'name': '0/200-ind',
            'w0_w1': data['before_alignment']['distance']['l2_distance'],
            'w0_w1_recovered': data['after_alignment']['distance']['l2_distance'],
            'org_train_loss_max': max(data['before_alignment']['barrier']['train_loss']),
            'rec_train_loss_max': max(data['after_alignment']['barrier']['train_loss']),
            'is_independent': True,
            'is_same_init': True
        })
    else:
        print(f"Warning: {ind_0split_path} not found, skipping 0/200-ind")

    if not experiments:
        print("No data files found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    org_color = '#1f77b4'  # Blue for original
    rec_color = '#2ca02c'  # Green for recovered
    ind_diff_init_color = '#d62728'  # Red for independent (different init)
    ind_same_init_color = '#ff7f0e'  # Orange for independent (same init)

    # Separate LMC and independent experiments
    lmc_exps = [e for e in experiments if not e.get('is_independent', False)]
    ind_diff_init_exps = [e for e in experiments if e.get('is_independent', False) and not e.get('is_same_init', False)]
    ind_same_init_exps = [e for e in experiments if e.get('is_independent', False) and e.get('is_same_init', False)]

    # Plot LMC experiments - Original
    if lmc_exps:
        x_org = [e['w0_w1'] for e in lmc_exps]
        y_org = [e['org_train_loss_max'] for e in lmc_exps]
        ax.scatter(x_org, y_org, c=org_color, marker='o', s=100,
                   label='Original (LMC)', zorder=5)

        # Plot LMC experiments - Recovered
        x_rec = [e['w0_w1_recovered'] for e in lmc_exps]
        y_rec = [e['rec_train_loss_max'] for e in lmc_exps]
        ax.scatter(x_rec, y_rec, c=rec_color, marker='o', s=100,
                   label='Recovered (LMC)', zorder=5)

        # Add labels and arrows connecting original to recovered
        for e in lmc_exps:
            # Label original point
            ax.annotate(e['name'], (e['w0_w1'], e['org_train_loss_max']),
                        textcoords="offset points", xytext=(-5, 8), fontsize=8,
                        color=org_color)
            # Draw arrow from original to recovered
            ax.annotate('', xy=(e['w0_w1_recovered'], e['rec_train_loss_max']),
                        xytext=(e['w0_w1'], e['org_train_loss_max']),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

    # Plot independent experiments (different init) - only recovered point
    if ind_diff_init_exps:
        for e in ind_diff_init_exps:
            ax.scatter([e['w0_w1_recovered']], [e['rec_train_loss_max']], c=ind_diff_init_color, marker='D', s=150,
                       edgecolors='black', linewidths=1.5,
                       label='Recovered (diff init)', zorder=6)
            ax.annotate(e['name'], (e['w0_w1_recovered'], e['rec_train_loss_max']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8,
                        fontweight='bold', color=ind_diff_init_color)

    # Plot independent experiments (same init) - only recovered point
    if ind_same_init_exps:
        for e in ind_same_init_exps:
            ax.scatter([e['w0_w1_recovered']], [e['rec_train_loss_max']], c=ind_same_init_color, marker='^', s=150,
                       edgecolors='black', linewidths=1.5,
                       label='Recovered (same init)', zorder=6)
            ax.annotate(e['name'], (e['w0_w1_recovered'], e['rec_train_loss_max']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8,
                        fontweight='bold', color=ind_same_init_color)

    # Styling
    ax.set_xlabel('L2 Distance', fontsize=12)
    ax.set_ylabel('Maximum Train Loss', fontsize=12)
    ax.set_title('Train Loss: Original vs Recovered\n(Before and After Permutation Alignment)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    print("\n" + "=" * 85)
    print("TRAIN LOSS SUMMARY")
    print("=" * 85)
    print(f"{'Experiment':<15} {'w0_w1':>10} {'Org Loss':>12} {'w0_w1_rec':>10} {'Rec Loss':>12} {'Δ Loss':>12}")
    print("-" * 85)
    for e in experiments:
        if e.get('is_independent', False):
            marker = " **" if e.get('is_same_init', False) else " *"
        else:
            marker = ""
        delta = e['rec_train_loss_max'] - e['org_train_loss_max']
        print(f"{e['name']:<15} {e['w0_w1']:>10.2f} {e['org_train_loss_max']:>12.4f} {e['w0_w1_recovered']:>10.2f} {e['rec_train_loss_max']:>12.4f} {delta:>+11.4f}{marker}")
    print("-" * 85)
    print("* = independent (different random init)")
    print("** = independent (same random init, different training seeds)")
    print("Δ Loss = recovered - original (negative = improvement)")


if __name__ == '__main__':
    main()
