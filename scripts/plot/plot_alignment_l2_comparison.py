"""
Plot L2 distance comparison: original vs recovered distances.

X-axis: w0_w1 L2 distance (original distance between modes)
Y-axis: w0_w1_recovered and w1_w1_recovered L2 distances

Shows how alignment affects distances for different split configurations.
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
    parser = argparse.ArgumentParser(description='Plot L2 distance comparison')
    parser.add_argument('--output', type=str, default='plots/alignment_l2_comparison.png',
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
            experiments.append({
                'name': name,
                'w0_w1': data['distances']['w0_w1']['l2_distance'],
                'w0_w1_recovered': data['distances']['w0_w1_recovered']['l2_distance'],
                'w1_w1_recovered': data['distances']['w1_w1_recovered']['l2_distance'],
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
            'w1_w1_recovered': None,  # Not applicable for independent models
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
    w0_w1_rec_color = '#2ca02c'  # Green
    w1_w1_rec_color = '#ff7f0e'  # Orange
    ind_color = '#d62728'        # Red

    # Separate LMC and independent experiments
    lmc_exps = [e for e in experiments if not e['is_independent']]
    ind_exps = [e for e in experiments if e['is_independent']]

    # Plot LMC experiments
    if lmc_exps:
        x_lmc = [e['w0_w1'] for e in lmc_exps]
        y_w0_rec = [e['w0_w1_recovered'] for e in lmc_exps]
        y_w1_rec = [e['w1_w1_recovered'] for e in lmc_exps]

        # Plot with connecting lines
        ax.plot(x_lmc, y_w0_rec, 'o-', color=w0_w1_rec_color, markersize=10,
                linewidth=2, label='w₀ ↔ w₁_recovered (LMC)', zorder=5)
        ax.plot(x_lmc, y_w1_rec, 's-', color=w1_w1_rec_color, markersize=10,
                linewidth=2, label='w₁ ↔ w₁_recovered (LMC)', zorder=5)

        # Add labels
        for e in lmc_exps:
            ax.annotate(e['name'], (e['w0_w1'], e['w0_w1_recovered']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Plot independent experiment - highlighted differently
    if ind_exps:
        for e in ind_exps:
            ax.scatter([e['w0_w1']], [e['w0_w1_recovered']], c=ind_color, marker='D', s=200,
                       edgecolors='black', linewidths=2,
                       label='w₀ ↔ w₁_aligned (independent)', zorder=6)
            ax.annotate(e['name'], (e['w0_w1'], e['w0_w1_recovered']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight='bold')

    # Reference lines
    # Diagonal y=x line (no improvement)
    all_x = [e['w0_w1'] for e in experiments]
    x_range = np.linspace(min(all_x) * 0.9, max(all_x) * 1.1, 100)
    ax.plot(x_range, x_range, '--', color='gray', alpha=0.5, linewidth=1.5,
            label='y = x (no change)')

    # Horizontal line at y=0 (perfect recovery for w1_w1_recovered)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5,
               label='Perfect recovery (y=0)')

    # Styling
    ax.set_xlabel('Original L2 Distance (w₀ ↔ w₁)', fontsize=12)
    ax.set_ylabel('L2 Distance After Alignment', fontsize=12)
    ax.set_title('L2 Distance Recovery After Permutation Alignment',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')

    # Set axis limits
    ax.set_xlim(min(all_x) * 0.8, max(all_x) * 1.1)
    y_max = max([e['w0_w1_recovered'] for e in experiments] +
                [e['w1_w1_recovered'] for e in experiments if e['w1_w1_recovered'] is not None])
    ax.set_ylim(-2, y_max * 1.1)

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
    print("L2 DISTANCE SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<15} {'w0↔w1':>12} {'w0↔w1_rec':>12} {'w1↔w1_rec':>12} {'Δ(w0↔w1)':>12}")
    print("-" * 80)
    for e in experiments:
        marker = " *" if e['is_independent'] else ""
        w1_rec_str = f"{e['w1_w1_recovered']:>11.2f}" if e['w1_w1_recovered'] is not None else "        N/A"
        delta = e['w0_w1'] - e['w0_w1_recovered']
        print(f"{e['name']:<15} {e['w0_w1']:>12.2f} {e['w0_w1_recovered']:>12.2f} {w1_rec_str} {delta:>+11.2f}{marker}")
    print("-" * 80)
    print("* = independently trained (no shared initialization)")
    print("Δ(w0↔w1) = original - recovered (positive = distance reduced)")


if __name__ == '__main__':
    main()
