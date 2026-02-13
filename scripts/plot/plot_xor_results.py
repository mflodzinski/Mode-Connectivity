"""
Plot XOR Experiment Results.

Visualizes:
1. Barrier before vs after alignment for each pair
2. Distribution of barriers
3. L2 distance vs barrier relationship
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


def load_results(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Plot XOR experiment results')
    parser.add_argument('--input', type=str, default='results/xor/results.json',
                        help='Input results JSON file')
    parser.add_argument('--output', type=str, default='plots/xor_results.png',
                        help='Output plot file')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    # Load results
    input_path = os.path.join(project_root, args.input)
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    results = load_results(input_path)
    pairs = results['pair_results']

    # Extract data
    barriers_before = [p['barrier_before']['barrier'] for p in pairs]
    barriers_after = [p['barrier_after']['barrier'] for p in pairs]
    distances_before = [p['distance_before']['l2_distance'] for p in pairs]
    distances_after = [p['distance_after']['l2_distance'] for p in pairs]
    pair_labels = [f"({p['seed_a']},{p['seed_b']})" for p in pairs]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Colors
    before_color = '#1f77b4'  # Blue
    after_color = '#2ca02c'   # Green

    # =========================================================================
    # Plot 1: Barrier comparison bar chart
    # =========================================================================
    ax1 = axes[0, 0]
    x = np.arange(len(pairs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, barriers_before, width, label='Before alignment',
                    color=before_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, barriers_after, width, label='After alignment',
                    color=after_color, alpha=0.8)

    ax1.set_xlabel('Pair', fontsize=10)
    ax1.set_ylabel('Barrier (%)', fontsize=10)
    ax1.set_title('Barrier Before vs After Alignment', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='LMC threshold')
    ax1.set_ylim(0, max(max(barriers_before), max(barriers_after)) * 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Plot 2: Barrier distribution (histogram)
    # =========================================================================
    ax2 = axes[0, 1]
    bins = np.linspace(0, 100, 21)

    ax2.hist(barriers_before, bins=bins, alpha=0.6, label='Before alignment',
             color=before_color, edgecolor='black')
    ax2.hist(barriers_after, bins=bins, alpha=0.6, label='After alignment',
             color=after_color, edgecolor='black')

    ax2.set_xlabel('Barrier (%)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Barrier Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: L2 distance vs barrier (before alignment)
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.scatter(distances_before, barriers_before, c=before_color, s=80,
                alpha=0.7, edgecolors='black', linewidths=0.5)

    # Add labels
    for i, label in enumerate(pair_labels):
        ax3.annotate(label, (distances_before[i], barriers_before[i]),
                     textcoords="offset points", xytext=(3, 3), fontsize=7)

    ax3.set_xlabel('L2 Distance', fontsize=10)
    ax3.set_ylabel('Barrier (%)', fontsize=10)
    ax3.set_title('Distance vs Barrier (Before Alignment)', fontsize=11, fontweight='bold')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: L2 distance vs barrier (after alignment)
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.scatter(distances_after, barriers_after, c=after_color, s=80,
                alpha=0.7, edgecolors='black', linewidths=0.5)

    # Add labels
    for i, label in enumerate(pair_labels):
        ax4.annotate(label, (distances_after[i], barriers_after[i]),
                     textcoords="offset points", xytext=(3, 3), fontsize=7)

    ax4.set_xlabel('L2 Distance', fontsize=10)
    ax4.set_ylabel('Barrier (%)', fontsize=10)
    ax4.set_title('Distance vs Barrier (After Alignment)', fontsize=11, fontweight='bold')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Match axes limits for comparison
    x_max = max(max(distances_before), max(distances_after)) * 1.1
    y_max = max(max(barriers_before), max(barriers_after)) * 1.1
    ax3.set_xlim(0, x_max)
    ax3.set_ylim(-2, y_max)
    ax4.set_xlim(0, x_max)
    ax4.set_ylim(-2, y_max)

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()

    # Print summary
    summary = results['summary']
    print("\n" + "=" * 60)
    print("XOR EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Architecture: {results['config']['architecture']}")
    print(f"Number of networks: {len(results['trained_seeds'])}")
    print(f"Number of pairs: {summary['num_pairs']}")
    print()
    print("Barriers BEFORE alignment:")
    print(f"  Mean: {summary['barriers_before']['mean']:.2f}%")
    print(f"  Std:  {summary['barriers_before']['std']:.2f}%")
    print(f"  Range: [{summary['barriers_before']['min']:.2f}%, {summary['barriers_before']['max']:.2f}%]")
    print()
    print("Barriers AFTER alignment:")
    print(f"  Mean: {summary['barriers_after']['mean']:.2f}%")
    print(f"  Std:  {summary['barriers_after']['std']:.2f}%")
    print(f"  Range: [{summary['barriers_after']['min']:.2f}%, {summary['barriers_after']['max']:.2f}%]")
    print()
    print(f"LMC pairs (barrier < {summary['lmc_threshold']}%):")
    print(f"  Before: {summary['num_lmc_before']}/{summary['num_pairs']}")
    print(f"  After:  {summary['num_lmc_after']}/{summary['num_pairs']}")
    print()

    # Conclusion
    if summary['num_lmc_after'] == summary['num_pairs']:
        print("CONCLUSION: ONE-BASIN HYPOTHESIS SUPPORTED")
        print("All pairs are LMC after alignment!")
    elif summary['num_lmc_after'] > summary['num_lmc_before']:
        improvement = summary['num_lmc_after'] - summary['num_lmc_before']
        print(f"CONCLUSION: PARTIAL SUPPORT ({improvement} more pairs LMC after alignment)")
    else:
        print("CONCLUSION: ONE-BASIN HYPOTHESIS NOT SUPPORTED")


if __name__ == '__main__':
    main()
