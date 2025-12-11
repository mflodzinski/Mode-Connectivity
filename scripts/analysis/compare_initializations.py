"""
Script to compare Bezier curves trained with different initialization methods.
Analyzes how initialization affects final curve quality and training dynamics.
"""
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_evaluation(eval_dir):
    """Load evaluation results from a directory."""
    curve_file = Path(eval_dir) / "curve.npz"

    if not curve_file.exists():
        return None

    data = np.load(str(curve_file))

    return {
        'ts': data['ts'],
        'tr_loss': data['tr_loss'],
        'tr_acc': data['tr_acc'],
        'te_loss': data['te_loss'],
        'te_acc': data['te_acc'],
    }


def analyze_curve(data):
    """Analyze curve quality metrics."""
    # Maximum test error along the curve
    max_test_error = 100.0 - np.min(data['te_acc'])

    # Average test error along the curve
    avg_test_error = 100.0 - np.mean(data['te_acc'])

    # Endpoint test errors
    endpoint0_error = 100.0 - data['te_acc'][0]
    endpoint1_error = 100.0 - data['te_acc'][-1]

    # Midpoint test error
    mid_idx = len(data['te_acc']) // 2
    midpoint_error = 100.0 - data['te_acc'][mid_idx]

    # Barrier: difference between max and average of endpoints
    avg_endpoint_error = (endpoint0_error + endpoint1_error) / 2.0
    barrier = max_test_error - avg_endpoint_error

    return {
        'max_test_error': max_test_error,
        'avg_test_error': avg_test_error,
        'endpoint0_error': endpoint0_error,
        'endpoint1_error': endpoint1_error,
        'midpoint_error': midpoint_error,
        'avg_endpoint_error': avg_endpoint_error,
        'barrier': barrier,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Bezier curves trained with different initializations"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/vgg16/cifar10",
        help="Root directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vgg16/cifar10/figures",
        help="Directory to save comparison plots and tables"
    )

    args = parser.parse_args()

    # Define experiments to compare
    experiments = {
        # Biased linear
        'alpha_0.1': 'curve_init_alpha0.1',
        'alpha_0.25': 'curve_init_alpha0.25',
        'alpha_0.5': 'curve_init_alpha0.5',
        'alpha_0.75': 'curve_init_alpha0.75',
        'alpha_0.9': 'curve_init_alpha0.9',
        # Perturbed
        'perturbed_0.01': 'curve_init_perturbed_small',
        'perturbed_0.05': 'curve_init_perturbed_medium',
        'perturbed_0.1': 'curve_init_perturbed_large',
        # Sphere
        'sphere_inside': 'curve_init_sphere_inside',
        'sphere_outside': 'curve_init_sphere_outside',
    }

    print("\n" + "="*80)
    print("INITIALIZATION METHOD COMPARISON")
    print("="*80)

    # Load all results
    results = {}
    for label, exp_dir in experiments.items():
        eval_dir = Path(args.results_dir) / exp_dir / "evaluations"

        print(f"\nLoading: {label}")
        print(f"  Directory: {eval_dir}")

        data = load_evaluation(eval_dir)
        if data is None:
            print(f"  ⚠️  WARNING: No evaluation found")
            continue

        metrics = analyze_curve(data)
        results[label] = {
            'data': data,
            'metrics': metrics
        }

        print(f"  ✓ Loaded")
        print(f"    Max test error: {metrics['max_test_error']:.2f}%")
        print(f"    Barrier: {metrics['barrier']:.2f}%")

    if not results:
        print("\n❌ ERROR: No results found")
        return 1

    print(f"\n✓ Loaded {len(results)} experiment results")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Method':<20} {'Max Err':<10} {'Avg Err':<10} {'Mid Err':<10} {'Barrier':<10}")
    print("-"*80)

    for label in experiments.keys():
        if label not in results:
            continue

        m = results[label]['metrics']
        print(f"{label:<20} {m['max_test_error']:>8.2f}% {m['avg_test_error']:>8.2f}% "
              f"{m['midpoint_error']:>8.2f}% {m['barrier']:>8.2f}%")

    print("="*80)

    # Group results by initialization type
    biased_results = {k: v for k, v in results.items() if k.startswith('alpha_')}
    perturbed_results = {k: v for k, v in results.items() if k.startswith('perturbed_')}
    sphere_results = {k: v for k, v in results.items() if k.startswith('sphere_')}

    # Create comparison plots
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: All curves together
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Biased linear
    ax = axes[0, 0]
    for label in sorted(biased_results.keys()):
        data = biased_results[label]['data']
        ax.plot(data['ts'], 100 - data['te_acc'], label=label, linewidth=2)
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Biased Linear Initialization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Perturbed linear
    ax = axes[0, 1]
    for label in sorted(perturbed_results.keys()):
        data = perturbed_results[label]['data']
        ax.plot(data['ts'], 100 - data['te_acc'], label=label, linewidth=2)
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Perturbed Linear Initialization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sphere-constrained
    ax = axes[1, 0]
    for label in sorted(sphere_results.keys()):
        data = sphere_results[label]['data']
        ax.plot(data['ts'], 100 - data['te_acc'], label=label, linewidth=2)
    ax.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax.set_ylabel('Test Error (%)', fontsize=12)
    ax.set_title('Sphere-Constrained Initialization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: Barrier comparison
    ax = axes[1, 1]

    labels = []
    barriers = []
    colors = []

    for label in sorted(results.keys()):
        labels.append(label)
        barriers.append(results[label]['metrics']['barrier'])

        if label.startswith('alpha_'):
            colors.append('tab:blue')
        elif label.startswith('perturbed_'):
            colors.append('tab:orange')
        else:
            colors.append('tab:green')

    bars = ax.barh(range(len(labels)), barriers, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Barrier Height (%)', fontsize=12)
    ax.set_title('Barrier Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, barriers)):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=9)

    plt.tight_layout()

    output_file = Path(args.output_dir) / "initialization_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_file}")

    # Plot 2: Overlay all methods on one plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for label in sorted(results.keys()):
        data = results[label]['data']

        if label.startswith('alpha_'):
            color = 'tab:blue'
            linestyle = '-'
        elif label.startswith('perturbed_'):
            color = 'tab:orange'
            linestyle = '--'
        else:
            color = 'tab:green'
            linestyle = '-.'

        ax.plot(data['ts'], 100 - data['te_acc'],
                label=label, linewidth=2, color=color, linestyle=linestyle, alpha=0.7)

    ax.set_xlabel('t (interpolation parameter)', fontsize=14)
    ax.set_ylabel('Test Error (%)', fontsize=14)
    ax.set_title('All Initialization Methods Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = Path(args.output_dir) / "initialization_comparison_overlay.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved overlay plot: {output_file}")

    # Analysis summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    # Best and worst
    best_label = min(results.keys(), key=lambda k: results[k]['metrics']['barrier'])
    worst_label = max(results.keys(), key=lambda k: results[k]['metrics']['barrier'])

    print(f"\nBest initialization: {best_label}")
    print(f"  Barrier: {results[best_label]['metrics']['barrier']:.2f}%")
    print(f"  Max test error: {results[best_label]['metrics']['max_test_error']:.2f}%")

    print(f"\nWorst initialization: {worst_label}")
    print(f"  Barrier: {results[worst_label]['metrics']['barrier']:.2f}%")
    print(f"  Max test error: {results[worst_label]['metrics']['max_test_error']:.2f}%")

    # Group statistics
    if biased_results:
        avg_barrier = np.mean([v['metrics']['barrier'] for v in biased_results.values()])
        print(f"\nBiased linear (average barrier): {avg_barrier:.2f}%")

    if perturbed_results:
        avg_barrier = np.mean([v['metrics']['barrier'] for v in perturbed_results.values()])
        print(f"Perturbed linear (average barrier): {avg_barrier:.2f}%")

    if sphere_results:
        avg_barrier = np.mean([v['metrics']['barrier'] for v in sphere_results.values()])
        print(f"Sphere-constrained (average barrier): {avg_barrier:.2f}%")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
