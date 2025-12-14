"""Analyze Bezier curve properties and comparisons.

Modes:
  layer-distances:     Analyze layer-wise distances along curve
  symmetry:            Verify symmetry plane constraints
  compare-seeds:       Compare curves from different random seeds
  compare-inits:       Compare different initialization methods
  checkpoint-distance: Calculate L2 distance between two checkpoints
"""

import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, '../lib')

from lib.curves import analyzer as curve_analyzer
from lib.core import checkpoint, output as io
from lib.evaluation import metrics


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Bezier curve properties'
    )

    # Required arguments
    parser.add_argument('--mode', required=True,
                       choices=['layer-distances', 'symmetry', 'compare-seeds', 'compare-inits', 'checkpoint-distance'],
                       help='Analysis mode')

    # Common arguments
    parser.add_argument('--curve', type=str,
                       help='Path to curve checkpoint')
    parser.add_argument('--endpoint0', type=str,
                       help='Path to first endpoint checkpoint')
    parser.add_argument('--endpoint1', type=str,
                       help='Path to second endpoint checkpoint')
    parser.add_argument('--output', type=str,
                       help='Output directory or file')

    # Layer-distances specific
    parser.add_argument('--num-points', type=int, default=61,
                       help='[layer-distances] Number of points along curve (default: 61)')
    parser.add_argument('--swap-metadata', type=str,
                       help='[layer-distances] Path to swap metadata JSON file')
    parser.add_argument('--permutation-invariant', action='store_true', default=True,
                       help='[layer-distances] Use permutation-invariant distance (default: True)')

    # Symmetry specific
    parser.add_argument('--name', type=str, default='unknown',
                       help='[symmetry] Experiment name for reporting')

    # Compare-seeds specific
    parser.add_argument('--checkpoint-dirs', nargs='+',
                       help='[compare-seeds] List of checkpoint directories')
    parser.add_argument('--checkpoint-name', type=str, default='checkpoint-200.pt',
                       help='[compare-seeds] Checkpoint filename (default: checkpoint-200.pt)')
    parser.add_argument('--threshold', type=float, default=1e-6,
                       help='[compare-seeds] Threshold for similarity (default: 1e-6)')

    # Compare-inits specific
    parser.add_argument('--results-dir', type=str,
                       help='[compare-inits] Root results directory')
    parser.add_argument('--plot', action='store_true',
                       help='[compare-inits] Generate plots')

    # Checkpoint-distance specific
    parser.add_argument('--checkpoint1', type=str,
                       help='[checkpoint-distance] First checkpoint path')
    parser.add_argument('--checkpoint2', type=str,
                       help='[checkpoint-distance] Second checkpoint path')
    parser.add_argument('--show-top-k', type=int, default=10,
                       help='[checkpoint-distance] Number of top layers to display (default: 10)')
    parser.add_argument('--sort-by', type=str, default='normalized',
                       choices=['normalized', 'absolute'],
                       help='[checkpoint-distance] Sort by normalized or absolute distance (default: normalized)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CURVE ANALYSIS: {args.mode.upper().replace('-', ' ')}")
    print(f"{'='*70}\n")

    if args.mode == 'layer-distances':
        # ============================================================
        # MODE: LAYER-WISE DISTANCES
        # ============================================================
        if not args.curve or not args.endpoint0:
            parser.error("--curve and --endpoint0 required for layer-distances mode")

        print(f"Curve:     {args.curve}")
        print(f"Endpoint:  {args.endpoint0}")
        print(f"Points:    {args.num_points}")

        # Create analyzer
        analyzer = curve_analyzer.CurveAnalyzer(args.curve)

        # Compute distances
        print(f"\nComputing layer-wise distances along curve...")
        results = analyzer.compute_layer_distances_along_curve(
            original_checkpoint=args.endpoint0,
            num_points=args.num_points,
            permutation_invariant=args.permutation_invariant
        )

        # Save results
        output_dir = io.ensure_dir(args.output or 'results/layer_distances')
        io.save_npz(
            output_dir / 'layer_distances.npz',
            t_values=np.array(results['t_values']),
            **{k: np.array(v) for k, v in results['layer_distances'].items()}
        )

        # If swap metadata provided, load and include in output
        if args.swap_metadata:
            metadata = io.load_json(args.swap_metadata)
            io.save_json(metadata, output_dir / 'swap_metadata.json')

        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  - layer_distances.npz")
        if args.swap_metadata:
            print(f"  - swap_metadata.json")

    elif args.mode == 'symmetry':
        # ============================================================
        # MODE: VERIFY SYMMETRY PLANE
        # ============================================================
        if not args.curve or not args.endpoint0 or not args.endpoint1:
            parser.error("--curve, --endpoint0, and --endpoint1 required for symmetry mode")

        print(f"Curve:      {args.curve}")
        print(f"Endpoint 0: {args.endpoint0}")
        print(f"Endpoint 1: {args.endpoint1}")
        print(f"Name:       {args.name}")

        # Create analyzer
        analyzer = curve_analyzer.CurveAnalyzer(args.curve)

        # Verify symmetry
        results = analyzer.verify_symmetry_plane(
            endpoint0=args.endpoint0,
            endpoint1=args.endpoint1,
            verbose=True
        )

        # Print verdict based on experiment name
        print(f"\n{'='*70}")
        print("VERDICT")
        print(f"{'='*70}")

        if 'polygon' in args.name.lower():
            if results['is_symmetric']:
                print(f"✓ Polygon curve: Middle point lies on symmetry plane")
                print(f"  This supports the polygon/polychain interpretation")
            else:
                print(f"✗ Polygon curve: Middle point NOT on symmetry plane")
                print(f"  Distance: {results['total_distance']:.6e}")
        elif 'symmetry' in args.name.lower() or 'plane' in args.name.lower():
            if results['is_symmetric']:
                print(f"✓ Middle bend point lies on symmetry plane")
                print(f"  The curve satisfies the symmetry constraint")
            else:
                print(f"✗ Middle bend point does NOT lie on symmetry plane")
                print(f"  Distance: {results['total_distance']:.6e}")
        else:
            if results['is_symmetric']:
                print(f"✓ Symmetry constraint satisfied")
            else:
                print(f"✗ Symmetry constraint violated")

        print(f"{'='*70}\n")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            if not output_path.suffix:
                output_path = output_path / 'symmetry_verification.json'
            io.save_json(results, output_path)
            print(f"✓ Results saved to {output_path}")

    elif args.mode == 'compare-seeds':
        # ============================================================
        # MODE: COMPARE CURVES FROM DIFFERENT SEEDS
        # ============================================================
        if not args.checkpoint_dirs or len(args.checkpoint_dirs) < 2:
            parser.error("At least 2 --checkpoint-dirs required for compare-seeds mode")

        print(f"Comparing {len(args.checkpoint_dirs)} curves:")
        for i, dir_path in enumerate(args.checkpoint_dirs):
            print(f"  {i+1}. {dir_path}")

        # Load all curves
        curve_paths = [Path(d) / args.checkpoint_name for d in args.checkpoint_dirs]
        analyzers = [curve_analyzer.CurveAnalyzer(str(p)) for p in curve_paths]

        # Load first curve
        analyzers[0].load_curve()

        # Compare with others
        other_paths = [str(p) for p in curve_paths[1:]]
        results = analyzers[0].compare_with_other_curves(
            other_checkpoints=other_paths,
            verbose=True
        )

        # Check if curves are identical
        print(f"\n{'='*70}")
        print("SIMILARITY ANALYSIS")
        print(f"{'='*70}")

        all_similar = True
        for comp in results['comparisons']:
            is_similar = comp['normalized_l2'] < args.threshold
            status = "✓ IDENTICAL" if is_similar else "✗ DIFFERENT"
            print(f"\nCurve 1 vs Curve {comp['curve_index']}: {status}")
            print(f"  Normalized L2: {comp['normalized_l2']:.6e}")
            print(f"  Threshold:     {args.threshold:.6e}")

            if not is_similar:
                all_similar = False

        print(f"\n{'='*70}")
        if all_similar:
            print("✓ All curves are identical (within threshold)")
        else:
            print("✗ Curves differ (exceeds threshold)")
        print(f"{'='*70}\n")

        # Save results if output specified
        if args.output:
            io.save_json(results, args.output)
            print(f"✓ Results saved to {args.output}")

    elif args.mode == 'compare-inits':
        # ============================================================
        # MODE: COMPARE INITIALIZATION METHODS
        # ============================================================
        if not args.results_dir:
            parser.error("--results-dir required for compare-inits mode")

        print(f"Results directory: {args.results_dir}")

        # Load evaluation results from different initialization methods
        results_dir = Path(args.results_dir)
        init_methods = ['biased', 'perturbed', 'sphere']  # Expected subdirectories

        print(f"\nLooking for initialization methods: {init_methods}")

        # Check which methods exist
        available_methods = []
        for method in init_methods:
            method_dir = results_dir / method
            if method_dir.exists():
                available_methods.append(method)
                print(f"  ✓ Found: {method}")
            else:
                print(f"  ✗ Missing: {method}")

        if len(available_methods) < 2:
            print(f"\n✗ Need at least 2 initialization methods to compare")
            return

        # Load and analyze results for each method
        print(f"\n{'='*70}")
        print("LOADING RESULTS")
        print(f"{'='*70}\n")

        method_results = {}
        for method in available_methods:
            # Try to load eval_curve.npz
            eval_file = results_dir / method / 'eval_curve.npz'
            if eval_file.exists():
                data = io.load_npz(eval_file)
                method_results[method] = data
                print(f"✓ Loaded {method}: {len(data)} metrics")
            else:
                print(f"✗ Could not find {eval_file}")

        if len(method_results) < 2:
            print(f"\n✗ Could not load enough results to compare")
            return

        # Analyze and compare
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        # Simple comparison - could be extended with plotting
        for method, data in method_results.items():
            print(f"{method.upper()}:")
            if 'test_loss' in data:
                print(f"  Final test loss: {data['test_loss'][-1]:.4f}")
            if 'test_acc' in data:
                print(f"  Final test acc:  {data['test_acc'][-1]:.4f}")
            print()

        # Save comparison if output specified
        if args.output:
            output_dir = io.ensure_dir(args.output)
            io.save_json(
                {method: {k: v.tolist() if hasattr(v, 'tolist') else v
                         for k, v in data.items()}
                 for method, data in method_results.items()},
                output_dir / 'initialization_comparison.json'
            )
            print(f"✓ Comparison saved to {output_dir}/initialization_comparison.json")

        # Generate plots if requested
        if args.plot:
            print(f"\n⚠ Plotting not yet implemented")
            print(f"  Use matplotlib to plot test_loss and test_acc curves")

    elif args.mode == 'checkpoint-distance':
        # ============================================================
        # MODE: CHECKPOINT DISTANCE
        # ============================================================
        if not args.checkpoint1 or not args.checkpoint2:
            parser.error("--checkpoint1 and --checkpoint2 required for checkpoint-distance mode")

        print(f"Checkpoint 1: {args.checkpoint1}")
        print(f"Checkpoint 2: {args.checkpoint2}")

        # Load checkpoints
        state1 = checkpoint.load_state_dict(args.checkpoint1)
        state2 = checkpoint.load_state_dict(args.checkpoint2)

        # Calculate L2 distance
        print(f"\n{'='*70}")
        print("COMPUTING L2 DISTANCE")
        print(f"{'='*70}\n")

        distance_stats = metrics.l2_distance(
            state1,
            state2,
            compute_per_layer=True
        )

        # Display overall statistics
        print(f"Total L2 distance:      {distance_stats['total_l2']:.6f}")
        print(f"Normalized L2 distance: {distance_stats['normalized_total_l2']:.6f}")
        print(f"Total parameters:       {distance_stats['total_params']:,}")

        # Sort layers by distance
        sort_key = 'normalized_l2' if args.sort_by == 'normalized' else 'l2'
        sorted_layers = sorted(
            distance_stats['layer_distances'].items(),
            key=lambda x: x[1][sort_key],
            reverse=True
        )

        # Display top K layers
        print(f"\n{'='*70}")
        print(f"TOP {args.show_top_k} LAYERS BY {args.sort_by.upper()} DISTANCE")
        print(f"{'='*70}")
        print(f"{'Rank':<6}{'Layer':<50}{'L2 Dist':<15}{'Normalized':<15}")
        print("-" * 70)

        for i, (layer_name, stats) in enumerate(sorted_layers[:args.show_top_k]):
            print(f"{i+1:<6}{layer_name[:48]:<50}{stats['l2']:<15.6f}{stats['normalized_l2']:<15.6f}")

        # Count zero-distance layers
        zero_dist_layers = [
            name for name, stats in distance_stats['layer_distances'].items()
            if stats['l2'] < 1e-10
        ]

        if zero_dist_layers:
            print(f"\n{'='*70}")
            print(f"ZERO-DISTANCE LAYERS ({len(zero_dist_layers)})")
            print(f"{'='*70}")
            for layer_name in zero_dist_layers[:10]:  # Show max 10
                print(f"  - {layer_name}")
            if len(zero_dist_layers) > 10:
                print(f"  ... and {len(zero_dist_layers) - 10} more")

        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")

        if distance_stats['normalized_total_l2'] < 1e-6:
            print("✓ Checkpoints are essentially identical")
        elif distance_stats['normalized_total_l2'] < 0.01:
            print("→ Very small distance - likely same initialization or minor differences")
        elif distance_stats['normalized_total_l2'] < 0.1:
            print("→ Small distance - models are similar")
        elif distance_stats['normalized_total_l2'] < 1.0:
            print("→ Moderate distance - models have noticeable differences")
        else:
            print("→ Large distance - models are substantially different")

        print(f"{'='*70}")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            if not output_path.suffix:
                output_path = output_path / 'checkpoint_distance.json'

            # Convert to serializable format
            save_data = {
                'checkpoint1': args.checkpoint1,
                'checkpoint2': args.checkpoint2,
                'total_l2': distance_stats['total_l2'],
                'normalized_total_l2': distance_stats['normalized_total_l2'],
                'total_params': distance_stats['total_params'],
                'layer_distances': {
                    name: {
                        'l2': stats['l2'],
                        'normalized_l2': stats['normalized_l2'],
                        'num_params': stats['num_params']
                    }
                    for name, stats in distance_stats['layer_distances'].items()
                }
            }

            io.save_json(save_data, output_path)
            print(f"\n✓ Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
