"""Analyze predictions along Bezier curves.

This consolidated script replaces:
- analyze_prediction_changes.py
- compare_endpoint_accuracies.py
- save_unstable_sample_images.py

Modes:
  analyze:           Analyze prediction changes along curve
  compare-endpoints: Compare endpoint accuracies and agreement
  save-images:       Save images of unstable samples
  all:               Run all analyses and generate report
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, '../lib')

from lib.analysis import prediction_analyzer
from lib.core import output as io


def main():
    parser = argparse.ArgumentParser(
        description='Analyze predictions along Bezier curves'
    )

    # Required arguments
    parser.add_argument('--mode', required=True,
                       choices=['analyze', 'compare-endpoints', 'save-images', 'all'],
                       help='Analysis mode')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions file (.npz)')

    # Common arguments
    parser.add_argument('--output', type=str,
                       help='Output directory for results')

    # Save-images specific
    parser.add_argument('--top-k', type=int, default=20,
                       help='[save-images] Number of top unstable samples to save (default: 20)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       help='[save-images] Dataset name (default: CIFAR10)')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='[save-images] Path to dataset (default: ./data)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"PREDICTION ANALYSIS: {args.mode.upper().replace('-', ' ')}")
    print(f"{'='*70}")
    print(f"\nPredictions file: {args.predictions}")

    # Create analyzer
    analyzer = prediction_analyzer.PredictionAnalyzer(args.predictions)

    if args.mode in ['analyze', 'all']:
        # ============================================================
        # ANALYZE PREDICTION CHANGES
        # ============================================================
        print(f"\n{'='*70}")
        print("ANALYZING PREDICTION CHANGES")
        print(f"{'='*70}")

        results = analyzer.analyze_changes(verbose=True)

        # Save results if output specified
        if args.output:
            output_dir = io.ensure_dir(args.output)
            io.save_json(results, output_dir / 'prediction_analysis.json')
            print(f"\n✓ Results saved to {output_dir}/prediction_analysis.json")

    if args.mode in ['compare-endpoints', 'all']:
        # ============================================================
        # COMPARE ENDPOINT ACCURACIES
        # ============================================================
        print(f"\n{'='*70}")
        print("COMPARING ENDPOINT ACCURACIES")
        print(f"{'='*70}")

        endpoint_results = analyzer.compare_endpoints(verbose=True)

        # Save results if output specified
        if args.output:
            output_dir = io.ensure_dir(args.output)
            io.save_json(endpoint_results, output_dir / 'endpoint_comparison.json')
            print(f"\n✓ Results saved to {output_dir}/endpoint_comparison.json")

    if args.mode in ['save-images', 'all']:
        # ============================================================
        # SAVE UNSTABLE SAMPLE IMAGES
        # ============================================================
        if not args.output:
            print("\n⚠ WARNING: --output required for save-images mode")
        else:
            print(f"\n{'='*70}")
            print("SAVING UNSTABLE SAMPLE IMAGES")
            print(f"{'='*70}")

            output_dir = io.ensure_dir(args.output)

            analyzer.save_sample_images(
                output_dir=str(output_dir),
                top_k=args.top_k,
                dataset_name=args.dataset,
                data_path=args.data_path
            )

    if args.mode == 'all':
        # ============================================================
        # GENERATE COMPREHENSIVE REPORT
        # ============================================================
        if not args.output:
            print("\n⚠ WARNING: --output required for generating report")
        else:
            print(f"\n{'='*70}")
            print("GENERATING COMPREHENSIVE REPORT")
            print(f"{'='*70}")

            output_dir = io.ensure_dir(args.output)

            analyzer.generate_report(
                output_dir=str(output_dir),
                dataset_name=args.dataset,
                data_path=args.data_path
            )

    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
