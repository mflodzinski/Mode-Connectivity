"""Unified mode connectivity evaluation script.

This consolidated script replaces:
- eval_linear.py
- eval_polygon_cpu.py
- eval_symmetry_plane.py
- eval_symmetry_comparison.py

Modes:
  linear:     Linear interpolation between two endpoints
  curve:      Bezier/PolyChain curve evaluation
  symmetry:   Symmetry plane path (w1 → θ* → w2)
  comparison: Compare linear vs symmetry plane paths
"""

import argparse
import os
import numpy as np

# Add lib to path
import sys
sys.path.insert(0, '../lib')

from lib.core import setup, checkpoint, models, data, output
from lib.evaluation import interpolation, evaluate
from lib.utils.args import ArgumentParserBuilder

# Aliases for backward compatibility
EvalSetup = setup.EvalSetup  # Use the wrapper class
CheckpointLoader = checkpoint.CheckpointLoader
Interpolator = interpolation.Interpolator
PathEvaluator = evaluate.PathEvaluator
ResultSaver = output.ResultSaver


def evaluate_linear(args):
    """Evaluate linear interpolation between two endpoints.

    Output: linear.npz with ts, l2_norm, tr/te loss/acc/err
    """
    print(f"\n{'='*70}")
    print("LINEAR INTERPOLATION EVALUATION")
    print(f"{'='*70}\n")

    # Setup
    EvalSetup.add_external_path()
    device = EvalSetup.get_device()
    loaders, num_classes = EvalSetup.load_data(
        args.dataset, args.data_path, args.batch_size,
        args.num_workers, args.transform, args.use_test,
        shuffle_train=False
    )
    architecture = EvalSetup.get_architecture(args.model)

    # Load checkpoints
    loader = CheckpointLoader(device)
    w1, w2 = loader.load_endpoints(args.init_start, args.init_end)

    # Create model
    model = EvalSetup.create_standard_model(architecture, num_classes, device)

    # Prepare evaluation
    ts = np.linspace(0.0, 1.0, args.num_points)
    evaluator = PathEvaluator(loaders, device)

    # Create interpolation function
    def linear_interp(t):
        return Interpolator.linear(w1, w2, t)

    # Evaluate path
    results = evaluator.evaluate_path(
        model, ts, linear_interp,
        update_bn=True, verbose=True
    )

    # Compute L2 norms along path
    l2_norms = np.zeros(len(ts))
    for i, t in enumerate(ts):
        weights = linear_interp(t)
        l2_norms[i] = Interpolator.compute_l2_norm(weights)

    # Save results
    output_path = ResultSaver.get_standard_output_path(args.dir, 'linear.npz')
    ResultSaver.save_with_l2(
        output_path,
        results['ts'],
        {'loss': results['tr_loss'], 'acc': results['tr_acc'], 'err': results['tr_err']},
        {'loss': results['te_loss'], 'acc': results['te_acc'], 'err': results['te_err']},
        l2_norms
    )

    print(f"\n✓ Results saved to {output_path}")


def evaluate_curve(args):
    """Evaluate curve model (Bezier, PolyChain, etc.).

    Output: curve.npz with ts, tr/te loss/acc/err
    """
    print(f"\n{'='*70}")
    print(f"{args.curve.upper()} CURVE EVALUATION")
    print(f"{'='*70}\n")

    # Setup
    EvalSetup.add_external_path()
    device = EvalSetup.get_device()
    loaders, num_classes = EvalSetup.load_data(
        args.dataset, args.data_path, args.batch_size,
        args.num_workers, args.transform, args.use_test,
        shuffle_train=False
    )
    architecture = EvalSetup.get_architecture(args.model)

    # Create curve model
    curve_model = EvalSetup.create_curve_model(
        architecture, num_classes, args.curve, args.num_bends, device
    )

    # Load curve checkpoint
    loader = CheckpointLoader(device)
    loader.load_into_model(curve_model, args.ckpt)

    # Prepare evaluation
    ts = np.linspace(0.0, 1.0, args.num_points)
    evaluator = PathEvaluator(loaders, device)

    # Evaluate curve
    results = evaluator.evaluate_curve_path(
        curve_model, ts,
        update_bn=True, verbose=True
    )

    # Save results
    output_path = ResultSaver.get_standard_output_path(args.dir, 'curve.npz')
    ResultSaver.save_from_dict(output_path, results)

    print(f"\n✓ Results saved to {output_path}")


def evaluate_symmetry(args):
    """Evaluate symmetry plane path (w1 → θ* → w2).

    Output: symmetry_plane.npz with ts, tr/te loss/acc/err
    """
    print(f"\n{'='*70}")
    print("SYMMETRY PLANE EVALUATION")
    print(f"{'='*70}\n")

    # Setup
    EvalSetup.add_external_path()
    device = EvalSetup.get_device()
    loaders, num_classes = EvalSetup.load_data(
        args.dataset, args.data_path, args.batch_size,
        args.num_workers, args.transform, args.use_test,
        shuffle_train=False
    )
    architecture = EvalSetup.get_architecture(args.model)

    # Load checkpoints
    loader = CheckpointLoader(device)
    w1, theta, w2 = loader.load_symmetry(
        args.init_start, args.theta_checkpoint, args.init_end
    )

    # Create model
    model = EvalSetup.create_standard_model(architecture, num_classes, device)

    # Prepare evaluation
    ts = np.linspace(0.0, 1.0, args.num_points)
    evaluator = PathEvaluator(loaders, device)

    # Create interpolation function
    def symmetry_interp(t):
        return Interpolator.symmetry_plane(w1, theta, w2, t)

    # Evaluate path
    results = evaluator.evaluate_path(
        model, ts, symmetry_interp,
        update_bn=True, verbose=True
    )

    # Save results
    output_path = ResultSaver.get_standard_output_path(args.dir, 'symmetry_plane.npz')
    ResultSaver.save_from_dict(output_path, results)

    print(f"\n✓ Results saved to {output_path}")


def evaluate_comparison(args):
    """Compare linear vs symmetry plane paths.

    Output: comparison.npz with both paths' metrics
    """
    print(f"\n{'='*70}")
    print("CONNECTIVITY METHOD COMPARISON")
    print(f"{'='*70}\n")

    # Setup
    EvalSetup.add_external_path()
    device = EvalSetup.get_device()
    loaders, num_classes = EvalSetup.load_data(
        args.dataset, args.data_path, args.batch_size,
        args.num_workers, args.transform, args.use_test,
        shuffle_train=False
    )
    architecture = EvalSetup.get_architecture(args.model)

    # Load checkpoints
    loader = CheckpointLoader(device)
    w1, theta, w2 = loader.load_symmetry(
        args.init_start, args.theta_checkpoint, args.init_end
    )

    # Create model
    model = EvalSetup.create_standard_model(architecture, num_classes, device)

    # Prepare evaluation
    ts = np.linspace(0.0, 1.0, args.num_points)
    evaluator = PathEvaluator(loaders, device)

    # Evaluate linear path
    print(f"\n{'='*70}")
    print("1. EVALUATING LINEAR PATH")
    print(f"{'='*70}")

    def linear_interp(t):
        return Interpolator.linear(w1, w2, t)

    linear_results = evaluator.evaluate_path(
        model, ts, linear_interp,
        update_bn=True, verbose=True
    )

    # Evaluate symmetry plane path
    print(f"\n{'='*70}")
    print("2. EVALUATING SYMMETRY PLANE PATH")
    print(f"{'='*70}")

    def symmetry_interp(t):
        return Interpolator.symmetry_plane(w1, theta, w2, t)

    symplane_results = evaluator.evaluate_path(
        model, ts, symmetry_interp,
        update_bn=True, verbose=True
    )

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    linear_max_loss = np.max(linear_results['te_loss'])
    symplane_max_loss = np.max(symplane_results['te_loss'])
    linear_max_err = np.max(linear_results['te_err'])
    symplane_max_err = np.max(symplane_results['te_err'])

    print(f"\nLinear Path:")
    print(f"  Max test loss: {linear_max_loss:.4f}")
    print(f"  Max test error: {linear_max_err:.2f}%")

    print(f"\nSymmetry Plane Path:")
    print(f"  Max test loss: {symplane_max_loss:.4f}")
    print(f"  Max test error: {symplane_max_err:.2f}%")

    print(f"\nImprovement:")
    loss_improvement = (linear_max_loss - symplane_max_loss) / linear_max_loss * 100
    err_improvement = (linear_max_err - symplane_max_err) / linear_max_err * 100
    print(f"  Loss reduction: {loss_improvement:.2f}%")
    print(f"  Error reduction: {err_improvement:.2f}%")

    # Save results
    output_path = ResultSaver.get_standard_output_path(args.dir, 'comparison.npz')
    ResultSaver.save_comparison(output_path, ts, linear_results, symplane_results)

    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified mode connectivity evaluation'
    )

    # Custom arguments specific to this script
    parser.add_argument('--mode', required=True,
                       choices=['linear', 'curve', 'symmetry', 'comparison'],
                       help='Evaluation mode')
    parser.add_argument('--dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--use-test', action='store_true', default=True,
                       help='Use test set (default: True)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_model_args(parser)
    ArgumentParserBuilder.add_dataset_args(parser)

    # Evaluation arguments
    parser.add_argument('--num-points', type=int, default=61,
                       help='Number of points to evaluate along path (default: 61)')

    # Linear mode arguments
    parser.add_argument('--init-start', type=str,
                       help='[linear/symmetry/comparison] Path to start endpoint checkpoint')
    parser.add_argument('--init-end', type=str,
                       help='[linear/symmetry/comparison] Path to end endpoint checkpoint')

    # Curve mode arguments
    parser.add_argument('--ckpt', type=str,
                       help='[curve] Path to curve checkpoint')
    parser.add_argument('--curve', type=str, default='PolyChain',
                       help='[curve] Curve type (default: PolyChain)')
    parser.add_argument('--num-bends', type=int, default=3,
                       help='[curve] Number of bend points (default: 3)')

    # Symmetry/comparison mode arguments
    parser.add_argument('--theta-checkpoint', type=str,
                       help='[symmetry/comparison] Path to symmetry point checkpoint')

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == 'linear':
        if not args.init_start or not args.init_end:
            parser.error("--init_start and --init_end required for linear mode")
    elif args.mode == 'curve':
        if not args.ckpt:
            parser.error("--ckpt required for curve mode")
    elif args.mode in ['symmetry', 'comparison']:
        if not args.init_start or not args.init_end or not args.theta_checkpoint:
            parser.error("--init_start, --init_end, and --theta_checkpoint required for symmetry/comparison mode")

    # Route to appropriate mode
    if args.mode == 'linear':
        evaluate_linear(args)
    elif args.mode == 'curve':
        evaluate_curve(args)
    elif args.mode == 'symmetry':
        evaluate_symmetry(args)
    elif args.mode == 'comparison':
        evaluate_comparison(args)

    print(f"\n{'='*70}")
    print("✓ EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
