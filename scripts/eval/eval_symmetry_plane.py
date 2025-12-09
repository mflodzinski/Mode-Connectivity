"""
Evaluate two-segment linear path through optimized symmetry plane point.

Evaluates the path w₁ → θ* → w₂ where θ* is the optimal point found
via symmetry plane optimization.

Saves results in same format as eval_linear.py and eval_curve.py for easy comparison.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import data
import models
import utils


def eval_symmetry_plane(args):
    """Evaluate two-segment path at multiple points."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("SYMMETRY PLANE PATH EVALUATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation points: {args.num_points}")
    print("=" * 80)

    # Load data
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        split_classes=None
    )

    # Load model architecture
    architecture = getattr(models, args.model)
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    model.to(device)

    criterion = F.cross_entropy

    # Load checkpoints
    print("\nLoading checkpoints...")
    print(f"  Endpoint 1: {args.init_start}")
    print(f"  Midpoint θ*: {args.theta_checkpoint}")
    print(f"  Endpoint 2: {args.init_end}")

    ckpt1 = torch.load(args.init_start, map_location=device)
    ckpt_theta = torch.load(args.theta_checkpoint, map_location=device)
    ckpt2 = torch.load(args.init_end, map_location=device)

    w1 = ckpt1.get('model_state', ckpt1)
    theta = ckpt_theta.get('model_state', ckpt_theta)
    w2 = ckpt2.get('model_state', ckpt2)

    # Prepare evaluation
    num_points = args.num_points
    half_points = num_points // 2

    # t values for each segment
    # Segment 1: t ∈ [0, 0.5], Segment 2: t ∈ [0.5, 1]
    t_segment1 = np.linspace(0, 0.5, half_points, endpoint=False)
    t_segment2 = np.linspace(0.5, 1.0, num_points - half_points + 1)

    t_values = np.concatenate([t_segment1, t_segment2])

    print(f"\nEvaluating at {len(t_values)} points along two-segment path...")
    print(f"  Segment 1 (w₁ → θ*): {half_points} points")
    print(f"  Segment 2 (θ* → w₂): {num_points - half_points + 1} points")

    # Storage for results
    train_loss = np.zeros(len(t_values))
    train_acc = np.zeros(len(t_values))
    test_loss = np.zeros(len(t_values))
    test_acc = np.zeros(len(t_values))
    train_err = np.zeros(len(t_values))
    test_err = np.zeros(len(t_values))

    columns = ['t', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']

    # Evaluate each point
    for i, t in enumerate(t_values):
        # Determine which segment and interpolate
        if t <= 0.5:
            # Segment 1: w₁ → θ*
            # Map t ∈ [0, 0.5] to local_t ∈ [0, 1]
            local_t = t * 2.0
            w_t = {key: (1 - local_t) * w1[key] + local_t * theta[key]
                   for key in w1.keys()}
        else:
            # Segment 2: θ* → w₂
            # Map t ∈ [0.5, 1] to local_t ∈ [0, 1]
            local_t = (t - 0.5) * 2.0
            w_t = {key: (1 - local_t) * theta[key] + local_t * w2[key]
                   for key in theta.keys()}

        # Load interpolated weights
        model.load_state_dict(w_t)

        # Update batch normalization running statistics
        utils.update_bn(loaders['train'], model, device=device)

        # Evaluate on train set
        train_res = utils.test(loaders['train'], model, criterion, device=device)
        train_loss[i] = train_res['loss']
        train_acc[i] = train_res['accuracy']
        train_err[i] = 100.0 - train_acc[i]

        # Evaluate on test set
        test_res = utils.test(loaders['test'], model, criterion, device=device)
        test_loss[i] = test_res['loss']
        test_acc[i] = test_res['accuracy']
        test_err[i] = 100.0 - test_acc[i]

        values = [t, train_loss[i], train_acc[i], test_loss[i], test_acc[i]]

        print(f"t={t:.3f} | Train Loss: {train_loss[i]:.4f} | "
              f"Train Acc: {train_acc[i]:.2f}% | "
              f"Test Loss: {test_loss[i]:.4f} | "
              f"Test Acc: {test_acc[i]:.2f}%")

    # Save results
    output_path = os.path.join(args.dir, 'symmetry_plane.npz')
    np.savez(
        output_path,
        t_values=t_values,
        train_loss=train_loss,
        train_accuracy=train_acc,
        train_err=train_err,
        test_loss=test_loss,
        test_accuracy=test_acc,
        test_err=test_err,
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Max train loss: {np.max(train_loss):.4f}")
    print(f"  Max test loss:  {np.max(test_loss):.4f}")
    print(f"  Max train error: {np.max(train_err):.2f}%")
    print(f"  Max test error:  {np.max(test_err):.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Symmetry Plane Path')

    # Checkpoints
    parser.add_argument('--init_start', type=str, required=True,
                       help='Path to first endpoint checkpoint')
    parser.add_argument('--theta_checkpoint', type=str, required=True,
                       help='Path to optimized theta checkpoint')
    parser.add_argument('--init_end', type=str, required=True,
                       help='Path to second endpoint checkpoint')

    # Model and data
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--transform', type=str, required=True, help='Transform type')
    parser.add_argument('--use_test', action='store_true', help='Use test set')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    # Evaluation
    parser.add_argument('--num_points', type=int, default=61,
                       help='Total number of evaluation points')

    # Output
    parser.add_argument('--dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    eval_symmetry_plane(args)
