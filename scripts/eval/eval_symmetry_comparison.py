"""
Compare different connectivity methods side-by-side.

Evaluates and compares:
1. Direct linear interpolation (w₁ → w₂)
2. Two-segment symmetry plane path (w₁ → θ* → w₂)
3. Bezier curve (optional, if available)

Saves combined results for easy plotting and analysis.
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


def eval_linear_path(model, w1, w2, t_values, loaders, criterion, device):
    """Evaluate direct linear interpolation."""
    losses_train = []
    losses_test = []
    accs_train = []
    accs_test = []

    for t in t_values:
        # Linear interpolation
        w_t = {key: (1 - t) * w1[key] + t * w2[key] for key in w1.keys()}
        model.load_state_dict(w_t)

        # Update BN statistics
        utils.update_bn(loaders['train'], model, device=device)

        # Evaluate
        train_res = utils.test(loaders['train'], model, criterion, device=device)
        test_res = utils.test(loaders['test'], model, criterion, device=device)

        losses_train.append(train_res['loss'])
        losses_test.append(test_res['loss'])
        accs_train.append(train_res['accuracy'])
        accs_test.append(test_res['accuracy'])

    return {
        'train_loss': np.array(losses_train),
        'test_loss': np.array(losses_test),
        'train_acc': np.array(accs_train),
        'test_acc': np.array(accs_test),
    }


def eval_symmetry_path(model, w1, theta, w2, t_values, loaders, criterion, device):
    """Evaluate two-segment symmetry plane path."""
    losses_train = []
    losses_test = []
    accs_train = []
    accs_test = []

    for t in t_values:
        # Determine segment and interpolate
        if t <= 0.5:
            local_t = t * 2.0
            w_t = {key: (1 - local_t) * w1[key] + local_t * theta[key]
                   for key in w1.keys()}
        else:
            local_t = (t - 0.5) * 2.0
            w_t = {key: (1 - local_t) * theta[key] + local_t * w2[key]
                   for key in theta.keys()}

        model.load_state_dict(w_t)
        utils.update_bn(loaders['train'], model, device=device)

        train_res = utils.test(loaders['train'], model, criterion, device=device)
        test_res = utils.test(loaders['test'], model, criterion, device=device)

        losses_train.append(train_res['loss'])
        losses_test.append(test_res['loss'])
        accs_train.append(train_res['accuracy'])
        accs_test.append(test_res['accuracy'])

    return {
        'train_loss': np.array(losses_train),
        'test_loss': np.array(losses_test),
        'train_acc': np.array(accs_train),
        'test_acc': np.array(accs_test),
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("CONNECTIVITY METHODS COMPARISON")
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
    print(f"  Endpoint 2: {args.init_end}")
    print(f"  Theta: {args.theta_checkpoint}")

    ckpt1 = torch.load(args.init_start, map_location=device)
    ckpt2 = torch.load(args.init_end, map_location=device)
    ckpt_theta = torch.load(args.theta_checkpoint, map_location=device)

    w1 = ckpt1.get('model_state', ckpt1)
    w2 = ckpt2.get('model_state', ckpt2)
    theta = ckpt_theta.get('model_state', ckpt_theta)

    # Evaluation points
    t_values = np.linspace(0, 1, args.num_points)

    # Evaluate linear interpolation
    print("\nEvaluating direct linear interpolation...")
    linear_results = eval_linear_path(model, w1, w2, t_values, loaders, criterion, device)
    print("  ✓ Linear interpolation complete")

    # Evaluate symmetry plane path
    print("\nEvaluating symmetry plane path...")
    symplane_results = eval_symmetry_path(model, w1, theta, w2, t_values, loaders, criterion, device)
    print("  ✓ Symmetry plane path complete")

    # Compute comparison metrics
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print("\nMaximum Test Loss:")
    print(f"  Linear interpolation: {np.max(linear_results['test_loss']):.4f}")
    print(f"  Symmetry plane path:  {np.max(symplane_results['test_loss']):.4f}")
    improvement_loss = np.max(linear_results['test_loss']) - np.max(symplane_results['test_loss'])
    print(f"  Improvement: {improvement_loss:.4f} ({improvement_loss/np.max(linear_results['test_loss'])*100:.1f}%)")

    print("\nMaximum Test Error:")
    linear_max_err = 100.0 - np.min(linear_results['test_acc'])
    symplane_max_err = 100.0 - np.min(symplane_results['test_acc'])
    print(f"  Linear interpolation: {linear_max_err:.2f}%")
    print(f"  Symmetry plane path:  {symplane_max_err:.2f}%")
    improvement_err = linear_max_err - symplane_max_err
    print(f"  Improvement: {improvement_err:.2f}%")

    # Save combined results
    output_path = os.path.join(args.dir, 'comparison.npz')
    np.savez(
        output_path,
        t_values=t_values,
        # Linear
        linear_train_loss=linear_results['train_loss'],
        linear_test_loss=linear_results['test_loss'],
        linear_train_acc=linear_results['train_acc'],
        linear_test_acc=linear_results['test_acc'],
        linear_train_err=100.0 - linear_results['train_acc'],
        linear_test_err=100.0 - linear_results['test_acc'],
        # Symmetry plane
        symplane_train_loss=symplane_results['train_loss'],
        symplane_test_loss=symplane_results['test_loss'],
        symplane_train_acc=symplane_results['train_acc'],
        symplane_test_acc=symplane_results['test_acc'],
        symplane_train_err=100.0 - symplane_results['train_acc'],
        symplane_test_err=100.0 - symplane_results['test_acc'],
    )

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Connectivity Methods')

    # Checkpoints
    parser.add_argument('--init_start', type=str, required=True)
    parser.add_argument('--init_end', type=str, required=True)
    parser.add_argument('--theta_checkpoint', type=str, required=True)

    # Model and data
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--transform', type=str, required=True)
    parser.add_argument('--use_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # Evaluation
    parser.add_argument('--num_points', type=int, default=61)

    # Output
    parser.add_argument('--dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
