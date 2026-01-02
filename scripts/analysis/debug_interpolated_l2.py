#!/usr/bin/env python3
"""
Debug script to investigate why interpolated L2 norm calculation differs
between training (train.py) and evaluation (eval_curve.py).

This script loads a trained curve checkpoint and calculates the interpolated
L2 norm at t=0.5 using both train mode and eval mode to identify the source
of the discrepancy.
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/dnn-mode-connectivity'))

import curves
import models


def calculate_interpolated_l2(model, t_value, mode='eval'):
    """
    Calculate interpolated L2 norm at given t value.

    Args:
        model: CurveNet model
        t_value: Parameter value (0.0 to 1.0)
        mode: 'train' or 'eval' to test model mode effect

    Returns:
        tuple: (l2_norm, weights_shape, weights_sample)
    """
    if mode == 'eval':
        model.eval()
    else:
        model.train()

    t = torch.FloatTensor([t_value])
    if torch.cuda.is_available():
        t = t.cuda()

    with torch.no_grad():
        weights = model.weights(t)

    l2_norm = np.sqrt(np.sum(np.square(weights)))

    return l2_norm, weights.shape, weights[:10]  # Return first 10 values as sample


def calculate_middle_point_l2(model):
    """Calculate L2 norm of raw middle point parameters (w_1)."""
    l2_sum = 0.0
    for name, param in model.named_parameters():
        if '_1' in name and param.requires_grad:
            l2_sum += torch.sum(param ** 2).item()
    return np.sqrt(l2_sum)


def main():
    parser = argparse.ArgumentParser(description='Debug interpolated L2 norm calculation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default='VGG16',
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Dataset name')
    parser.add_argument('--curve', type=str, default='Bezier',
                        help='Curve type')
    parser.add_argument('--num_bends', type=int, default=3,
                        help='Number of bends')

    args = parser.parse_args()

    print("=" * 80)
    print("DEBUGGING INTERPOLATED L2 NORM CALCULATION")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Curve: {args.curve} (num_bends={args.num_bends})")
    print()

    # Determine number of classes
    num_classes = 100 if args.dataset == 'CIFAR100' else 10

    # Load model architecture
    architecture = getattr(models, args.model)
    curve_type = getattr(curves, args.curve)

    # Create curve model
    model = curves.CurveNet(
        num_classes,
        curve_type,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    if torch.cuda.is_available():
        model = model.cuda()

    print("✓ Model loaded successfully")
    print()

    # Calculate middle point L2 norm (raw parameters)
    print("-" * 80)
    print("MIDDLE POINT L2 NORM (raw w_1 parameters)")
    print("-" * 80)
    middle_l2 = calculate_middle_point_l2(model)
    print(f"Middle point L2 norm: {middle_l2:.6f}")
    print()

    # Calculate interpolated L2 at t=0.5 in EVAL mode
    print("-" * 80)
    print("INTERPOLATED L2 AT t=0.5 (model.eval())")
    print("-" * 80)
    l2_eval, shape_eval, sample_eval = calculate_interpolated_l2(model, 0.5, mode='eval')
    print(f"L2 norm: {l2_eval:.6f}")
    print(f"Weights shape: {shape_eval}")
    print(f"First 10 values: {sample_eval}")
    print()

    # Calculate interpolated L2 at t=0.5 in TRAIN mode
    print("-" * 80)
    print("INTERPOLATED L2 AT t=0.5 (model.train())")
    print("-" * 80)
    l2_train, shape_train, sample_train = calculate_interpolated_l2(model, 0.5, mode='train')
    print(f"L2 norm: {l2_train:.6f}")
    print(f"Weights shape: {shape_train}")
    print(f"First 10 values: {sample_train}")
    print()

    # Calculate coefficients at t=0.5 for quadratic Bezier
    print("-" * 80)
    print("BEZIER COEFFICIENTS AT t=0.5")
    print("-" * 80)
    t = torch.FloatTensor([0.5])
    if torch.cuda.is_available():
        t = t.cuda()
    coeffs = model.coeff_layer(t)
    print(f"Coefficients: {coeffs.cpu().numpy()}")
    print(f"Expected for quadratic Bezier: [0.25, 0.5, 0.25]")
    print()

    # Compare results
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Middle point L2:              {middle_l2:.6f}")
    print(f"Interpolated L2 (eval mode):  {l2_eval:.6f}")
    print(f"Interpolated L2 (train mode): {l2_train:.6f}")
    print()
    print(f"Ratio (middle / eval):  {middle_l2 / l2_eval:.6f}")
    print(f"Ratio (middle / train): {middle_l2 / l2_train:.6f}")
    print(f"Difference (eval - train): {abs(l2_eval - l2_train):.6f}")
    print()

    # Check if mode matters
    if abs(l2_eval - l2_train) < 1e-6:
        print("✓ Model mode (train/eval) does NOT affect interpolated L2 calculation")
    else:
        print("⚠️  Model mode (train/eval) DOES affect interpolated L2 calculation")
        print(f"   Difference: {abs(l2_eval - l2_train):.6f}")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
