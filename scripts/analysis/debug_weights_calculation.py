#!/usr/bin/env python3
"""
Debug script to trace exactly what model.weights(t) calculates and returns.
This will help understand why eval_curve.py might be producing different results.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/dnn-mode-connectivity'))

import curves
import models


def main():
    checkpoint_path = 'results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/checkpoints/checkpoint-150.pt'

    print("=" * 80)
    print("DETAILED WEIGHTS CALCULATION DEBUG")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = curves.CurveNet(
        10,
        curves.Bezier,
        models.VGG16.curve,
        3,
        architecture_kwargs=models.VGG16.kwargs,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Calculate at t=0.5
    t = torch.FloatTensor([0.5])

    print("-" * 80)
    print("STEP 1: Calculate Bezier coefficients")
    print("-" * 80)
    coeffs_t = model.coeff_layer(t)
    print(f"t = {t.item()}")
    print(f"Bezier coefficients: {coeffs_t.numpy()}")
    print(f"Expected for quadratic Bezier at t=0.5: [0.25, 0.5, 0.25]")
    print()

    print("-" * 80)
    print("STEP 2: Manual parameter extraction")
    print("-" * 80)

    # Extract all parameters manually
    params_0 = []
    params_1 = []
    params_2 = []

    param_count_0 = 0
    param_count_1 = 0
    param_count_2 = 0

    for name, param in model.named_parameters():
        if '_0' in name:
            params_0.append(param.detach().cpu().numpy().ravel())
            param_count_0 += param.numel()
        elif '_1' in name:
            params_1.append(param.detach().cpu().numpy().ravel())
            param_count_1 += param.numel()
        elif '_2' in name:
            params_2.append(param.detach().cpu().numpy().ravel())
            param_count_2 += param.numel()

    w0 = np.concatenate(params_0)
    w1 = np.concatenate(params_1)
    w2 = np.concatenate(params_2)

    print(f"w0 (endpoint 0): {len(params_0)} parameter groups, {param_count_0:,} total parameters")
    print(f"  L2 norm: {np.sqrt(np.sum(w0**2)):.6f}")
    print(f"  First 5 values: {w0[:5]}")
    print()

    print(f"w1 (middle point): {len(params_1)} parameter groups, {param_count_1:,} total parameters")
    print(f"  L2 norm: {np.sqrt(np.sum(w1**2)):.6f}")
    print(f"  First 5 values: {w1[:5]}")
    print()

    print(f"w2 (endpoint 1): {len(params_2)} parameter groups, {param_count_2:,} total parameters")
    print(f"  L2 norm: {np.sqrt(np.sum(w2**2)):.6f}")
    print(f"  First 5 values: {w2[:5]}")
    print()

    print("-" * 80)
    print("STEP 3: Manual Bezier interpolation")
    print("-" * 80)

    c0, c1, c2 = coeffs_t.numpy()
    w_manual = c0 * w0 + c1 * w1 + c2 * w2
    l2_manual = np.sqrt(np.sum(w_manual**2))

    print(f"w(t=0.5) = {c0:.4f}*w0 + {c1:.4f}*w1 + {c2:.4f}*w2")
    print(f"  Shape: {w_manual.shape}")
    print(f"  L2 norm: {l2_manual:.6f}")
    print(f"  First 5 values: {w_manual[:5]}")
    print()

    print("-" * 80)
    print("STEP 4: Using model.weights(t)")
    print("-" * 80)

    with torch.no_grad():
        weights_from_model = model.weights(t)

    l2_from_model = np.sqrt(np.sum(np.square(weights_from_model)))

    print(f"  Shape: {weights_from_model.shape}")
    print(f"  L2 norm: {l2_from_model:.6f}")
    print(f"  First 5 values: {weights_from_model[:5]}")
    print()

    print("-" * 80)
    print("STEP 5: Compare results")
    print("-" * 80)

    print(f"Manual calculation L2: {l2_manual:.6f}")
    print(f"model.weights() L2:    {l2_from_model:.6f}")
    print(f"Difference:            {abs(l2_manual - l2_from_model):.10f}")
    print()

    # Check if arrays are identical
    arrays_equal = np.allclose(w_manual, weights_from_model, rtol=1e-5, atol=1e-8)
    print(f"Arrays are identical: {arrays_equal}")

    if not arrays_equal:
        max_diff = np.max(np.abs(w_manual - weights_from_model))
        print(f"  Max element-wise difference: {max_diff:.10f}")
        diff_indices = np.where(np.abs(w_manual - weights_from_model) > 1e-6)[0]
        print(f"  Number of differing elements: {len(diff_indices)}")
        if len(diff_indices) > 0:
            print(f"  First differing index: {diff_indices[0]}")
            print(f"    Manual: {w_manual[diff_indices[0]]}")
            print(f"    Model:  {weights_from_model[diff_indices[0]]}")
    print()

    print("-" * 80)
    print("STEP 6: Test what eval_curve.py calculates")
    print("-" * 80)

    # Simulate what eval_curve.py does
    print("Simulating eval_curve.py calculation:")
    t_cuda_sim = torch.FloatTensor([0.5])  # CPU version
    weights_eval_sim = model.weights(t_cuda_sim)
    l2_eval_sim = np.sqrt(np.sum(np.square(weights_eval_sim)))

    print(f"  weights = model.weights(t)")
    print(f"  l2_norm = np.sqrt(np.sum(np.square(weights)))")
    print(f"  Result: {l2_eval_sim:.6f}")
    print()

    print("-" * 80)
    print("STEP 7: Check what's in existing curve.npz")
    print("-" * 80)

    curve_path = 'results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/evaluations/curve.npz'
    if os.path.exists(curve_path):
        curve_data = np.load(curve_path)
        print(f"Loaded: {curve_path}")
        print(f"  L2 at t=0.5 (index 30): {curve_data['l2_norm'][30]:.6f}")
        print(f"  Discrepancy from correct value: {abs(curve_data['l2_norm'][30] - l2_manual):.6f}")
    else:
        print(f"File not found: {curve_path}")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Correct L2 at t=0.5 (manual calculation):     {l2_manual:.6f}")
    print(f"model.weights() result:                       {l2_from_model:.6f}")
    print(f"eval_curve.py simulation:                     {l2_eval_sim:.6f}")
    if os.path.exists(curve_path):
        print(f"Existing curve.npz file:                      {curve_data['l2_norm'][30]:.6f}")
    print()
    print("Conclusion:")
    if abs(l2_manual - l2_from_model) < 1e-6:
        print("  ✓ model.weights() calculates correctly")
    else:
        print("  ✗ model.weights() has a bug!")

    if os.path.exists(curve_path):
        if abs(l2_manual - curve_data['l2_norm'][30]) < 1e-6:
            print("  ✓ curve.npz matches correct calculation")
        else:
            print("  ✗ curve.npz does NOT match - file is wrong/outdated")
    print("=" * 80)


if __name__ == '__main__':
    main()
