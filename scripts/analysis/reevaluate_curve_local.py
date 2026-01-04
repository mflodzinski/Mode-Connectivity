#!/usr/bin/env python3
"""
Re-evaluate a curve checkpoint locally (CPU-only) to verify L2 norms.
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/dnn-mode-connectivity'))

import curves
import models


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate curve checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=61)

    args = parser.parse_args()

    print("=" * 80)
    print("RE-EVALUATING CURVE CHECKPOINT (CPU)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of points: {args.num_points}")
    print()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

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

    # Generate t values
    ts = np.linspace(0.0, 1.0, args.num_points)
    l2_norms = np.zeros(args.num_points)

    print("Evaluating L2 norms along curve...")
    for i, t_val in enumerate(ts):
        t = torch.FloatTensor([t_val])
        with torch.no_grad():
            weights = model.weights(t)
        l2_norms[i] = np.sqrt(np.sum(np.square(weights)))

        if i % 10 == 0 or i == args.num_points - 1:
            print(f"  t={t_val:.3f}: L2={l2_norms[i]:.6f}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"L2 at t=0.0: {l2_norms[0]:.6f}")
    print(f"L2 at t=0.5: {l2_norms[args.num_points//2]:.6f}")
    print(f"L2 at t=1.0: {l2_norms[-1]:.6f}")
    print()
    print(f"Min L2: {np.min(l2_norms):.6f}")
    print(f"Max L2: {np.max(l2_norms):.6f}")
    print(f"Mean L2: {np.mean(l2_norms):.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
