#!/usr/bin/env python3
"""
Plot train and test loss/error on the same plot to compare overfitting behavior.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add lib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from lib.analysis import plotting


def plot_train_test_comparison(curve_file, output_path, title):
    """Create comparison plot showing train and test metrics together."""

    # Load data
    print(f"Loading evaluation data from: {curve_file}")
    data = np.load(curve_file)

    t = data['ts']
    train_loss = data['tr_loss']
    test_loss = data['te_loss']
    train_err = data['tr_err']
    test_err = data['te_err']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss (Train vs Test)
    ax = axes[0]
    ax.plot(t, train_loss, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax.plot(t, test_loss, 'r-', linewidth=2, label='Test Loss', alpha=0.8)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train vs Test Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    # Plot 2: Error (Train vs Test)
    ax = axes[1]
    ax.plot(t, train_err, 'b-', linewidth=2, label='Train Error', alpha=0.8)
    ax.plot(t, test_err, 'r-', linewidth=2, label='Test Error', alpha=0.8)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Train vs Test Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Print statistics
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print("\nEndpoint Metrics:")
    print(f"  t=0.0 - Train Loss: {train_loss[0]:.4f}, Test Loss: {test_loss[0]:.4f}")
    print(f"  t=0.0 - Train Error: {train_err[0]:.2f}%, Test Error: {test_err[0]:.2f}%")
    print(f"  t=1.0 - Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}")
    print(f"  t=1.0 - Train Error: {train_err[-1]:.2f}%, Test Error: {test_err[-1]:.2f}%")

    print("\nAverage Metrics:")
    print(f"  Train Loss: {np.mean(train_loss):.4f}, Test Loss: {np.mean(test_loss):.4f}")
    print(f"  Train Error: {np.mean(train_err):.2f}%, Test Error: {np.mean(test_err):.2f}%")

    print("\nMax Metrics:")
    print(f"  Train Loss: {np.max(train_loss):.4f}, Test Loss: {np.max(test_loss):.4f}")
    print(f"  Train Error: {np.max(train_err):.2f}%, Test Error: {np.max(test_err):.2f}%")

    # Overfitting gap
    avg_train_err = np.mean(train_err)
    avg_test_err = np.mean(test_err)
    gap = avg_test_err - avg_train_err
    print(f"\nOverfitting Gap (Test - Train Error): {gap:.2f}%")
    print("=" * 80)

    # Save figure
    plt.tight_layout()
    plotting.save_figure(fig, output_path)
    print(f"\nPlot saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot train vs test loss and error on the same plots'
    )
    parser.add_argument('--curve-file', type=str, required=True,
                       help='Path to curve.npz file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for the plot')
    parser.add_argument('--title', type=str, default='Train vs Test Comparison',
                       help='Plot title')

    args = parser.parse_args()

    plot_train_test_comparison(args.curve_file, args.output, args.title)
