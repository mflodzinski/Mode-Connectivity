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


def plot_train_test_comparison(bezier_file, polygon_file, output_path, title):
    """Create comparison plot showing train and test metrics for both Bezier and Polygon methods."""

    # Load Bezier data
    print(f"Loading Bezier data from: {bezier_file}")
    bezier_data = np.load(bezier_file)

    # Load Polygon data
    print(f"Loading Polygon data from: {polygon_file}")
    polygon_data = np.load(polygon_file)

    t_bezier = bezier_data['ts']
    t_polygon = polygon_data['ts']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss (Train vs Test for both methods)
    # Bezier = blue, Polygon = red; solid = train, dashed = test
    ax = axes[0]
    ax.plot(t_bezier, bezier_data['tr_loss'], 'b-', linewidth=2, label='Bezier Train', alpha=0.8)
    ax.plot(t_bezier, bezier_data['te_loss'], 'b--', linewidth=2, label='Bezier Test', alpha=0.8)
    ax.plot(t_polygon, polygon_data['tr_loss'], 'r-', linewidth=2, label='Polygon Train', alpha=0.8)
    ax.plot(t_polygon, polygon_data['te_loss'], 'r--', linewidth=2, label='Polygon Test', alpha=0.8)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train vs Test Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    # Plot 2: Error (Train vs Test for both methods)
    # Bezier = blue, Polygon = red; solid = train, dashed = test
    ax = axes[1]
    ax.plot(t_bezier, bezier_data['tr_err'], 'b-', linewidth=2, label='Bezier Train', alpha=0.8)
    ax.plot(t_bezier, bezier_data['te_err'], 'b--', linewidth=2, label='Bezier Test', alpha=0.8)
    ax.plot(t_polygon, polygon_data['tr_err'], 'r-', linewidth=2, label='Polygon Train', alpha=0.8)
    ax.plot(t_polygon, polygon_data['te_err'], 'r--', linewidth=2, label='Polygon Test', alpha=0.8)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Train vs Test Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Print statistics
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    print("\n--- BEZIER CURVE ---")
    print("\nEndpoint Metrics:")
    print(f"  t=0.0 - Train Loss: {bezier_data['tr_loss'][0]:.4f}, Test Loss: {bezier_data['te_loss'][0]:.4f}")
    print(f"  t=0.0 - Train Error: {bezier_data['tr_err'][0]:.2f}%, Test Error: {bezier_data['te_err'][0]:.2f}%")
    print(f"  t=1.0 - Train Loss: {bezier_data['tr_loss'][-1]:.4f}, Test Loss: {bezier_data['te_loss'][-1]:.4f}")
    print(f"  t=1.0 - Train Error: {bezier_data['tr_err'][-1]:.2f}%, Test Error: {bezier_data['te_err'][-1]:.2f}%")

    print("\nAverage Metrics:")
    print(f"  Train Loss: {np.mean(bezier_data['tr_loss']):.4f}, Test Loss: {np.mean(bezier_data['te_loss']):.4f}")
    print(f"  Train Error: {np.mean(bezier_data['tr_err']):.2f}%, Test Error: {np.mean(bezier_data['te_err']):.2f}%")

    bezier_gap = np.mean(bezier_data['te_err']) - np.mean(bezier_data['tr_err'])
    print(f"\nOverfitting Gap: {bezier_gap:.2f}%")

    print("\n--- POLYGON CHAIN ---")
    print("\nEndpoint Metrics:")
    print(f"  t=0.0 - Train Loss: {polygon_data['tr_loss'][0]:.4f}, Test Loss: {polygon_data['te_loss'][0]:.4f}")
    print(f"  t=0.0 - Train Error: {polygon_data['tr_err'][0]:.2f}%, Test Error: {polygon_data['te_err'][0]:.2f}%")
    print(f"  t=1.0 - Train Loss: {polygon_data['tr_loss'][-1]:.4f}, Test Loss: {polygon_data['te_loss'][-1]:.4f}")
    print(f"  t=1.0 - Train Error: {polygon_data['tr_err'][-1]:.2f}%, Test Error: {polygon_data['te_err'][-1]:.2f}%")

    print("\nAverage Metrics:")
    print(f"  Train Loss: {np.mean(polygon_data['tr_loss']):.4f}, Test Loss: {np.mean(polygon_data['te_loss']):.4f}")
    print(f"  Train Error: {np.mean(polygon_data['tr_err']):.2f}%, Test Error: {np.mean(polygon_data['te_err']):.2f}%")

    polygon_gap = np.mean(polygon_data['te_err']) - np.mean(polygon_data['tr_err'])
    print(f"\nOverfitting Gap: {polygon_gap:.2f}%")
    print("=" * 80)

    # Save figure
    plt.tight_layout()
    plotting.save_figure(fig, output_path)
    print(f"\nPlot saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot train vs test loss and error for Bezier and Polygon methods'
    )
    parser.add_argument('--bezier-file', type=str, required=True,
                       help='Path to Bezier curve.npz file')
    parser.add_argument('--polygon-file', type=str, required=True,
                       help='Path to Polygon curve.npz file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for the plot')
    parser.add_argument('--title', type=str, default='Train vs Test Comparison',
                       help='Plot title')

    args = parser.parse_args()

    plot_train_test_comparison(args.bezier_file, args.polygon_file, args.output, args.title)
