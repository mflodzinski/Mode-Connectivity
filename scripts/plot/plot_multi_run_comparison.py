"""Plot comparison of multiple curve runs (different seeds) without linear interpolation.

Compares 3 different runs (e.g., seed 0, 42, 123) of the same curve training.
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
from lib.utils.args import ArgumentParserBuilder

parser = argparse.ArgumentParser(description='Plot multi-run comparison of curves')

# Custom arguments
parser.add_argument('--run1', type=str, required=True, metavar='PATH',
                    help='path to run 1 curve.npz results')
parser.add_argument('--run2', type=str, required=True, metavar='PATH',
                    help='path to run 2 curve.npz results')
parser.add_argument('--run3', type=str, required=True, metavar='PATH',
                    help='path to run 3 curve.npz results')
parser.add_argument('--run1-label', type=str, default='Run 1 (seed 0)',
                    help='label for run 1')
parser.add_argument('--run2-label', type=str, default='Run 2 (seed 42)',
                    help='label for run 2')
parser.add_argument('--run3-label', type=str, default='Run 3 (seed 123)',
                    help='label for run 3')
parser.add_argument('--title', type=str, default='Multi-Run Comparison',
                    help='plot title')

# Standard arguments using ArgumentParserBuilder
ArgumentParserBuilder.add_plot_output_args(parser, required=False)

# Override default output
parser.set_defaults(output='results/multi_run_comparison.png')

args = parser.parse_args()

# Load results
run1_data = np.load(args.run1)
run2_data = np.load(args.run2)
run3_data = np.load(args.run3)

# Extract data - Run 1
run1_ts = run1_data['ts']
run1_tr_err = run1_data['tr_err']
run1_tr_loss = run1_data['tr_loss']
run1_te_err = run1_data['te_err']
run1_te_loss = run1_data['te_loss']

# Extract data - Run 2
run2_ts = run2_data['ts']
run2_tr_err = run2_data['tr_err']
run2_tr_loss = run2_data['tr_loss']
run2_te_err = run2_data['te_err']
run2_te_loss = run2_data['te_loss']

# Extract data - Run 3
run3_ts = run3_data['ts']
run3_tr_err = run3_data['tr_err']
run3_tr_loss = run3_data['tr_loss']
run3_te_err = run3_data['te_err']
run3_te_loss = run3_data['te_loss']

# Create figure with 4 subplots (2x2 layout)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot styles
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']
linestyles = ['-', '-', '-']

# Panel 1: Test Error
ax = axes[0, 0]
ax.plot(run1_ts, run1_te_err, color=colors[0], marker=markers[0], linestyle=linestyles[0],
        label=args.run1_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run2_ts, run2_te_err, color=colors[1], marker=markers[1], linestyle=linestyles[1],
        label=args.run2_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run3_ts, run3_te_err, color=colors[2], marker=markers[2], linestyle=linestyles[2],
        label=args.run3_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.set_xlabel('t (interpolation parameter)', fontsize=12)
ax.set_ylabel('Test Error (%)', fontsize=12)
ax.set_title('Test Error along Path', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

# Panel 2: Test Loss
ax = axes[0, 1]
ax.plot(run1_ts, run1_te_loss, color=colors[0], marker=markers[0], linestyle=linestyles[0],
        label=args.run1_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run2_ts, run2_te_loss, color=colors[1], marker=markers[1], linestyle=linestyles[1],
        label=args.run2_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run3_ts, run3_te_loss, color=colors[2], marker=markers[2], linestyle=linestyles[2],
        label=args.run3_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.set_xlabel('t (interpolation parameter)', fontsize=12)
ax.set_ylabel('Test Loss', fontsize=12)
ax.set_title('Test Loss along Path', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

# Panel 3: Train Error
ax = axes[1, 0]
ax.plot(run1_ts, run1_tr_err, color=colors[0], marker=markers[0], linestyle=linestyles[0],
        label=args.run1_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run2_ts, run2_tr_err, color=colors[1], marker=markers[1], linestyle=linestyles[1],
        label=args.run2_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run3_ts, run3_tr_err, color=colors[2], marker=markers[2], linestyle=linestyles[2],
        label=args.run3_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.set_xlabel('t (interpolation parameter)', fontsize=12)
ax.set_ylabel('Train Error (%)', fontsize=12)
ax.set_title('Train Error along Path', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

# Panel 4: Train Loss
ax = axes[1, 1]
ax.plot(run1_ts, run1_tr_loss, color=colors[0], marker=markers[0], linestyle=linestyles[0],
        label=args.run1_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run2_ts, run2_tr_loss, color=colors[1], marker=markers[1], linestyle=linestyles[1],
        label=args.run2_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.plot(run3_ts, run3_tr_loss, color=colors[2], marker=markers[2], linestyle=linestyles[2],
        label=args.run3_label, linewidth=2, markersize=3, markevery=5, alpha=0.8)
ax.set_xlabel('t (interpolation parameter)', fontsize=12)
ax.set_ylabel('Train Loss', fontsize=12)
ax.set_title('Train Loss along Path', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

plt.suptitle(args.title, fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
plotting.save_figure(fig, args.output)

# Generate summary statistics text
summary_lines = []
summary_lines.append("="*70)
summary_lines.append("MULTI-RUN COMPARISON SUMMARY")
summary_lines.append("="*70)
summary_lines.append("")

for run_idx, (run_data, run_label) in enumerate([
    (run1_data, args.run1_label),
    (run2_data, args.run2_label),
    (run3_data, args.run3_label)
], 1):
    te_err = run_data['te_err']
    tr_err = run_data['tr_err']
    ts = run_data['ts']

    summary_lines.append(f"{run_label}:")
    summary_lines.append("  Train metrics:")
    summary_lines.append(f"    Endpoint 1 train error: {tr_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 train error: {tr_err[-1]:.2f}%")
    summary_lines.append(f"    Max train error: {np.max(tr_err):.2f}% at t={ts[np.argmax(tr_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(tr_err) - max(tr_err[0], tr_err[-1]):.2f}%")
    summary_lines.append("  Test metrics:")
    summary_lines.append(f"    Endpoint 1 test error: {te_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 test error: {te_err[-1]:.2f}%")
    summary_lines.append(f"    Max test error: {np.max(te_err):.2f}% at t={ts[np.argmax(te_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(te_err) - max(te_err[0], te_err[-1]):.2f}%")
    summary_lines.append("")

# Calculate statistics across runs
run1_barrier = np.max(run1_data['te_err']) - max(run1_data['te_err'][0], run1_data['te_err'][-1])
run2_barrier = np.max(run2_data['te_err']) - max(run2_data['te_err'][0], run2_data['te_err'][-1])
run3_barrier = np.max(run3_data['te_err']) - max(run3_data['te_err'][0], run3_data['te_err'][-1])

summary_lines.append("Cross-run statistics:")
summary_lines.append(f"  Mean barrier height: {np.mean([run1_barrier, run2_barrier, run3_barrier]):.2f}%")
summary_lines.append(f"  Std barrier height: {np.std([run1_barrier, run2_barrier, run3_barrier]):.2f}%")
summary_lines.append(f"  Min barrier height: {np.min([run1_barrier, run2_barrier, run3_barrier]):.2f}%")
summary_lines.append(f"  Max barrier height: {np.max([run1_barrier, run2_barrier, run3_barrier]):.2f}%")

summary_lines.append("="*70)

# Print to stdout
print("\n" + "\n".join(summary_lines))

# Save summary to text file
summary_path = args.output.replace('.png', '_summary.txt')
plotting.save_summary_text(summary_lines, summary_path)
