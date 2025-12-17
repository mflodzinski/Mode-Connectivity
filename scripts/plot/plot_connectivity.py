"""Plot linear interpolation vs Bezier curve connectivity.

Compares the loss landscape along linear path vs curved path between two endpoints.
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

parser = argparse.ArgumentParser(description='Plot connectivity comparison: regularized vs non-regularized curves')

# Custom arguments
parser.add_argument('--linear', type=str, required=True, metavar='PATH',
                    help='path to linear.npz results')
parser.add_argument('--curve-reg', type=str, default=None, metavar='PATH',
                    help='path to regularized curve.npz results (optional)')
parser.add_argument('--curve-noreg', type=str, default=None, metavar='PATH',
                    help='path to non-regularized curve.npz results (optional)')
parser.add_argument('--l2-evolution', type=str, default=None, metavar='PATH',
                    help='path to middle_point_l2_norms.npz (training evolution) - usually for reg curve')
parser.add_argument('--title', type=str, default='Mode Connectivity',
                    help='plot title')

# Standard arguments using ArgumentParserBuilder
ArgumentParserBuilder.add_plot_output_args(parser, required=False)

# Override default output
parser.set_defaults(output='results/vgg16/cifar10/curve/figures/connectivity_comparison.png')

args = parser.parse_args()

# Validate that at least one curve is provided
if args.curve_reg is None and args.curve_noreg is None:
    parser.error("At least one of --curve-reg or --curve-noreg must be provided")

# Load results
linear_data = np.load(args.linear)
curve_reg_data = np.load(args.curve_reg) if args.curve_reg is not None else None
curve_noreg_data = np.load(args.curve_noreg) if args.curve_noreg is not None else None

# Extract data - Linear
linear_ts = linear_data['ts']
linear_tr_err = linear_data['tr_err']
linear_tr_loss = linear_data['tr_loss']
linear_te_err = linear_data['te_err']
linear_te_loss = linear_data['te_loss']
linear_l2_norm = linear_data.get('l2_norm', None)

# Extract data - Curve with regularization (if provided)
if curve_reg_data is not None:
    curve_reg_ts = curve_reg_data['ts']
    curve_reg_tr_err = curve_reg_data['tr_err']
    curve_reg_tr_loss = curve_reg_data['tr_loss']
    curve_reg_te_err = curve_reg_data['te_err']
    curve_reg_te_loss = curve_reg_data['te_loss']
    curve_reg_l2_norm = curve_reg_data.get('l2_norm', None)
else:
    curve_reg_ts = curve_reg_tr_err = curve_reg_tr_loss = None
    curve_reg_te_err = curve_reg_te_loss = curve_reg_l2_norm = None

# Extract data - Curve without regularization (if provided)
if curve_noreg_data is not None:
    curve_noreg_ts = curve_noreg_data['ts']
    curve_noreg_tr_err = curve_noreg_data['tr_err']
    curve_noreg_tr_loss = curve_noreg_data['tr_loss']
    curve_noreg_te_err = curve_noreg_data['te_err']
    curve_noreg_te_loss = curve_noreg_data['te_loss']
    curve_noreg_l2_norm = curve_noreg_data.get('l2_norm', None)
else:
    curve_noreg_ts = curve_noreg_tr_err = curve_noreg_tr_loss = None
    curve_noreg_te_err = curve_noreg_te_loss = curve_noreg_l2_norm = None

# Load L2 evolution data if provided
l2_evolution_data = None
if args.l2_evolution is not None:
    l2_evolution_data = np.load(args.l2_evolution)

# Disable hardcoded L2 evolution loading (not relevant for single curve plots)
l2_evolution_noreg_data = None

# Create figure with 6 subplots (3x2 layout)
fig = plt.figure(figsize=(18, 18))
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)  # L2 norm along path
ax6 = plt.subplot(3, 2, 6)  # L2 norm training evolution

# Plot test error
ax1.plot(linear_ts, linear_te_err, 'o--', label='Linear',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
if curve_reg_te_err is not None:
    ax1.plot(curve_reg_ts, curve_reg_te_err, 's-', label='Curve (with reg)',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.8)
if curve_noreg_te_err is not None:
    ax1.plot(curve_noreg_ts, curve_noreg_te_err, 'D-', label='Curve (no reg)',
             color='#1f77b4', linewidth=2, markersize=4, alpha=0.8)
ax1.set_xlabel('t (interpolation parameter)', fontsize=12)
ax1.set_ylabel('Test Error (%)', fontsize=12)
ax1.set_title('Test Error along Path', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 1.05)

# Plot test loss
ax2.plot(linear_ts, linear_te_loss, 'o--', label='Linear',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
if curve_reg_te_loss is not None:
    ax2.plot(curve_reg_ts, curve_reg_te_loss, 's-', label='Curve (with reg)',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.8)
if curve_noreg_te_loss is not None:
    ax2.plot(curve_noreg_ts, curve_noreg_te_loss, 'D-', label='Curve (no reg)',
             color='#1f77b4', linewidth=2, markersize=4, alpha=0.8)
ax2.set_xlabel('t (interpolation parameter)', fontsize=12)
ax2.set_ylabel('Test Loss', fontsize=12)
ax2.set_title('Test Loss along Path', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 1.05)

# Plot train error
ax3.plot(linear_ts, linear_tr_err, 'o--', label='Linear',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
if curve_reg_tr_err is not None:
    ax3.plot(curve_reg_ts, curve_reg_tr_err, 's-', label='Curve (with reg)',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.8)
if curve_noreg_tr_err is not None:
    ax3.plot(curve_noreg_ts, curve_noreg_tr_err, 'D-', label='Curve (no reg)',
             color='#1f77b4', linewidth=2, markersize=4, alpha=0.8)
ax3.set_xlabel('t (interpolation parameter)', fontsize=12)
ax3.set_ylabel('Train Error (%)', fontsize=12)
ax3.set_title('Train Error along Path', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.05, 1.05)

# Plot train loss
ax4.plot(linear_ts, linear_tr_loss, 'o--', label='Linear',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
if curve_reg_tr_loss is not None:
    ax4.plot(curve_reg_ts, curve_reg_tr_loss, 's-', label='Curve (with reg)',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.8)
if curve_noreg_tr_loss is not None:
    ax4.plot(curve_noreg_ts, curve_noreg_tr_loss, 'D-', label='Curve (no reg)',
             color='#1f77b4', linewidth=2, markersize=4, alpha=0.8)
ax4.set_xlabel('t (interpolation parameter)', fontsize=12)
ax4.set_ylabel('Train Loss', fontsize=12)
ax4.set_title('Train Loss along Path', fontsize=13)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.05, 1.05)

# Plot L2 norm along path (if available)
if linear_l2_norm is not None:
    ax5.plot(linear_ts, linear_l2_norm, 'o--', label='Linear',
             color='#d62728', linewidth=2, markersize=4, alpha=0.7)
if curve_reg_l2_norm is not None:
    ax5.plot(curve_reg_ts, curve_reg_l2_norm, 's-', label='Curve (with reg)',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.8)
if curve_noreg_l2_norm is not None:
    ax5.plot(curve_noreg_ts, curve_noreg_l2_norm, 'D-', label='Curve (no reg)',
             color='#1f77b4', linewidth=2, markersize=4, alpha=0.8)

if linear_l2_norm is not None or curve_reg_l2_norm is not None or curve_noreg_l2_norm is not None:

    # Add vertical line at t=0.5
    ax5.axvline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax5.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax5.set_ylabel('L2 Norm of Weights', fontsize=12)
    ax5.set_title('L2 Norm along Path (Interpolated Model)', fontsize=13)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.05, 1.05)

# Plot L2 norm training evolution (if available)
if l2_evolution_data is not None or l2_evolution_noreg_data is not None:

    # Plot for the L2 evolution data loaded via argument (assuming Reg/Primary)
    if l2_evolution_data is not None:
        epochs = l2_evolution_data['epochs']
        raw_l2_norms = l2_evolution_data['l2_norms']

        ax6.plot(epochs, raw_l2_norms, linewidth=2, color='#2E86AB', marker='o',
                 markersize=3, markevery=max(1, len(epochs)//20),
                 label='Raw middle point ||w₁|| (Reg/Primary)')

    # NEW: Plot for the non-regularized L2 evolution data loaded via hardcoded path
    if l2_evolution_noreg_data is not None:
        epochs_noreg = l2_evolution_noreg_data['epochs']
        raw_l2_norms_noreg = l2_evolution_noreg_data['l2_norms']

        ax6.plot(epochs_noreg, raw_l2_norms_noreg, linewidth=2, color='#ff7f0e', marker='s',
                 markersize=3, markevery=max(1, len(epochs_noreg)//20),
                 label='Raw middle point ||w₁|| (No Reg)')

    # Add horizontal reference lines for comparison
    if linear_l2_norm is not None:
        linear_mid_idx = len(linear_ts) // 2
        ax6.axhline(linear_l2_norm[linear_mid_idx], color='#d62728', linestyle='--',
                   alpha=0.5, linewidth=1.5,
                   label=f'Linear t=0.5: {linear_l2_norm[linear_mid_idx]:.2f}')
    if curve_reg_l2_norm is not None:
        curve_reg_mid_idx = len(curve_reg_ts) // 2
        ax6.axhline(curve_reg_l2_norm[curve_reg_mid_idx], color='#2ca02c', linestyle='--',
                   alpha=0.5, linewidth=1.5,
                   label=f'Curve (reg) t=0.5: {curve_reg_l2_norm[curve_reg_mid_idx]:.2f}')
    if curve_noreg_l2_norm is not None:
        curve_noreg_mid_idx = len(curve_noreg_ts) // 2
        ax6.axhline(curve_noreg_l2_norm[curve_noreg_mid_idx], color='#1f77b4', linestyle='--',
                   alpha=0.5, linewidth=1.5,
                   label=f'Curve (no reg) t=0.5: {curve_noreg_l2_norm[curve_noreg_mid_idx]:.2f}')

    ax6.set_xlabel('Training Epoch', fontsize=12)
    ax6.set_ylabel('L2 Norm', fontsize=12)
    ax6.set_title('L2 Norm Evolution During Curve Training', fontsize=13)
    ax6.legend(fontsize=10, loc='best')
    ax6.grid(True, alpha=0.3)

plt.suptitle(args.title, fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plotting.save_figure(fig, args.output)

# Generate summary statistics text
summary_lines = []
summary_lines.append("="*70)
summary_lines.append("CONNECTIVITY COMPARISON SUMMARY")
summary_lines.append("="*70)
summary_lines.append("")

summary_lines.append("Linear Interpolation:")
summary_lines.append("  Train metrics:")
summary_lines.append(f"    Endpoint 1 train error: {linear_tr_err[0]:.2f}%")
summary_lines.append(f"    Endpoint 2 train error: {linear_tr_err[-1]:.2f}%")
summary_lines.append(f"    Max train error: {np.max(linear_tr_err):.2f}% at t={linear_ts[np.argmax(linear_tr_err)]:.3f}")
summary_lines.append(f"    Barrier height: {np.max(linear_tr_err) - max(linear_tr_err[0], linear_tr_err[-1]):.2f}%")
summary_lines.append("  Test metrics:")
summary_lines.append(f"    Endpoint 1 test error: {linear_te_err[0]:.2f}%")
summary_lines.append(f"    Endpoint 2 test error: {linear_te_err[-1]:.2f}%")
summary_lines.append(f"    Max test error: {np.max(linear_te_err):.2f}% at t={linear_ts[np.argmax(linear_te_err)]:.3f}")
summary_lines.append(f"    Barrier height: {np.max(linear_te_err) - max(linear_te_err[0], linear_te_err[-1]):.2f}%")
summary_lines.append("")

if curve_reg_tr_err is not None:
    summary_lines.append("Curve (with regularization):")
    summary_lines.append("  Train metrics:")
    summary_lines.append(f"    Endpoint 1 train error: {curve_reg_tr_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 train error: {curve_reg_tr_err[-1]:.2f}%")
    summary_lines.append(f"    Max train error: {np.max(curve_reg_tr_err):.2f}% at t={curve_reg_ts[np.argmax(curve_reg_tr_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(curve_reg_tr_err) - max(curve_reg_tr_err[0], curve_reg_tr_err[-1]):.2f}%")
    summary_lines.append("  Test metrics:")
    summary_lines.append(f"    Endpoint 1 test error: {curve_reg_te_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 test error: {curve_reg_te_err[-1]:.2f}%")
    summary_lines.append(f"    Max test error: {np.max(curve_reg_te_err):.2f}% at t={curve_reg_ts[np.argmax(curve_reg_te_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(curve_reg_te_err) - max(curve_reg_te_err[0], curve_reg_te_err[-1]):.2f}%")
    summary_lines.append("")

if curve_noreg_tr_err is not None:
    summary_lines.append("Curve (without regularization):")
    summary_lines.append("  Train metrics:")
    summary_lines.append(f"    Endpoint 1 train error: {curve_noreg_tr_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 train error: {curve_noreg_tr_err[-1]:.2f}%")
    summary_lines.append(f"    Max train error: {np.max(curve_noreg_tr_err):.2f}% at t={curve_noreg_ts[np.argmax(curve_noreg_tr_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(curve_noreg_tr_err) - max(curve_noreg_tr_err[0], curve_noreg_tr_err[-1]):.2f}%")
    summary_lines.append("  Test metrics:")
    summary_lines.append(f"    Endpoint 1 test error: {curve_noreg_te_err[0]:.2f}%")
    summary_lines.append(f"    Endpoint 2 test error: {curve_noreg_te_err[-1]:.2f}%")
    summary_lines.append(f"    Max test error: {np.max(curve_noreg_te_err):.2f}% at t={curve_noreg_ts[np.argmax(curve_noreg_te_err)]:.3f}")
    summary_lines.append(f"    Barrier height: {np.max(curve_noreg_te_err) - max(curve_noreg_te_err[0], curve_noreg_te_err[-1]):.2f}%")
    summary_lines.append("")

linear_barrier = np.max(linear_te_err) - max(linear_te_err[0], linear_te_err[-1])

if curve_reg_te_err is not None and curve_noreg_te_err is not None:
    curve_reg_barrier = np.max(curve_reg_te_err) - max(curve_reg_te_err[0], curve_reg_te_err[-1])
    curve_noreg_barrier = np.max(curve_noreg_te_err) - max(curve_noreg_te_err[0], curve_noreg_te_err[-1])
    barrier_reduction_reg = linear_barrier - curve_reg_barrier
    barrier_reduction_noreg = linear_barrier - curve_noreg_barrier
    summary_lines.append(f"Barrier reduction by curve (with reg): {barrier_reduction_reg:.2f}%")
    summary_lines.append(f"Barrier reduction by curve (no reg): {barrier_reduction_noreg:.2f}%")
    summary_lines.append(f"Difference (reg - noreg): {curve_reg_barrier - curve_noreg_barrier:.2f}%")
    summary_lines.append("")
    if barrier_reduction_reg > 1.0 or barrier_reduction_noreg > 1.0:
        summary_lines.append("✓ Mode connectivity confirmed: Curves significantly reduce barrier")
    elif barrier_reduction_reg > 0.1 or barrier_reduction_noreg > 0.1:
        summary_lines.append("✓ Partial mode connectivity: Curves reduce barrier")
    else:
        summary_lines.append("✗ Limited mode connectivity: Barrier remains high")
elif curve_reg_te_err is not None:
    curve_reg_barrier = np.max(curve_reg_te_err) - max(curve_reg_te_err[0], curve_reg_te_err[-1])
    barrier_reduction = linear_barrier - curve_reg_barrier
    summary_lines.append(f"Barrier reduction by curve (with reg): {barrier_reduction:.2f}%")
    summary_lines.append("")
    if barrier_reduction > 1.0:
        summary_lines.append("✓ Mode connectivity confirmed: Curve significantly reduces barrier")
    elif barrier_reduction > 0.1:
        summary_lines.append("✓ Partial mode connectivity: Curve reduces barrier")
    else:
        summary_lines.append("✗ Limited mode connectivity: Barrier remains high")
elif curve_noreg_te_err is not None:
    curve_noreg_barrier = np.max(curve_noreg_te_err) - max(curve_noreg_te_err[0], curve_noreg_te_err[-1])
    barrier_reduction = linear_barrier - curve_noreg_barrier
    summary_lines.append(f"Barrier reduction by curve (no reg): {barrier_reduction:.2f}%")
    summary_lines.append("")
    if barrier_reduction > 1.0:
        summary_lines.append("✓ Mode connectivity confirmed: Curve significantly reduces barrier")
    elif barrier_reduction > 0.1:
        summary_lines.append("✓ Partial mode connectivity: Curve reduces barrier")
    else:
        summary_lines.append("✗ Limited mode connectivity: Barrier remains high")

summary_lines.append("="*70)

# Print to stdout
print("\n" + "\n".join(summary_lines))

# Save summary to text file
summary_path = args.output.replace('.png', '_summary.txt')
plotting.save_summary_text(summary_lines, summary_path)