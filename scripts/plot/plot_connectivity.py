"""Plot linear interpolation vs Bezier curve connectivity.

Compares the loss landscape along linear path vs curved path between two endpoints.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Plot connectivity comparison')
parser.add_argument('--linear', type=str, required=True, metavar='PATH',
                    help='path to linear.npz results')
parser.add_argument('--curve', type=str, required=True, metavar='PATH',
                    help='path to curve.npz results')
parser.add_argument('--l2_evolution', type=str, default=None, metavar='PATH',
                    help='path to middle_point_l2_norms.npz (training evolution)')
parser.add_argument('--output', type=str, default='results/vgg16/cifar10/curve/figures/connectivity_comparison.png',
                    help='output figure path')
parser.add_argument('--title', type=str, default='Mode Connectivity: Linear vs Bezier Curve',
                    help='plot title')

args = parser.parse_args()

# Load results
linear_data = np.load(args.linear)
curve_data = np.load(args.curve)

# Extract data
linear_ts = linear_data['ts']
linear_tr_err = linear_data['tr_err']
linear_tr_loss = linear_data['tr_loss']
linear_te_err = linear_data['te_err']
linear_te_loss = linear_data['te_loss']
linear_l2_norm = linear_data.get('l2_norm', None)

curve_ts = curve_data['ts']
curve_tr_err = curve_data['tr_err']
curve_tr_loss = curve_data['tr_loss']
curve_te_err = curve_data['te_err']
curve_te_loss = curve_data['te_loss']
curve_l2_norm = curve_data.get('l2_norm', None)

# Load L2 evolution data if provided
l2_evolution_data = None
if args.l2_evolution is not None:
    l2_evolution_data = np.load(args.l2_evolution)

# Create figure with 6 subplots (3x2 layout)
fig = plt.figure(figsize=(18, 18))
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)  # L2 norm along path
ax6 = plt.subplot(3, 2, 6)  # L2 norm training evolution

# Plot test error
ax1.plot(linear_ts, linear_te_err, 'o-', label='Linear Interpolation',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
ax1.plot(curve_ts, curve_te_err, 's-', label='Bezier Curve',
         color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)
ax1.set_xlabel('t (interpolation parameter)', fontsize=12)
ax1.set_ylabel('Test Error (%)', fontsize=12)
ax1.set_title('Test Error along Path', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 1.05)

# Plot test loss
ax2.plot(linear_ts, linear_te_loss, 'o-', label='Linear Interpolation',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
ax2.plot(curve_ts, curve_te_loss, 's-', label='Bezier Curve',
         color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)
ax2.set_xlabel('t (interpolation parameter)', fontsize=12)
ax2.set_ylabel('Test Loss', fontsize=12)
ax2.set_title('Test Loss along Path', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 1.05)

# Plot train error
ax3.plot(linear_ts, linear_tr_err, 'o-', label='Linear Interpolation',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
ax3.plot(curve_ts, curve_tr_err, 's-', label='Bezier Curve',
         color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)
ax3.set_xlabel('t (interpolation parameter)', fontsize=12)
ax3.set_ylabel('Train Error (%)', fontsize=12)
ax3.set_title('Train Error along Path', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.05, 1.05)

# Plot train loss
ax4.plot(linear_ts, linear_tr_loss, 'o-', label='Linear Interpolation',
         color='#d62728', linewidth=2, markersize=4, alpha=0.7)
ax4.plot(curve_ts, curve_tr_loss, 's-', label='Bezier Curve',
         color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)
ax4.set_xlabel('t (interpolation parameter)', fontsize=12)
ax4.set_ylabel('Train Loss', fontsize=12)
ax4.set_title('Train Loss along Path', fontsize=13)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.05, 1.05)

# Plot L2 norm along path (if available)
if linear_l2_norm is not None and curve_l2_norm is not None:
    ax5.plot(linear_ts, linear_l2_norm, 'o-', label='Linear Interpolation',
             color='#d62728', linewidth=2, markersize=4, alpha=0.7)
    ax5.plot(curve_ts, curve_l2_norm, 's-', label='Bezier Curve',
             color='#2ca02c', linewidth=2, markersize=4, alpha=0.7)

    # Add annotations for middle points
    linear_mid_idx = len(linear_ts) // 2
    curve_mid_idx = len(curve_ts) // 2
    ax5.axvline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax5.plot(linear_ts[linear_mid_idx], linear_l2_norm[linear_mid_idx], 'o',
             color='#d62728', markersize=8, markeredgewidth=2, markerfacecolor='none',
             label=f'Linear t=0.5: {linear_l2_norm[linear_mid_idx]:.2f}')
    ax5.plot(curve_ts[curve_mid_idx], curve_l2_norm[curve_mid_idx], 's',
             color='#2ca02c', markersize=8, markeredgewidth=2, markerfacecolor='none',
             label=f'Bezier t=0.5: {curve_l2_norm[curve_mid_idx]:.2f}')

    ax5.set_xlabel('t (interpolation parameter)', fontsize=12)
    ax5.set_ylabel('L2 Norm of Weights', fontsize=12)
    ax5.set_title('L2 Norm along Path (Interpolated Model)', fontsize=13)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.05, 1.05)

# Plot L2 norm training evolution (if available)
if l2_evolution_data is not None:
    epochs = l2_evolution_data['epochs']
    raw_l2_norms = l2_evolution_data['l2_norms']
    interpolated_l2_norms = l2_evolution_data.get('interpolated_l2_norms', None)

    ax6.plot(epochs, raw_l2_norms, linewidth=2, color='#2E86AB', marker='o',
             markersize=3, markevery=max(1, len(epochs)//20),
             label='Raw middle point ||w₁||')

    if interpolated_l2_norms is not None:
        ax6.plot(epochs, interpolated_l2_norms, linewidth=2, color='#A23B72', marker='s',
                 markersize=3, markevery=max(1, len(epochs)//20),
                 label='Interpolated at t=0.5')

        # Highlight connection to path L2 norms
        if linear_l2_norm is not None and curve_l2_norm is not None:
            linear_mid_idx = len(linear_ts) // 2
            curve_mid_idx = len(curve_ts) // 2

            # Add horizontal lines showing where training started/ended
            ax6.axhline(linear_l2_norm[linear_mid_idx], color='#d62728', linestyle='--',
                       alpha=0.5, linewidth=1.5,
                       label=f'Linear t=0.5: {linear_l2_norm[linear_mid_idx]:.2f}')
            ax6.axhline(curve_l2_norm[curve_mid_idx], color='#2ca02c', linestyle='--',
                       alpha=0.5, linewidth=1.5,
                       label=f'Bezier t=0.5: {curve_l2_norm[curve_mid_idx]:.2f}')

    ax6.set_xlabel('Training Epoch', fontsize=12)
    ax6.set_ylabel('L2 Norm', fontsize=12)
    ax6.set_title('L2 Norm Evolution During Curve Training', fontsize=13)
    ax6.legend(fontsize=10, loc='best')
    ax6.grid(True, alpha=0.3)

plt.suptitle(args.title, fontsize=14, fontweight='bold')
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Save figure
plt.savefig(args.output, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {args.output}")

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

summary_lines.append("Bezier Curve:")
summary_lines.append("  Train metrics:")
summary_lines.append(f"    Endpoint 1 train error: {curve_tr_err[0]:.2f}%")
summary_lines.append(f"    Endpoint 2 train error: {curve_tr_err[-1]:.2f}%")
summary_lines.append(f"    Max train error: {np.max(curve_tr_err):.2f}% at t={curve_ts[np.argmax(curve_tr_err)]:.3f}")
summary_lines.append(f"    Barrier height: {np.max(curve_tr_err) - max(curve_tr_err[0], curve_tr_err[-1]):.2f}%")
summary_lines.append("  Test metrics:")
summary_lines.append(f"    Endpoint 1 test error: {curve_te_err[0]:.2f}%")
summary_lines.append(f"    Endpoint 2 test error: {curve_te_err[-1]:.2f}%")
summary_lines.append(f"    Max test error: {np.max(curve_te_err):.2f}% at t={curve_ts[np.argmax(curve_te_err)]:.3f}")
summary_lines.append(f"    Barrier height: {np.max(curve_te_err) - max(curve_te_err[0], curve_te_err[-1]):.2f}%")
summary_lines.append("")

barrier_reduction = (np.max(linear_te_err) - max(linear_te_err[0], linear_te_err[-1])) - \
                    (np.max(curve_te_err) - max(curve_te_err[0], curve_te_err[-1]))
summary_lines.append(f"Barrier reduction by Bezier curve: {barrier_reduction:.2f}%")
summary_lines.append("")

if barrier_reduction > 1.0:
    summary_lines.append("✓ Mode connectivity confirmed: Bezier curve significantly reduces barrier")
elif barrier_reduction > 0.1:
    summary_lines.append("✓ Partial mode connectivity: Bezier curve reduces barrier")
else:
    summary_lines.append("✗ Limited mode connectivity: Barrier remains high")

summary_lines.append("="*70)

# Print to stdout
print("\n" + "\n".join(summary_lines))

# Save summary to text file
summary_path = args.output.replace('.png', '_summary.txt')
with open(summary_path, 'w') as f:
    f.write("\n".join(summary_lines))
print(f"\nSummary saved to: {summary_path}")
