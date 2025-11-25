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

curve_ts = curve_data['ts']
curve_tr_err = curve_data['tr_err']
curve_tr_loss = curve_data['tr_loss']
curve_te_err = curve_data['te_err']
curve_te_loss = curve_data['te_loss']

# Create figure with 4 subplots (2x2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

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

plt.suptitle(args.title, fontsize=14, fontweight='bold')
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Save figure
plt.savefig(args.output, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {args.output}")

# Print summary statistics
print("\n" + "="*70)
print("CONNECTIVITY COMPARISON SUMMARY")
print("="*70)

print(f"\nLinear Interpolation:")
print(f"  Train metrics:")
print(f"    Endpoint 1 train error: {linear_tr_err[0]:.2f}%")
print(f"    Endpoint 2 train error: {linear_tr_err[-1]:.2f}%")
print(f"    Max train error: {np.max(linear_tr_err):.2f}% at t={linear_ts[np.argmax(linear_tr_err)]:.3f}")
print(f"    Barrier height: {np.max(linear_tr_err) - max(linear_tr_err[0], linear_tr_err[-1]):.2f}%")
print(f"  Test metrics:")
print(f"    Endpoint 1 test error: {linear_te_err[0]:.2f}%")
print(f"    Endpoint 2 test error: {linear_te_err[-1]:.2f}%")
print(f"    Max test error: {np.max(linear_te_err):.2f}% at t={linear_ts[np.argmax(linear_te_err)]:.3f}")
print(f"    Barrier height: {np.max(linear_te_err) - max(linear_te_err[0], linear_te_err[-1]):.2f}%")

print(f"\nBezier Curve:")
print(f"  Train metrics:")
print(f"    Endpoint 1 train error: {curve_tr_err[0]:.2f}%")
print(f"    Endpoint 2 train error: {curve_tr_err[-1]:.2f}%")
print(f"    Max train error: {np.max(curve_tr_err):.2f}% at t={curve_ts[np.argmax(curve_tr_err)]:.3f}")
print(f"    Barrier height: {np.max(curve_tr_err) - max(curve_tr_err[0], curve_tr_err[-1]):.2f}%")
print(f"  Test metrics:")
print(f"    Endpoint 1 test error: {curve_te_err[0]:.2f}%")
print(f"    Endpoint 2 test error: {curve_te_err[-1]:.2f}%")
print(f"    Max test error: {np.max(curve_te_err):.2f}% at t={curve_ts[np.argmax(curve_te_err)]:.3f}")
print(f"    Barrier height: {np.max(curve_te_err) - max(curve_te_err[0], curve_te_err[-1]):.2f}%")

barrier_reduction = (np.max(linear_te_err) - max(linear_te_err[0], linear_te_err[-1])) - \
                    (np.max(curve_te_err) - max(curve_te_err[0], curve_te_err[-1]))
print(f"\nBarrier reduction by Bezier curve: {barrier_reduction:.2f}%")

if barrier_reduction > 1.0:
    print("\n✓ Mode connectivity confirmed: Bezier curve significantly reduces barrier")
elif barrier_reduction > 0.1:
    print("\n✓ Partial mode connectivity: Bezier curve reduces barrier")
else:
    print("\n✗ Limited mode connectivity: Barrier remains high")

print("="*70)
