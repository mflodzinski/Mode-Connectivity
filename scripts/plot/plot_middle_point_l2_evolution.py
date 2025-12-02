"""Plot the evolution of L2 norm of the middle point during curve training."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Plot L2 norm evolution of middle point during curve training')
parser.add_argument('--data', type=str, required=True, metavar='PATH',
                    help='path to middle_point_l2_norms.npz file')
parser.add_argument('--output', type=str, default=None, metavar='PATH',
                    help='output path for figure (default: same directory as data)')
parser.add_argument('--title', type=str, default=None,
                    help='custom title for the plot')
parser.add_argument('--dpi', type=int, default=300,
                    help='DPI for saved figure (default: 300)')

args = parser.parse_args()

# Load data
data = np.load(args.data)
epochs = data['epochs']
l2_norms = data['l2_norms']
interpolated_l2_norms = data.get('interpolated_l2_norms', None)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot L2 norm evolution - raw middle point parameters
ax.plot(epochs, l2_norms, linewidth=2, color='#2E86AB', marker='o',
        markersize=3, markevery=max(1, len(epochs)//20),
        label='Raw middle point ||w₁||')

# Plot interpolated L2 norm if available
if interpolated_l2_norms is not None:
    ax.plot(epochs, interpolated_l2_norms, linewidth=2, color='#A23B72', marker='s',
            markersize=3, markevery=max(1, len(epochs)//20),
            label='Interpolated at t=0.5')

# Styling
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('L2 Norm', fontsize=12)
ax.set_title(args.title or 'Evolution of L2 Norms During Curve Training',
             fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best', fontsize=10, framealpha=0.9)

# Add statistics text box
stats_text = f'Raw middle point ||w₁||:\n'
stats_text += f'  Initial: {l2_norms[0]:.2f}\n'
stats_text += f'  Final: {l2_norms[-1]:.2f}\n'
stats_text += f'  Min: {np.min(l2_norms):.2f} (epoch {epochs[np.argmin(l2_norms)]})\n'
stats_text += f'  Max: {np.max(l2_norms):.2f} (epoch {epochs[np.argmax(l2_norms)]})'

if interpolated_l2_norms is not None:
    stats_text += f'\n\nInterpolated at t=0.5:\n'
    stats_text += f'  Initial: {interpolated_l2_norms[0]:.2f}\n'
    stats_text += f'  Final: {interpolated_l2_norms[-1]:.2f}\n'
    stats_text += f'  Min: {np.min(interpolated_l2_norms):.2f} (epoch {epochs[np.argmin(interpolated_l2_norms)]})\n'
    stats_text += f'  Max: {np.max(interpolated_l2_norms):.2f} (epoch {epochs[np.argmax(interpolated_l2_norms)]})'

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()

# Determine output path
if args.output is None:
    data_dir = os.path.dirname(args.data)
    figures_dir = data_dir.replace('/evaluations', '/figures')
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, 'middle_point_l2_evolution.png')
else:
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

# Save figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f'Figure saved to: {output_path}')

# Show plot
plt.show()
