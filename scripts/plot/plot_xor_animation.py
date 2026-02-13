"""
Create interactive HTML animations for XOR decision boundary interpolation.

For each pair of aligned networks, generates 51 frames (t=0.00, 0.02, ..., 1.00)
showing how the decision boundary evolves during linear interpolation.

Output: Separate HTML files per pair with slider, play/pause, and speed controls.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)


# =============================================================================
# XOR Dataset and Model (mirrors xor_experiment.py for standalone use)
# =============================================================================
XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class SimpleMLP(nn.Module):
    """Simple MLP for XOR: 2 inputs -> H hidden -> 1 output (logit)."""

    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def apply_permutation_to_state(state, perm):
    """Apply a permutation to the hidden layer of a state dict."""
    perm = list(perm)
    new_state = OrderedDict()
    new_state['fc1.weight'] = state['fc1.weight'][perm, :]
    new_state['fc1.bias'] = state['fc1.bias'][perm]
    new_state['fc2.weight'] = state['fc2.weight'][:, perm]
    new_state['fc2.bias'] = state['fc2.bias'].clone()
    return new_state


def evaluate_model(model):
    """Evaluate model on XOR dataset."""
    model.eval()
    with torch.no_grad():
        outputs = model(XOR_DATA)
        loss = F.binary_cross_entropy_with_logits(outputs, XOR_LABELS).item()
        pred = (torch.sigmoid(outputs) >= 0.5).long()
        accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100
    return {'loss': loss, 'accuracy': accuracy}


# =============================================================================
# Animation Generation
# =============================================================================
def generate_boundary_data(state_a, state_b, t, hidden_size, grid_resolution=100):
    """Generate decision boundary data for interpolation at t.

    Returns:
        xx, yy: meshgrid coordinates
        Z: prediction grid (0 or 1)
        accuracy: accuracy at this interpolation point
    """
    # Interpolate weights
    interp_state = OrderedDict()
    for key in state_a:
        interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]

    # Load into model
    model = SimpleMLP(hidden_size=hidden_size)
    model.load_state_dict(interp_state)
    model.eval()

    # Create grid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(grid_points)
        preds = (torch.sigmoid(outputs) >= 0.5).long().squeeze(1).numpy()

    Z = preds.reshape(xx.shape)

    # Evaluate accuracy
    res = evaluate_model(model)
    accuracy = res['accuracy']

    return xx, yy, Z, accuracy


def create_pair_animation(state_a, state_b_aligned, seed_a, seed_b, hidden_size, num_frames=51):
    """Create Plotly animation for one pair.

    Args:
        state_a: State dict for model A
        state_b_aligned: State dict for model B (after alignment)
        seed_a: Seed of model A
        seed_b: Seed of model B
        num_frames: Number of animation frames (default: 51 for t=0.00, 0.02, ..., 1.00)

    Returns:
        Plotly Figure object with animation
    """
    ts = np.linspace(0, 1, num_frames)

    # XOR point colors and data
    xor_x = XOR_DATA[:, 0].numpy()
    xor_y = XOR_DATA[:, 1].numpy()
    xor_labels = XOR_LABELS.long().squeeze(1).numpy()
    point_colors = ['#d62728' if l == 0 else '#2ca02c' for l in xor_labels]

    # Generate all frame data first
    frame_data = []
    for t in ts:
        xx, yy, Z, acc = generate_boundary_data(state_a, state_b_aligned, t, hidden_size)
        frame_data.append((xx, yy, Z, acc))

    # Create initial frame data
    xx, yy, Z, acc = frame_data[0]

    # Create figure with initial data
    fig = go.Figure()

    # Add contour for decision boundary (class 0 region)
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            showlines=True,
            coloring='fill'
        ),
        line=dict(width=3, color='black'),
        hoverinfo='skip',
        name='Decision Boundary'
    ))

    # Add XOR data points
    fig.add_trace(go.Scatter(
        x=xor_x,
        y=xor_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=point_colors,
            line=dict(width=2, color='black')
        ),
        text=[f'({int(x)},{int(y)})={l}' for x, y, l in zip(xor_x, xor_y, xor_labels)],
        textposition='top right',
        textfont=dict(size=12),
        hoverinfo='text',
        hovertext=[f'Input: ({int(x)},{int(y)})<br>Label: {l}' for x, y, l in zip(xor_x, xor_y, xor_labels)],
        name='XOR Points'
    ))

    # Create frames for animation
    frames = []
    for i, t in enumerate(ts):
        xx, yy, Z, acc = frame_data[i]

        frame = go.Frame(
            data=[
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    showscale=False,
                    colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=1,
                        showlines=True,
                        coloring='fill'
                    ),
                    line=dict(width=3, color='black'),
                    hoverinfo='skip'
                ),
                go.Scatter(
                    x=xor_x,
                    y=xor_y,
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=point_colors,
                        line=dict(width=2, color='black')
                    ),
                    text=[f'({int(x)},{int(y)})={l}' for x, y, l in zip(xor_x, xor_y, xor_labels)],
                    textposition='top right',
                    textfont=dict(size=12),
                    hoverinfo='text',
                    hovertext=[f'Input: ({int(x)},{int(y)})<br>Label: {l}' for x, y, l in zip(xor_x, xor_y, xor_labels)]
                )
            ],
            name=f't={t:.2f}',
            layout=go.Layout(
                title=dict(
                    text=f'Interpolation: seed{seed_a} → seed{seed_b} (aligned)<br>t = {t:.2f} | Accuracy = {acc:.0f}%',
                    font=dict(size=16)
                )
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Create slider
    sliders = [dict(
        active=0,
        yanchor='top',
        xanchor='left',
        currentvalue=dict(
            font=dict(size=14),
            prefix='t = ',
            visible=True,
            xanchor='center'
        ),
        transition=dict(duration=50, easing='cubic-in-out'),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        y=0,
        steps=[
            dict(
                args=[[f't={t:.2f}'],
                      dict(mode='immediate',
                           frame=dict(duration=50, redraw=True),
                           transition=dict(duration=50))],
                label=f'{t:.2f}',
                method='animate'
            )
            for t in ts
        ]
    )]

    # Create play/pause buttons with speed control
    updatemenus = [
        dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.0,
            xanchor='left',
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=100, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=50, easing='quadratic-in-out'),
                        mode='immediate'
                    )]
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate',
                        transition=dict(duration=0)
                    )]
                ),
                dict(
                    label='⏪ Reset',
                    method='animate',
                    args=[['t=0.00'], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0)
                    )]
                )
            ]
        ),
        # Speed control dropdown
        dict(
            type='dropdown',
            showactive=True,
            y=1.15,
            x=0.35,
            xanchor='left',
            buttons=[
                dict(
                    label='0.5x Speed',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=200, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=100)
                    )]
                ),
                dict(
                    label='1x Speed',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=100, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=50)
                    )]
                ),
                dict(
                    label='2x Speed',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=50, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=25)
                    )]
                ),
                dict(
                    label='4x Speed',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=25, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=10)
                    )]
                )
            ]
        )
    ]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interpolation: seed{seed_a} → seed{seed_b} (aligned)<br>t = 0.00 | Accuracy = {frame_data[0][3]:.0f}%',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='x₁',
            range=[-0.5, 1.5],
            scaleanchor='y',
            scaleratio=1,
            dtick=0.5
        ),
        yaxis=dict(
            title='x₂',
            range=[-0.5, 1.5],
            dtick=0.5
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        width=700,
        height=700,
        showlegend=False,
        margin=dict(t=120, b=80)
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Create XOR interpolation animations')
    parser.add_argument('--input', type=str, default='results/xor_2h',
                        help='Input directory with results.json and checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for animations (default: input/animations)')
    parser.add_argument('--num-frames', type=int, default=51,
                        help='Number of frames (default: 51 for t=0.00, 0.02, ..., 1.00)')
    args = parser.parse_args()

    # Resolve paths
    input_dir = os.path.join(project_root, args.input)
    if args.output is None:
        output_dir = os.path.join(input_dir, 'animations')
    else:
        output_dir = os.path.join(project_root, args.output)

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results_path = os.path.join(input_dir, 'results.json')
    print(f"Loading results from {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Read hidden size from results.json (with fallback)
    config = results.get('config', {})
    hidden_size = config.get('hidden_size', results.get('hidden_size'))

    # Load checkpoints
    checkpoints_dir = os.path.join(input_dir, 'checkpoints')
    seeds = results['trained_seeds']
    models = {}

    print(f"Loading {len(seeds)} checkpoints...")
    for seed in seeds:
        checkpoint_path = os.path.join(checkpoints_dir, f'seed{seed}.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        models[seed] = checkpoint['model_state']
        if hidden_size is None:
            hidden_size = models[seed]['fc1.weight'].shape[0]
        print(f"  Loaded seed {seed}")
    if hidden_size is None:
        raise ValueError("Hidden size not found in results.json and could not be inferred from checkpoints.")

    # Process each pair
    pair_results = results['pair_results']
    print(f"\nGenerating animations for {len(pair_results)} pairs...")

    for pr in pair_results:
        seed_a = pr['seed_a']
        seed_b = pr['seed_b']
        best_perm = pr['best_permutation']

        print(f"\nPair ({seed_a}, {seed_b}):")
        print(f"  Permutation: {best_perm}")
        print(f"  Barrier after alignment: {pr['barrier_after']['barrier']:.1f}%")

        # Get states
        state_a = models[seed_a]
        state_b = models[seed_b]

        # Apply permutation to model B
        state_b_aligned = apply_permutation_to_state(state_b, best_perm)

        # Create animation
        print(f"  Generating {args.num_frames} frames...")
        fig = create_pair_animation(state_a, state_b_aligned, seed_a, seed_b, hidden_size, args.num_frames)

        # Save HTML
        output_path = os.path.join(output_dir, f'pair_{seed_a}_{seed_b}.html')
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"  Saved to {output_path}")

    print(f"\n{'=' * 60}")
    print(f"All animations saved to: {output_dir}")
    print(f"Open any HTML file in a browser to view the interactive animation.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
