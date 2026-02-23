"""
Create interactive boundary sliders for XOR curve paths saved as curve.npz.

Supports:
1) Single file mode: --curve-npz path/to/curve.npz
2) Batch mode: --evaluations-dir results/xor/evaluations --path-type linear|bezier|polychain|all
"""

import argparse
import glob
import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


def infer_hidden_size(num_params):
    """Infer hidden size for 2-H-1 MLP from flattened parameter vector length."""
    if (num_params - 1) % 4 != 0:
        raise ValueError(
            f"Cannot infer hidden size from {num_params} parameters. "
            "Expected size = 4*H + 1."
        )
    hidden_size = (num_params - 1) // 4
    if hidden_size <= 0:
        raise ValueError(f"Invalid inferred hidden size: {hidden_size}")
    return hidden_size


def logits_from_param_vector(x, param_vector, hidden_size):
    """Forward pass of XOR model directly from flattened parameter vector."""
    idx = 0

    fc1_weight_size = hidden_size * 2
    fc1_weight = param_vector[idx:idx + fc1_weight_size].view(hidden_size, 2)
    idx += fc1_weight_size

    fc1_bias = param_vector[idx:idx + hidden_size]
    idx += hidden_size

    fc2_weight = param_vector[idx:idx + hidden_size].view(1, hidden_size)
    idx += hidden_size

    fc2_bias = param_vector[idx:idx + 1]

    hidden = torch.relu(x @ fc1_weight.t() + fc1_bias)
    return hidden @ fc2_weight.t() + fc2_bias


def compute_boundary_from_vector(param_vector, hidden_size, grid_resolution=140):
    """Compute decision boundary grid and XOR accuracy for one parameter vector."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    params = torch.tensor(param_vector, dtype=torch.float32)

    with torch.no_grad():
        grid_logits = logits_from_param_vector(grid_points, params, hidden_size)
        grid_pred = (torch.sigmoid(grid_logits) >= 0.5).long().squeeze(1).numpy()

        logits = logits_from_param_vector(XOR_DATA, params, hidden_size)
        loss = F.binary_cross_entropy_with_logits(logits, XOR_LABELS).item()
        pred = (torch.sigmoid(logits) >= 0.5).long()
        accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100.0

    z = grid_pred.reshape(xx.shape)
    return xx, yy, z, accuracy, loss


def choose_snapshot_indices(num_points, num_snapshots):
    """Choose evenly spaced snapshot indices."""
    num_snapshots = max(2, min(num_snapshots, num_points))
    idx = np.linspace(0, num_points - 1, num_snapshots, dtype=int)
    return np.unique(idx)


def create_slider_figure(ts, param_vectors, num_snapshots, grid_resolution, title, te_acc=None, te_loss=None):
    """Create interactive Plotly slider figure."""
    hidden_size = infer_hidden_size(param_vectors.shape[1])
    snapshot_idx = choose_snapshot_indices(len(ts), num_snapshots)

    frame_data = []
    for i in snapshot_idx:
        xx, yy, z, acc_eval, loss_eval = compute_boundary_from_vector(
            param_vectors[i], hidden_size, grid_resolution
        )
        acc = float(te_acc[i]) if te_acc is not None and len(te_acc) == len(ts) else acc_eval
        loss = float(te_loss[i]) if te_loss is not None and len(te_loss) == len(ts) else loss_eval
        frame_data.append({
            'idx': int(i),
            't': float(ts[i]),
            'xx': xx,
            'yy': yy,
            'z': z,
            'acc': acc,
            'loss': loss,
        })

    xor_x = XOR_DATA[:, 0].numpy()
    xor_y = XOR_DATA[:, 1].numpy()
    xor_labels = XOR_LABELS.long().squeeze(1).numpy()
    point_colors = ['#d62728' if label == 0 else '#2ca02c' for label in xor_labels]

    first = frame_data[0]
    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=first['xx'][0],
        y=first['yy'][:, 0],
        z=first['z'],
        showscale=False,
        colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
        contours=dict(start=0.5, end=0.5, size=1, showlines=True, coloring='fill'),
        line=dict(width=3, color='black'),
        hoverinfo='skip',
        name='Decision Boundary',
    ))

    fig.add_trace(go.Scatter(
        x=xor_x,
        y=xor_y,
        mode='markers+text',
        marker=dict(size=20, color=point_colors, line=dict(width=2, color='black')),
        text=[f'({int(x)},{int(y)})={lbl}' for x, y, lbl in zip(xor_x, xor_y, xor_labels)],
        textposition='top right',
        hoverinfo='skip',
        name='XOR Points',
    ))

    frames = []
    for frame in frame_data:
        frame_name = f"idx={frame['idx']}"
        frames.append(go.Frame(
            data=[
                go.Contour(
                    x=frame['xx'][0],
                    y=frame['yy'][:, 0],
                    z=frame['z'],
                    showscale=False,
                    colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
                    contours=dict(start=0.5, end=0.5, size=1, showlines=True, coloring='fill'),
                    line=dict(width=3, color='black'),
                    hoverinfo='skip',
                ),
                go.Scatter(
                    x=xor_x,
                    y=xor_y,
                    mode='markers+text',
                    marker=dict(size=20, color=point_colors, line=dict(width=2, color='black')),
                    text=[f'({int(x)},{int(y)})={lbl}' for x, y, lbl in zip(xor_x, xor_y, xor_labels)],
                    textposition='top right',
                    hoverinfo='skip',
                ),
            ],
            name=frame_name,
            layout=go.Layout(
                title=dict(
                    text=(
                        f"{title}<br>"
                        f"t={frame['t']:.3f} | test acc={frame['acc']:.1f}% | test loss={frame['loss']:.4f}"
                    ),
                    font=dict(size=16),
                )
            ),
        ))
    fig.frames = frames

    slider_steps = []
    for frame in frame_data:
        slider_steps.append(dict(
            args=[
                [f"idx={frame['idx']}"],
                dict(mode='immediate', frame=dict(duration=80, redraw=True), transition=dict(duration=30)),
            ],
            label=f"{frame['t']:.2f}",
            method='animate',
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"t={first['t']:.3f} | test acc={first['acc']:.1f}% | test loss={first['loss']:.4f}"
            ),
            x=0.5,
            xanchor='center',
            font=dict(size=16),
        ),
        xaxis=dict(title='x1', range=[-0.5, 1.5], scaleanchor='y', scaleratio=1, dtick=0.5),
        yaxis=dict(title='x2', range=[-0.5, 1.5], dtick=0.5),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='t = '),
            pad=dict(b=10, t=50),
            steps=slider_steps,
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.0,
            xanchor='left',
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)],
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')],
                ),
            ],
        )],
        width=720,
        height=720,
        showlegend=False,
        margin=dict(t=120, b=80),
    )

    return fig


def load_npz_for_animation(npz_path):
    """Load NPZ and validate required fields."""
    data = np.load(npz_path)
    required = ['ts', 'param_vectors']
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"{npz_path} is missing keys: {missing}. "
            "Re-run xor_experiment.py with the updated exporter."
        )
    ts = data['ts']
    param_vectors = data['param_vectors']
    te_acc = data['te_acc'] if 'te_acc' in data else None
    te_loss = data['te_loss'] if 'te_loss' in data else None
    return ts, param_vectors, te_acc, te_loss


def process_single(npz_path, output_path, num_snapshots, grid_resolution, title):
    """Generate one HTML animation from one curve.npz."""
    ts, param_vectors, te_acc, te_loss = load_npz_for_animation(npz_path)
    fig = create_slider_figure(
        ts=ts,
        param_vectors=param_vectors,
        num_snapshots=num_snapshots,
        grid_resolution=grid_resolution,
        title=title,
        te_acc=te_acc,
        te_loss=te_loss,
    )
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved: {output_path}")


def collect_batch_tasks(evaluations_dir, path_type):
    """Collect (pair_name, path_name, npz_path) tasks from evaluations dir."""
    # "all" intentionally excludes alignment-specific legacy paths.
    all_default_paths = ['linear', 'bezier', 'polychain']
    selected_paths = all_default_paths if path_type == 'all' else [path_type]

    pair_dirs = sorted(glob.glob(os.path.join(evaluations_dir, 'seed*-seed*')))
    tasks = []
    for pair_dir in pair_dirs:
        pair_name = os.path.basename(pair_dir)
        for path_name in selected_paths:
            npz_path = os.path.join(pair_dir, path_name, 'curve.npz')
            if os.path.exists(npz_path):
                tasks.append((pair_name, path_name, npz_path))
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Plot XOR curve boundary sliders')
    parser.add_argument('--curve-npz', type=str, default=None,
                        help='Path to a single curve.npz file')
    parser.add_argument('--evaluations-dir', type=str, default=None,
                        help='Path to XOR evaluations dir (e.g., results/xor/evaluations)')
    parser.add_argument('--path-type', type=str, default='bezier',
                        choices=['linear', 'linear_before', 'linear_after', 'bezier', 'polychain', 'all'],
                        help='Path type in batch mode (default: bezier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path for single-file mode')
    parser.add_argument('--output-dir', type=str, default='results/xor/animations/curve_boundaries',
                        help='Output directory for batch mode')
    parser.add_argument('--num-snapshots', type=int, default=21,
                        help='Number of snapshots/frames in slider (default: 21)')
    parser.add_argument('--grid-resolution', type=int, default=140,
                        help='Decision boundary grid resolution (default: 140)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title (single-file mode)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail on first invalid/missing NPZ instead of skipping')
    args = parser.parse_args()

    if args.curve_npz is None and args.evaluations_dir is None:
        parser.error('Provide --curve-npz or --evaluations-dir')
    if args.curve_npz is not None and args.evaluations_dir is not None:
        parser.error('Use either --curve-npz or --evaluations-dir, not both')

    if args.curve_npz is not None:
        npz_path = args.curve_npz
        if not os.path.isabs(npz_path):
            npz_path = os.path.join(project_root, npz_path)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        if args.output is None:
            output_path = os.path.join(os.path.dirname(npz_path), 'boundary_slider.html')
        else:
            output_path = args.output
            if not os.path.isabs(output_path):
                output_path = os.path.join(project_root, output_path)

        title = args.title if args.title else f"XOR Boundary Slider: {os.path.basename(os.path.dirname(npz_path))}"
        process_single(npz_path, output_path, args.num_snapshots, args.grid_resolution, title)
        return

    evaluations_dir = args.evaluations_dir
    if not os.path.isabs(evaluations_dir):
        evaluations_dir = os.path.join(project_root, evaluations_dir)
    if not os.path.isdir(evaluations_dir):
        raise NotADirectoryError(f"Evaluations dir not found: {evaluations_dir}")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = collect_batch_tasks(evaluations_dir, args.path_type)
    if not tasks:
        raise FileNotFoundError(
            f"No curve.npz files found in {evaluations_dir} for path type '{args.path_type}'."
        )

    print(f"Found {len(tasks)} curve files. Generating sliders...")
    num_saved = 0
    num_skipped = 0
    for pair_name, path_name, npz_path in tasks:
        output_path = os.path.join(output_dir, f"{pair_name}_{path_name}_boundary_slider.html")
        title = f"XOR Boundary Slider: {pair_name} ({path_name})"
        try:
            process_single(npz_path, output_path, args.num_snapshots, args.grid_resolution, title)
            num_saved += 1
        except Exception as exc:
            if args.strict:
                raise
            num_skipped += 1
            print(f"Skipped {npz_path}: {exc}")

    print(f"Done. HTML files saved to: {output_dir}")
    print(f"Saved: {num_saved}, Skipped: {num_skipped}")


if __name__ == '__main__':
    main()
