"""Plot static XOR decision-boundary snapshots from a saved curve.npz path."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
DEFAULT_SNAPSHOTS = [0.0, 0.25, 0.5, 0.75, 1.0]
CLASS_COLORS = ["#2B6CB0", "#ECC94B"]
BOUNDARY_CMAP = LinearSegmentedColormap.from_list("xor_blue_yellow", CLASS_COLORS)


def infer_hidden_size(num_params: int) -> int:
    """Infer hidden size H from parameter count of a 2-H-1 XOR MLP."""
    if (num_params - 1) % 4 != 0:
        raise ValueError(f"Cannot infer hidden size from parameter count={num_params}")
    return (num_params - 1) // 4


def logits_from_param_vector(x: torch.Tensor, param_vector: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Evaluate a 2-H-1 XOR network encoded as one flat parameter vector."""
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


def compute_boundary(param_vector: np.ndarray, hidden_size: int, grid_resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute probability grid for one curve snapshot."""
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
        probs = torch.sigmoid(grid_logits).squeeze(1).numpy()
    return xx, yy, probs.reshape(xx.shape)


def plot_confidence_boundary(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, probs: np.ndarray) -> None:
    """Render one decision-boundary panel."""
    ax.contourf(
        xx,
        yy,
        probs,
        levels=np.linspace(0.0, 1.0, 41),
        cmap=BOUNDARY_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
    )
    ax.contour(xx, yy, probs, levels=[0.5], colors=["black"], linewidths=2)


def add_xor_points(ax: plt.Axes) -> None:
    """Overlay the four XOR examples."""
    labels = XOR_LABELS.long().squeeze(1).numpy()
    for point, label in zip(XOR_DATA.numpy(), labels):
        ax.scatter(
            point[0],
            point[1],
            c=CLASS_COLORS[label],
            s=160,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )


def style_axis(ax: plt.Axes, t: float, show_ylabel: bool, show_xlabel: bool) -> None:
    """Apply the same styling as the interpolation figures."""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"t = {t:g}", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    if show_xlabel:
        ax.set_xlabel(r"$x_1$", fontsize=12)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(r"$x_2$", fontsize=12)
    else:
        ax.set_ylabel("")


def plot_curve_snapshots(
    curve_npz: str,
    output_path: str,
    snapshots: list[float] | None = None,
    grid_resolution: int = 200,
    row_label: str = "Bezier Curve",
) -> None:
    """Render a one-row multi-snapshot figure from curve.npz."""
    if snapshots is None:
        snapshots = DEFAULT_SNAPSHOTS

    data = np.load(curve_npz)
    ts = np.asarray(data["ts"], dtype=np.float64)
    param_vectors = np.asarray(data["param_vectors"], dtype=np.float64)
    hidden_size = infer_hidden_size(param_vectors.shape[1])
    idxs = [int(np.argmin(np.abs(ts - t))) for t in snapshots]

    fig, axes = plt.subplots(1, len(idxs), figsize=(16, 3.8))
    if len(idxs) == 1:
        axes = [axes]

    for col, idx in enumerate(idxs):
        xx, yy, z = compute_boundary(param_vectors[idx], hidden_size, grid_resolution)
        plot_confidence_boundary(axes[col], xx, yy, z)
        add_xor_points(axes[col])
        style_axis(
            axes[col],
            t=float(ts[idx]),
            show_ylabel=(col == 0),
            show_xlabel=True,
        )

    fig.text(
        0.055,
        0.52,
        row_label,
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.14, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XOR curve snapshots from curve.npz")
    parser.add_argument("--curve-npz", required=True, help="Path to curve.npz")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--snapshots",
        type=float,
        nargs="+",
        default=None,
        help="Interpolation t values to visualize",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=200,
        help="Decision-boundary grid resolution",
    )
    parser.add_argument(
        "--row-label",
        type=str,
        default="Bezier Curve",
        help="Left-side row label",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_curve_snapshots(
        curve_npz=args.curve_npz,
        output_path=args.output,
        snapshots=args.snapshots,
        grid_resolution=args.grid_resolution,
        row_label=args.row_label,
    )


if __name__ == "__main__":
    main()
