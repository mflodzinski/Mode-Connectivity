"""Plot XOR decision-boundary interpolation grids before and after alignment.

This script loads saved XOR checkpoints, computes the optimal hidden-unit
permutation by exhaustive search, and renders a 2x5 grid of decision
boundaries for fixed interpolation snapshots t in {0, 0.25, 0.5, 0.75, 1}.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from itertools import permutations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
DEFAULT_SNAPSHOTS = [0.0, 0.25, 0.5, 0.75, 1.0]
CLASS_COLORS = ["#2B6CB0", "#ECC94B"]
BOUNDARY_CMAP = LinearSegmentedColormap.from_list(
    "xor_blue_yellow",
    CLASS_COLORS,
)


class SimpleMLP(nn.Module):
    """Simple XOR MLP: 2 inputs -> H hidden -> 1 output."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model(checkpoint_path: str) -> tuple[SimpleMLP, int]:
    """Load one XOR checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hidden_size = int(checkpoint["hidden_size"])
    model = SimpleMLP(hidden_size=hidden_size)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has neither 'model_state' nor 'state_dict'.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, hidden_size


def resolve_checkpoint_path(checkpoint_dir: str, seed: int) -> str:
    """Resolve either seed{N}.pt or seed_{N}.pt."""
    candidates = [
        os.path.join(checkpoint_dir, f"seed{seed}.pt"),
        os.path.join(checkpoint_dir, f"seed_{seed}.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find a checkpoint for seed {seed} in {checkpoint_dir}. "
        f"Tried: {candidates}"
    )


def compute_l2_distance(state_a: OrderedDict[str, torch.Tensor], state_b: OrderedDict[str, torch.Tensor]) -> float:
    """Compute L2 distance between two model states."""
    total_diff_sq = 0.0
    for key in state_a:
        diff = state_a[key].float() - state_b[key].float()
        total_diff_sq += (diff ** 2).sum().item()
    return total_diff_sq ** 0.5


def apply_permutation_to_state(
    state: OrderedDict[str, torch.Tensor],
    perm: tuple[int, ...] | list[int],
) -> OrderedDict[str, torch.Tensor]:
    """Apply a hidden-unit permutation to one XOR checkpoint state."""
    perm = list(perm)
    new_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    new_state["fc1.weight"] = state["fc1.weight"][perm, :]
    new_state["fc1.bias"] = state["fc1.bias"][perm]
    new_state["fc2.weight"] = state["fc2.weight"][:, perm]
    new_state["fc2.bias"] = state["fc2.bias"].clone()
    return new_state


def compute_barrier(model_a: SimpleMLP, model_b: SimpleMLP, num_points: int = 21) -> tuple[float, float]:
    """Return error barrier and loss barrier for linear interpolation."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_a["fc1.weight"].shape[0]
    interp_model = SimpleMLP(hidden_size=hidden_size)

    ts = np.linspace(0.0, 1.0, num_points)
    losses: list[float] = []
    accuracies: list[float] = []

    for t in ts:
        interp_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key in state_a:
            interp_state[key] = (1.0 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)
        interp_model.eval()
        with torch.no_grad():
            outputs = interp_model(XOR_DATA)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            accuracy = (preds == XOR_LABELS.long()).float().mean().item() * 100.0
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, XOR_LABELS).item()
        losses.append(loss)
        accuracies.append(accuracy)

    error_barrier = 100.0 - min(accuracies)
    endpoint_avg_loss = (losses[0] + losses[-1]) / 2.0
    loss_barrier = max(losses) - endpoint_avg_loss
    return error_barrier, loss_barrier


def align_models_exhaustive(model_a: SimpleMLP, model_b: SimpleMLP) -> tuple[SimpleMLP, list[int]]:
    """Find the optimal hidden permutation for model_b relative to model_a."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_b["fc1.weight"].shape[0]

    best_perm: tuple[int, ...] | None = None
    best_barrier = float("inf")
    best_loss_barrier = float("inf")
    best_distance = float("inf")

    for perm in permutations(range(hidden_size)):
        permuted_state = apply_permutation_to_state(state_b, perm)
        distance = compute_l2_distance(state_a, permuted_state)
        temp_model = SimpleMLP(hidden_size=hidden_size)
        temp_model.load_state_dict(permuted_state)
        barrier, loss_barrier = compute_barrier(model_a, temp_model)

        better = False
        if barrier < best_barrier:
            better = True
        elif barrier == best_barrier:
            if loss_barrier < best_loss_barrier:
                better = True
            elif loss_barrier == best_loss_barrier and distance < best_distance:
                better = True

        if better:
            best_perm = perm
            best_barrier = barrier
            best_loss_barrier = loss_barrier
            best_distance = distance

    if best_perm is None:
        raise RuntimeError("Failed to find an optimal permutation.")

    aligned_state = apply_permutation_to_state(state_b, best_perm)
    aligned_model = SimpleMLP(hidden_size=hidden_size)
    aligned_model.load_state_dict(aligned_state)
    aligned_model.eval()
    return aligned_model, list(best_perm)


def make_interpolated_model(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    t: float,
) -> SimpleMLP:
    """Create a model at interpolation parameter t."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_a["fc1.weight"].shape[0]
    interp_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in state_a:
        interp_state[key] = (1.0 - t) * state_a[key] + t * state_b[key]
    model = SimpleMLP(hidden_size=hidden_size)
    model.load_state_dict(interp_state)
    model.eval()
    return model


def compute_boundary_grid(model: SimpleMLP, grid_resolution: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute decision-boundary probabilities on a fixed grid."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(grid_points)
        probs = torch.sigmoid(outputs).squeeze(1).numpy()
    z = probs.reshape(xx.shape)
    return xx, yy, z


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


def style_axis(
    ax: plt.Axes,
    t: float,
    show_ylabel: bool,
    show_xlabel: bool,
    show_title: bool,
) -> None:
    """Apply the requested axis formatting."""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    if show_title:
        ax.set_title(f"t = {t:g}", fontsize=13, fontweight="bold")
    else:
        ax.set_title("")
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


def add_xor_points(ax: plt.Axes) -> None:
    """Plot the four XOR examples."""
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


def plot_pair(
    checkpoint_dir: str,
    seed_a: int,
    seed_b: int,
    output_path: str,
    grid_resolution: int = 200,
    snapshots: list[float] | None = None,
) -> None:
    """Render one 2x5 before/after interpolation figure."""
    model_a, _ = load_model(resolve_checkpoint_path(checkpoint_dir, seed_a))
    model_b, _ = load_model(resolve_checkpoint_path(checkpoint_dir, seed_b))
    aligned_model_b, best_perm = align_models_exhaustive(model_a, model_b)

    if snapshots is None:
        snapshots = DEFAULT_SNAPSHOTS

    fig, axes = plt.subplots(2, len(snapshots), figsize=(16, 7))

    for col, t in enumerate(snapshots):
        raw_model = make_interpolated_model(model_a, model_b, t)
        raw_xx, raw_yy, raw_z = compute_boundary_grid(raw_model, grid_resolution=grid_resolution)
        plot_confidence_boundary(axes[0, col], raw_xx, raw_yy, raw_z)
        add_xor_points(axes[0, col])
        style_axis(
            axes[0, col],
            t=t,
            show_ylabel=(col == 0),
            show_xlabel=False,
            show_title=True,
        )

        aligned_interp_model = make_interpolated_model(model_a, aligned_model_b, t)
        aligned_xx, aligned_yy, aligned_z = compute_boundary_grid(
            aligned_interp_model,
            grid_resolution=grid_resolution,
        )
        plot_confidence_boundary(axes[1, col], aligned_xx, aligned_yy, aligned_z)
        add_xor_points(axes[1, col])
        style_axis(
            axes[1, col],
            t=t,
            show_ylabel=(col == 0),
            show_xlabel=True,
            show_title=False,
        )

    fig.text(
        0.055,
        0.73,
        "Raw Interpolation",
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.285,
        "Aligned Interpolation",
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.09, wspace=0.08, hspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} (best permutation: {best_perm})")


def parse_pairs(pair_specs: list[str]) -> list[tuple[int, int]]:
    """Parse CLI pair strings of the form '2-4'."""
    pairs: list[tuple[int, int]] = []
    for spec in pair_specs:
        left, right = spec.split("-", 1)
        pairs.append((int(left), int(right)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XOR interpolation grids before and after optimal permutation.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory with XOR checkpoints saved as seed{N}.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the plots will be written",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        required=True,
        help="Seed pairs to plot, e.g. 2-4 9-12 11-14",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=200,
        help="Decision-boundary grid resolution",
    )
    parser.add_argument(
        "--snapshots",
        type=float,
        nargs="+",
        default=None,
        help="Interpolation t values to visualize, e.g. 0 0.25 0.5 0.75 1",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for seed_a, seed_b in parse_pairs(args.pairs):
        output_path = os.path.join(
            args.output_dir,
            f"interpolation_{seed_a}_{seed_b}_before_after.png",
        )
        plot_pair(
            checkpoint_dir=args.checkpoints_dir,
            seed_a=seed_a,
            seed_b=seed_b,
            output_path=output_path,
            grid_resolution=args.grid_resolution,
            snapshots=args.snapshots,
        )


if __name__ == "__main__":
    main()
