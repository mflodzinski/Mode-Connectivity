"""Reproduce the width-2 XOR examples with explicit identity/swap rows."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mode_connectivity.xor.xor_experiment import (
    SimpleMLP,
    XOR_DATA,
    XOR_LABELS,
    apply_permutation_to_state,
    compute_barrier,
)


PAIRS = {
    "successful_pair_seeds_2_4": (2, 4),
    "failed_pair_seeds_9_12": (9, 12),
}


def load_model(checkpoint_path: Path) -> SimpleMLP:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SimpleMLP(hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def transformed_model(model: SimpleMLP, permutation: tuple[int, ...]) -> SimpleMLP:
    transformed = SimpleMLP(hidden_size=model.hidden_size)
    transformed.load_state_dict(apply_permutation_to_state(model.state_dict(), permutation))
    transformed.eval()
    return transformed


def interpolation_model(
    left: SimpleMLP,
    right: SimpleMLP,
    t: float,
) -> SimpleMLP:
    state = OrderedDict(
        (name, torch.lerp(left.state_dict()[name], value, t))
        for name, value in right.state_dict().items()
    )
    interpolated = SimpleMLP(hidden_size=left.hidden_size)
    interpolated.load_state_dict(state)
    interpolated.eval()
    return interpolated


def plot_pair(
    left: SimpleMLP,
    right: SimpleMLP,
    output_stem: Path,
    *,
    grid_resolution: int = 200,
) -> None:
    identity = transformed_model(right, (0, 1))
    swap = transformed_model(right, (1, 0))
    rows = (("Identity", identity), ("Swap", swap))
    ts = np.linspace(0.0, 1.0, 5)

    axis_min, axis_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(axis_min, axis_max, grid_resolution),
        np.linspace(axis_min, axis_max, grid_resolution),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    fig, axes = plt.subplots(2, len(ts), figsize=(16, 6.7), sharex=True, sharey=True)
    point_colors = ("#2b6cb0", "#ecc94b")
    original_boundary_cmap = LinearSegmentedColormap.from_list(
        "xor_original_blue_yellow",
        point_colors,
    )
    labels = XOR_LABELS.long().squeeze(1).numpy()

    for row_index, (row_label, right_endpoint) in enumerate(rows):
        for column_index, t in enumerate(ts):
            ax = axes[row_index, column_index]
            model = interpolation_model(left, right_endpoint, float(t))
            with torch.no_grad():
                probabilities = torch.sigmoid(model(grid)).squeeze(1).numpy()

            ax.contourf(
                xx,
                yy,
                probabilities.reshape(xx.shape),
                levels=np.linspace(0.0, 1.0, 41),
                cmap=original_boundary_cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
            )
            ax.contour(
                xx,
                yy,
                probabilities.reshape(xx.shape),
                levels=[0.5],
                colors="black",
                linewidths=2,
            )

            for point, label in zip(XOR_DATA.numpy(), labels):
                ax.scatter(
                    point[0],
                    point[1],
                    c=point_colors[label],
                    s=115,
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )

            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(
                    rf"$\lambda = {t:g}$",
                    fontsize=22,
                    fontweight="bold",
                    pad=10,
                )
            if row_index == 1:
                ax.set_xlabel(r"$x_1$", fontsize=22, labelpad=5)
            if column_index == 0:
                ax.set_ylabel(r"$x_2$", fontsize=22, labelpad=5)

        row_center = 0.73 if row_index == 0 else 0.285
        fig.text(
            0.018,
            row_center,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=24,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.08, top=0.93, wspace=0.08, hspace=0.12)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    identity_result = compute_barrier(left, identity, num_points=21)
    swap_result = compute_barrier(left, swap, num_points=21)
    print(
        f"{output_stem.name}: "
        f"identity min accuracy={identity_result['min_accuracy']:.0f}%, "
        f"swap min accuracy={swap_result['min_accuracy']:.0f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=ROOT / "results/xor/xor_2h_15seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/xor/xor_2h_15seeds/identity_swap_figures",
    )
    args = parser.parse_args()

    checkpoint_dir = args.experiment_dir / "checkpoints"
    for output_name, (left_seed, right_seed) in PAIRS.items():
        left = load_model(checkpoint_dir / f"seed{left_seed}.pt")
        right = load_model(checkpoint_dir / f"seed{right_seed}.pt")
        plot_pair(left, right, args.output_dir / output_name)


if __name__ == "__main__":
    main()
