"""Plot XOR decision boundaries for every checkpointed seed."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class SimpleMLP(torch.nn.Module):
    """Simple MLP for XOR: 2 inputs -> H hidden -> 1 output."""

    def __init__(self, hidden_size=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def plot_confidence_boundary(ax, xx, yy, probs):
    """Plot confidence heatmap + 0.5 decision contour."""
    ax.contourf(
        xx,
        yy,
        probs,
        levels=np.linspace(0.0, 1.0, 41),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.85,
    )
    ax.contour(xx, yy, probs, levels=[0.5], colors=["black"], linewidths=2)


def evaluate_xor(model):
    """Return XOR loss and accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(XOR_DATA)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, XOR_LABELS).item()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == XOR_LABELS.long()).float().mean().item() * 100.0
    return loss, acc


def compute_grid_probs(model, grid_resolution):
    """Compute decision grid probabilities."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(grid_points)).squeeze(1).numpy()
    return xx, yy, probs.reshape(xx.shape)


def draw_seed_boundary(model, seed, out_path, grid_resolution):
    """Draw and save one seed boundary figure."""
    xx, yy, z = compute_grid_probs(model, grid_resolution=grid_resolution)
    loss, acc = evaluate_xor(model)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    plot_confidence_boundary(ax, xx, yy, z)

    labels = XOR_LABELS.long().squeeze(1).numpy()
    colors = ["#d62728", "#2ca02c"]
    for point, label in zip(XOR_DATA.numpy(), labels):
        ax.scatter(point[0], point[1], c=colors[label], s=170, edgecolors="black", linewidths=1.8, zorder=5)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Seed {seed} | acc={acc:.0f}% | loss={loss:.4f}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return loss, acc


def draw_montage(seed_rows, out_path):
    """Draw all seeds on one grid."""
    num_models = len(seed_rows)
    cols = 5
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, row in enumerate(seed_rows):
        ax = axes[idx]
        plot_confidence_boundary(ax, row["xx"], row["yy"], row["z"])
        labels = XOR_LABELS.long().squeeze(1).numpy()
        colors = ["#d62728", "#2ca02c"]
        for point, label in zip(XOR_DATA.numpy(), labels):
            ax.scatter(point[0], point[1], c=colors[label], s=70, edgecolors="black", linewidths=1.0, zorder=5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(f"s{row['seed']} | {row['acc']:.0f}% | {row['loss']:.3f}", fontsize=9)

    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot XOR boundaries for all seed checkpoints")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory with seed*.pt checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for boundary plots",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=220,
        help="Grid resolution for boundaries (default: 220)",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(checkpoints_dir.glob("seed*.pt"), key=lambda p: int(p.stem.replace("seed", "")))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    seed_rows = []
    for ckpt_path in ckpts:
        seed = int(ckpt_path.stem.replace("seed", ""))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"]
        hidden_size = int(state["fc1.weight"].shape[0])

        model = SimpleMLP(hidden_size=hidden_size)
        model.load_state_dict(state)

        individual_path = output_dir / f"boundary_seed{seed}.png"
        loss, acc = draw_seed_boundary(model, seed, individual_path, args.grid_resolution)
        xx, yy, z = compute_grid_probs(model, grid_resolution=140)
        seed_rows.append({"seed": seed, "loss": loss, "acc": acc, "xx": xx, "yy": yy, "z": z})
        print(f"SAVED {individual_path} acc={acc:.1f}% loss={loss:.6f}")

    montage_path = output_dir / "boundaries_all_15_seeds.png"
    draw_montage(seed_rows, montage_path)
    print(f"SAVED {montage_path}")


if __name__ == "__main__":
    main()
