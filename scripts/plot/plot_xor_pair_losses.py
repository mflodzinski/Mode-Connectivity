"""Plot XOR loss along linear interpolation before and after alignment."""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from itertools import permutations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
X_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]


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


def compute_barrier(model_a: SimpleMLP, model_b: SimpleMLP, num_points: int = 21) -> dict[str, list[float] | float]:
    """Compute linear interpolation losses and accuracies."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_a["fc1.weight"].shape[0]
    interp_model = SimpleMLP(hidden_size=hidden_size)

    ts = np.linspace(0.0, 1.0, num_points)
    results: dict[str, list[float] | float] = {
        "t": ts.tolist(),
        "loss": [],
        "accuracy": [],
    }

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
            loss = F.binary_cross_entropy_with_logits(outputs, XOR_LABELS).item()
        results["loss"].append(loss)
        results["accuracy"].append(accuracy)

    return results


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
        barrier_result = compute_barrier(model_a, temp_model)
        barrier = 100.0 - min(barrier_result["accuracy"])  # type: ignore[arg-type]
        losses = barrier_result["loss"]  # type: ignore[assignment]
        loss_barrier = max(losses) - (losses[0] + losses[-1]) / 2.0

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


def plot_pair_loss_curves(
    checkpoint_dir: str,
    seed_a: int,
    seed_b: int,
    output_path: str,
    num_points: int = 21,
    show_legend: bool = True,
) -> None:
    """Render one loss-vs-t plot for a pair before and after alignment."""
    model_a, _ = load_model(resolve_checkpoint_path(checkpoint_dir, seed_a))
    model_b, _ = load_model(resolve_checkpoint_path(checkpoint_dir, seed_b))
    model_b_aligned, best_perm = align_models_exhaustive(model_a, model_b)

    raw_results = compute_barrier(model_a, model_b, num_points=num_points)
    aligned_results = compute_barrier(model_a, model_b_aligned, num_points=num_points)

    ts = raw_results["t"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        ts,
        raw_results["loss"],
        label="Raw Interpolation",
        color="#7A7A7A",
        linewidth=2.8,
        linestyle=(0, (6.0, 2.5)),
    )
    ax.plot(
        ts,
        aligned_results["loss"],
        label="Aligned Interpolation",
        color="#C53030",
        linewidth=2.8,
        linestyle=(0, (1.2, 2.0)),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=14)
    ax.set_title(f"Seeds {seed_a}-{seed_b}", fontsize=14)
    fig.tight_layout()
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
    parser = argparse.ArgumentParser(description="Plot XOR loss curves before and after alignment.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory with XOR checkpoints saved as seed{N}.pt or seed_{N}.pt",
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
        "--num-points",
        type=int,
        default=21,
        help="Number of interpolation points for the plotted curve",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide the legend in the exported plot",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for seed_a, seed_b in parse_pairs(args.pairs):
        output_path = os.path.join(
            args.output_dir,
            f"loss_{seed_a}_{seed_b}_before_after.png",
        )
        plot_pair_loss_curves(
            checkpoint_dir=args.checkpoints_dir,
            seed_a=seed_a,
            seed_b=seed_b,
            output_path=output_path,
            num_points=args.num_points,
            show_legend=not args.no_legend,
        )


if __name__ == "__main__":
    main()
