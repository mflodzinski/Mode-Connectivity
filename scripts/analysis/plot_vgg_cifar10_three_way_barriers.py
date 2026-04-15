"""Plot grouped test-loss barriers for VGG CIFAR10 three-way interpolation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.alignment.permutation_pipeline import compute_paper_loss_barrier


METHOD_SPECS = [
    ("test_naive", "No Alignment", "tab:gray"),
    ("test_perm", "Sinkhorn Permutation Only (From Scratch)", "tab:orange"),
    ("test_scale", "Sinkhorn Permutation + Scale (From Scratch)", "tab:purple"),
]

ARCHITECTURES = ["vgg11", "vgg13", "vgg16", "vgg19"]


def load_curves(architecture: str) -> dict:
    path = PROJECT_ROOT / "results" / architecture / "cifar10" / "interpolation_comparison_three_way" / "curves.json"
    with open(path, "r") as handle:
        return json.load(handle)


def compute_barrier(losses: list[float], ts: list[float]) -> float:
    return float(compute_paper_loss_barrier(np.asarray(losses, dtype=np.float64), np.asarray(ts, dtype=np.float64)))


def main() -> None:
    plot_data: dict[str, list[float]] = {label: [] for _, label, _ in METHOD_SPECS}
    architecture_labels: list[str] = []

    for architecture in ARCHITECTURES:
        payload = load_curves(architecture)
        architecture_labels.append(str(payload["vgg_name"]))
        curves = payload["curves"]
        for curve_key, label, _ in METHOD_SPECS:
            curve = curves[curve_key]
            plot_data[label].append(compute_barrier(curve["losses"], curve["lambdas"]))

    output_root = PROJECT_ROOT / "results" / "vgg_cifar10_three_way_barriers"
    output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(architecture_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (_, label, color) in enumerate(METHOD_SPECS):
        offset = (index - 1) * width
        ax.bar(x + offset, plot_data[label], width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(architecture_labels)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Test Loss Barrier")
    ax.set_title("CIFAR10 Test Loss Barriers Across VGG Architectures")
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_root / "vgg_cifar10_three_way_test_loss_barriers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(output_root / "vgg_cifar10_three_way_test_loss_barriers.json", "w") as handle:
        json.dump(
            {
                "architectures": architecture_labels,
                "barriers": plot_data,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
