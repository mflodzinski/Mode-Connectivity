"""Plot the two terms hidden by a diagonal loss-barrier subtraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "raw": ("Raw: maximum interpolation loss", "tab:orange"),
    "wm": ("WM: maximum interpolation loss", "tab:red"),
    "transfer_wm": ("Final WM transferred: maximum interpolation loss", "tab:brown"),
}


def load_curves(result_root: Path, replicate: int, subset: str):
    controls = result_root / "controls" / str(replicate)
    paths = sorted(controls.glob("diagonal_*.json"))
    if not paths:
        raise FileNotFoundError(f"No diagonal results found in {controls}")
    epochs, peaks, endpoint_means = [], {m: [] for m in METHODS}, []
    for path in paths:
        epoch = int(path.stem.rsplit("_", 1)[1])
        records = json.loads(path.read_text())
        raw = records[f"raw/{subset}"]
        endpoints = 0.5 * (raw["losses"][0] + raw["losses"][-1])
        epochs.append(epoch)
        endpoint_means.append(endpoints)
        for method in METHODS:
            row = records[f"{method}/{subset}"]
            method_endpoints = 0.5 * (row["losses"][0] + row["losses"][-1])
            if not np.isclose(method_endpoints, endpoints, atol=1e-5, rtol=1e-4):
                raise ValueError(
                    f"Function-preserving method {method} changed endpoints at epoch {epoch}."
                )
            peaks[method].append(max(row["losses"]))
    return np.asarray(epochs), peaks, np.asarray(endpoint_means)


def plot(result_root: Path, replicate: int, subset: str, output: Path):
    epochs, peaks, endpoint_means = load_curves(result_root, replicate, subset)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, (label, color) in METHODS.items():
        ax.plot(epochs, peaks[method], color=color, linewidth=2, label=label)
    ax.plot(
        epochs,
        endpoint_means,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Mean endpoint loss (common to all methods)",
    )
    ax.set(
        title="Same-stage interpolation: peak loss and mean endpoint loss",
        xlabel="Completed epoch",
        ylabel="Cross-entropy loss",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--subset", default="test_eval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.result_root
        / "report"
        / "loss_components"
        / f"diagonal_{args.subset}_peak_vs_endpoint_mean.png"
    )
    plot(args.result_root, args.replicate, args.subset, output)
    print(output)


if __name__ == "__main__":
    main()
