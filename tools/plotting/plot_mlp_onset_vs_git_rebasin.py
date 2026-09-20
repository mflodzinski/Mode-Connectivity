#!/usr/bin/env python3
"""Compare this project's MLP onset run with the official Git Re-Basin artifact."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def published_barrier(losses: list[float]) -> float:
    values = np.asarray(losses, dtype=float)
    return float(values.max() - 0.5 * (values[0] + values[-1]))


def load_official(path: Path, split: str) -> np.ndarray:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    return np.asarray(
        [published_barrier(record[f"{split}_loss_interp"]) for record in records],
        dtype=float,
    )


def load_replication(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(path.read_text())
    records = sorted(records, key=lambda record: record["completed_epoch"])
    epochs = np.asarray([record["completed_epoch"] for record in records], dtype=int)
    barriers = np.asarray(
        [record["methods"]["wm"][f"{split}_full"]["loss"]["git_rebasin"] for record in records],
        dtype=float,
    )
    return epochs, barriers


def comparison_stats(official: np.ndarray, ours: np.ndarray) -> dict[str, object]:
    difference = ours - official

    def window(stop: int) -> dict[str, float]:
        delta = difference[:stop]
        return {
            "official_mean": float(official[:stop].mean()),
            "replication_mean": float(ours[:stop].mean()),
            "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean())),
            "correlation": float(np.corrcoef(official[:stop], ours[:stop])[0, 1]),
        }

    largest = int(np.abs(difference).argmax())
    return {
        "epochs_1_10": window(10),
        "epochs_1_25": window(25),
        "epochs_1_100": window(100),
        "fraction_within_0.025": float((np.abs(difference) <= 0.025).mean()),
        "fraction_within_0.05": float((np.abs(difference) <= 0.05).mean()),
        "largest_absolute_difference": float(abs(difference[largest])),
        "largest_difference_epoch": largest + 1,
        "selected_epochs": {
            str(epoch): {
                "official": float(official[epoch - 1]),
                "replication": float(ours[epoch - 1]),
            }
            for epoch in (1, 25, 50, 75, 100)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--official-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ours_path = args.result_root / "onset_report" / "profiles.json"
    output = args.output or args.result_root / "report" / "onset_vs_git_rebasin.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    colors = {"train": "#2878b5", "test": "#e87500"}
    summary: dict[str, object] = {
        "metric": "max(loss)-0.5*(endpoint_a_loss+endpoint_b_loss)",
        "epoch_mapping": "official checkpoint i corresponds to completed epoch i+1",
    }

    for split in ("train", "test"):
        official = load_official(args.official_artifact, split)
        epochs, ours = load_replication(ours_path, split)
        # Official checkpoint i is the state after i + 1 completed epochs.
        official_epochs = np.arange(1, len(official) + 1)
        keep = epochs > 0
        summary[split] = comparison_stats(official, ours[keep])

        axes[0].plot(
            official_epochs[:25], official[:25], color=colors[split], linestyle="--",
            linewidth=1.8, label=f"Git Re-Basin {split}",
        )
        axes[0].plot(
            epochs[keep][:25], ours[keep][:25], color=colors[split], marker="o",
            markersize=2.8, linewidth=1.2, label=f"Replication {split}",
        )
        axes[1].plot(
            official_epochs, official, color=colors[split], linestyle="--", linewidth=1.5,
            label=f"Git Re-Basin {split}",
        )
        axes[1].plot(
            epochs[keep], ours[keep], color=colors[split], alpha=0.85, linewidth=1.1,
            label=f"Replication {split}",
        )

    axes[0].set_title("Published window")
    axes[0].set_xlim(1, 25)
    axes[1].set_title("Complete training run")
    axes[1].set_xlim(1, 100)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel("Matched loss barrier")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("CIFAR-10 MLP onset: official artifact vs PyTorch replication")
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
