#!/usr/bin/env python3
"""Plot a run's weight-matched barrier over its complete training schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "GitRebasinCifarMLP": "three-hidden-layer width-512 MLP",
    "VGG11": "VGG11",
}


def published_barrier(losses: list[float]) -> float:
    values = np.asarray(losses, dtype=float)
    return float(values.max() - 0.5 * (values[0] + values[-1]))


def load_protocol(result_root: Path) -> dict:
    path = result_root / "protocol.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("config", {})


def load_onset_profiles(result_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    path = result_root / "onset_report" / "profiles.json"
    records = sorted(
        json.loads(path.read_text()), key=lambda record: record["completed_epoch"]
    )
    epochs = np.asarray([record["completed_epoch"] for record in records], dtype=int)
    curves = {
        "train": np.asarray(
            [
                record["methods"]["wm"]["train_full"]["loss"]["git_rebasin"]
                for record in records
            ],
            dtype=float,
        ),
        "test": np.asarray(
            [
                record["methods"]["wm"]["test_full"]["loss"]["git_rebasin"]
                for record in records
            ],
            dtype=float,
        ),
    }
    return epochs, curves, "full train and test sets"


def load_diagonal_controls(
    result_root: Path, replicate: int
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    controls = result_root / "controls" / str(replicate)
    paths = sorted(controls.glob("diagonal_*.json"))
    if not paths:
        raise FileNotFoundError(f"No diagonal controls found in {controls}")

    epochs = []
    values = {"train": [], "test": []}
    for path in paths:
        epoch = int(path.stem.rsplit("_", 1)[1])
        records = json.loads(path.read_text())
        epochs.append(epoch)
        for split, subset in (("train", "train_eval"), ("test", "test_eval")):
            values[split].append(
                published_barrier(records[f"wm/{subset}"]["losses"])
            )
    order = np.argsort(epochs)
    curves = {
        split: np.asarray(split_values, dtype=float)[order]
        for split, split_values in values.items()
    }
    return np.asarray(epochs, dtype=int)[order], curves, "fixed evaluation subsets"


def load_curves(
    result_root: Path, replicate: int
) -> tuple[np.ndarray, dict[str, np.ndarray], str, str]:
    protocol = load_protocol(result_root)
    model = protocol.get("model", "model")
    onset_path = result_root / "onset_report" / "profiles.json"
    if onset_path.exists():
        epochs, curves, evaluation = load_onset_profiles(result_root)
    else:
        epochs, curves, evaluation = load_diagonal_controls(result_root, replicate)
    return epochs, curves, evaluation, model


def plot(
    result_root: Path,
    output: Path,
    replicate: int,
    include_initialization: bool,
) -> None:
    epochs, curves, evaluation, model = load_curves(result_root, replicate)
    keep = np.ones(len(epochs), dtype=bool)
    if not include_initialization:
        keep &= epochs > 0
    epochs = epochs[keep]
    curves = {split: values[keep] for split, values in curves.items()}
    if not len(epochs):
        raise ValueError("No epochs remain after filtering")

    dense = len(epochs) > 30
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    styles = {
        "train": ("Train", "#2878b5"),
        "test": ("Test", "#e87500"),
    }
    for split, (label, color) in styles.items():
        ax.plot(
            epochs,
            curves[split],
            color=color,
            linewidth=1.8,
            marker=None if dense else "o",
            markersize=4.2,
            label=label,
        )

    model_label = MODEL_LABELS.get(model, model)
    ax.axhline(0.0, color="black", linewidth=0.75)
    ax.set(
        title=f"CIFAR-10 {model_label}: complete training run",
        xlabel="Completed epoch",
        ylabel="Weight-matched loss barrier",
        xlim=(float(epochs.min()), float(epochs.max())),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    payload = {
        "result_root": str(result_root),
        "model": model,
        "replicate": replicate,
        "include_initialization": include_initialization,
        "metric": "max(loss)-0.5*(endpoint_a_loss+endpoint_b_loss)",
        "alignment": "fresh weight matching at each same-stage checkpoint",
        "evaluation": evaluation,
        "completed_epochs": epochs.tolist(),
        "train": curves["train"].tolist(),
        "test": curves["test"].tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--include-initialization", action="store_true")
    args = parser.parse_args()
    output = args.output or (
        args.result_root / "report" / "matched_barrier_complete_training.png"
    )
    plot(
        args.result_root,
        output,
        args.replicate,
        args.include_initialization,
    )
    print(output)


if __name__ == "__main__":
    main()
