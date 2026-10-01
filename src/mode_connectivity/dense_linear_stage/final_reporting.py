"""Final-endpoint, all-method report across independent endpoint pairs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from .alignment import METHODS
from .protocol import pair_dir, root, write_json
from .reporting import METHOD_LABELS, METRICS, SUBSETS


def _rows(cfg):
    epoch = int(cfg["stages"][0])
    rows, profiles = [], {}
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        path = pair_dir(cfg, replicate, epoch, epoch) / "full_profiles.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing final profile: {path}")
        payload = json.loads(path.read_text())
        for method in METHODS:
            for subset in SUBSETS:
                key = f"{method}/{subset}/selected_dense"
                profile = payload[key]
                profiles[(replicate, method, subset)] = profile
                for metric in METRICS:
                    # The paper reports a single loss/error barrier: excess over
                    # the worse endpoint, stored as ``worse`` in the artifacts.
                    rows.append(dict(
                        replicate=replicate,
                        left_seed=int(seeds[0]),
                        right_seed=int(seeds[1]),
                        epoch=epoch,
                        method=method,
                        subset=subset,
                        metric=metric,
                        barrier="worse",
                        value=float(profile[metric]["worse"]),
                        points=len(profile["alphas"]),
                    ))
    return rows, profiles


def _plot_bars(cfg, rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    positions = np.arange(len(METHODS))
    for axis, (subset, title) in zip(
        axes, (("train_report", "Training loss barrier"), ("test_full", "Test loss barrier"))
    ):
        selected = [
            row for row in rows
            if row["subset"] == subset and row["metric"] == "loss"
        ]
        values = {
            method: [row["value"] for row in selected if row["method"] == method]
            for method in METHODS
        }
        means = [float(np.mean(values[method])) for method in METHODS]
        stds = [float(np.std(values[method], ddof=1)) for method in METHODS]
        axis.bar(positions, means, yerr=stds, capsize=4)
        axis.set(
            ylabel=title,
            xticks=positions,
            xticklabels=[METHOD_LABELS[method] for method in METHODS],
            title=title,
        )
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(
        f"{cfg.get('model', cfg['dataset'])}: mean ± sample SD over "
        f"{len(cfg['seed_pairs'])} endpoint pairs"
    )
    stem = "final_train_test_loss_barriers"
    fig.savefig(destination / f"{stem}.pdf")
    fig.savefig(destination / f"{stem}.png", dpi=180)
    plt.close(fig)
    return stem


def _plot_profiles(cfg, profiles, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    panels = (
        ("train_report", "loss", "Training-subset loss", "losses"),
        ("train_report", "error", "Training-subset error (%)", "errors"),
        ("test_full", "loss", "Test loss", "losses"),
        ("test_full", "error", "Test error (%)", "errors"),
    )
    for axis, (subset, metric, title, values_key) in zip(axes.flat, panels):
        for method in METHODS:
            curves = [
                profiles[(replicate, method, subset)]
                for replicate in range(len(cfg["seed_pairs"]))
            ]
            grid = np.asarray(curves[0]["alphas"], dtype=float)
            matrix = np.asarray([curve[values_key] for curve in curves], dtype=float)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1)
            axis.plot(grid, mean, label=METHOD_LABELS[method])
            axis.fill_between(grid, mean - std, mean + std, alpha=.13)
        axis.set(title=title, xlabel="Interpolation coefficient α")
        axis.grid(alpha=.2)
    axes[0, 0].legend(ncol=2, fontsize=7)
    stem = "final_absolute_profiles"
    fig.savefig(destination / f"{stem}.pdf")
    fig.savefig(destination / f"{stem}.png", dpi=180)
    plt.close(fig)
    return stem


def report(cfg):
    destination = root(cfg) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    rows, profiles = _rows(cfg)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "barriers.json", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["subset"], row["metric"])].append(row["value"])
    aggregates = []
    for (method, subset, metric), values in sorted(grouped.items()):
        aggregates.append(dict(
            method=method,
            subset=subset,
            metric=metric,
            barrier="worse",
            count=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)),
            values=[float(value) for value in values],
        ))
    write_json(destination / "aggregates.json", aggregates)
    summary = dict(
        benchmark_mode="final_alignment",
        dataset=cfg["dataset"],
        model=cfg.get("model"),
        epoch=int(cfg["stages"][0]),
        seed_pairs=cfg["seed_pairs"],
        methods=list(METHODS),
        replicates=len(cfg["seed_pairs"]),
        statistic="mean and sample standard deviation across independent endpoint pairs",
        hyperparameter_selection=(
            "Each optimized method is selected separately for each endpoint pair on "
            "the frozen validation split; test data are loaded only after choices are frozen."
        ),
        figures=[
            _plot_bars(cfg, rows, destination),
            _plot_profiles(cfg, profiles, destination),
        ],
    )
    write_json(destination / "summary.json", summary)
    return summary
