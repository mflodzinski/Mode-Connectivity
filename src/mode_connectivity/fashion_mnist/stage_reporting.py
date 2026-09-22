"""Aggregate separate linear and nonlinear training-stage results."""

from __future__ import annotations

import csv
import json

import numpy as np

from .evaluation import LINEAR_METHODS, NONLINEAR_METHODS
from .protocol import pair_dir, root, write_json


def _collect(cfg, profile_name, methods):
    rows = []
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        for epoch in cfg["stages"]:
            path = pair_dir(cfg, replicate, epoch) / profile_name
            if not path.exists():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text())
            if value["methods"] != methods:
                raise ValueError(f"Unexpected methods in {path}: {value['methods']}")
            for name, profile in value["profiles"].items():
                method, subset = name.split("/")
                for metric in ["loss", "error"]:
                    rows.append(
                        dict(
                            replicate=replicate,
                            left_seed=seeds[0],
                            right_seed=seeds[1],
                            epoch=epoch,
                            method=method,
                            subset=subset,
                            metric=metric,
                            chord=profile[metric]["chord"],
                            worse=profile[metric]["worse"],
                            mean=profile[metric]["mean"],
                            peak_alpha=profile[metric]["peak_alpha"],
                        )
                    )
    return rows


def _aggregate(cfg, rows, methods):
    aggregates = []
    for subset in ["train_eval", "validation_audit", "test_full"]:
        for metric in ["loss", "error"]:
            for method in methods:
                for epoch in cfg["stages"]:
                    values = [
                        row["chord"]
                        for row in rows
                        if row["subset"] == subset
                        and row["metric"] == metric
                        and row["method"] == method
                        and row["epoch"] == epoch
                    ]
                    aggregates.append(
                        dict(
                            subset=subset,
                            metric=metric,
                            method=method,
                            epoch=epoch,
                            n=len(values),
                            mean=float(np.mean(values)),
                            std=float(np.std(values, ddof=1))
                            if len(values) > 1
                            else 0.0,
                            values=values,
                        )
                    )
    return aggregates


def _write_common(destination, rows, aggregates, methods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "barriers.json", rows)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "aggregates.json", aggregates)
    figures = {}
    for metric, ylabel in [
        ("loss", "Loss chord barrier"),
        ("error", "Error chord barrier (pp)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in methods:
            selected = [
                row
                for row in aggregates
                if row["subset"] == "test_full"
                and row["metric"] == metric
                and row["method"] == method
            ]
            selected.sort(key=lambda row: row["epoch"])
            x = [row["epoch"] for row in selected]
            mean = np.asarray([row["mean"] for row in selected])
            std = np.asarray([row["std"] for row in selected])
            ax.plot(x, mean, marker="o", label=method.replace("_", "+"))
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set(xlabel="Completed epoch", ylabel=ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = destination / f"training_stage_{metric}.pdf"
        fig.savefig(path)
        fig.savefig(destination / f"training_stage_{metric}.png", dpi=180)
        plt.close(fig)
        figures[metric] = str(path)
    return figures


def report_linear(cfg):
    destination = root(cfg) / "report" / "linear_stage"
    rows = _collect(cfg, "linear_profiles.json", LINEAR_METHODS)
    aggregates = _aggregate(cfg, rows, LINEAR_METHODS)
    figures = _write_common(destination, rows, aggregates, LINEAR_METHODS)
    final = int(cfg["epochs"])
    final_rows = [
        row
        for row in rows
        if row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["epoch"] == final
    ]
    hierarchy_by_pair = []
    for replicate in range(len(cfg["seed_pairs"])):
        values = {
            row["method"]: row["chord"]
            for row in final_rows
            if row["replicate"] == replicate
        }
        hierarchy_by_pair.append(
            dict(
                replicate=replicate,
                seeds=cfg["seed_pairs"][replicate],
                barriers=values,
                strict_hierarchy=(
                    values["raw"] > values["permutation"] > values["permutation_scale"]
                ),
            )
        )
    summary = dict(
        family="linear",
        stages=cfg["stages"],
        methods=LINEAR_METHODS,
        figures=figures,
        final_epoch=final,
        hierarchy_by_pair=hierarchy_by_pair,
        strict_hierarchy_count=sum(
            row["strict_hierarchy"] for row in hierarchy_by_pair
        ),
        pair_count=len(hierarchy_by_pair),
    )
    write_json(destination / "summary.json", summary)
    return summary


def report_nonlinear(cfg):
    destination = root(cfg) / "report" / "nonlinear_stage"
    rows = _collect(cfg, "nonlinear_profiles.json", NONLINEAR_METHODS)
    aggregates = _aggregate(cfg, rows, NONLINEAR_METHODS)
    figures = _write_common(destination, rows, aggregates, NONLINEAR_METHODS)
    summary = dict(
        family="quadratic_bezier",
        endpoints="raw",
        stages=cfg["stages"],
        methods=NONLINEAR_METHODS,
        figures=figures,
    )
    write_json(destination / "summary.json", summary)
    return summary
