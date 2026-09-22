"""Aggregate tables and figures for Experiments 4 and 5."""

from __future__ import annotations

import csv
import json

import numpy as np

from .evaluation import METHODS
from .protocol import pair_dir, root, write_json


def report(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root(cfg) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        for epoch in cfg["stages"]:
            path = pair_dir(cfg, replicate, epoch) / "profiles.json"
            if not path.exists():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text())
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
    write_json(destination / "barriers.json", rows)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for subset in ["train_eval", "validation_audit", "test_full"]:
        for metric in ["loss", "error"]:
            for method in METHODS:
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
    write_json(destination / "aggregates.json", aggregates)

    for metric, ylabel in [
        ("loss", "Loss chord barrier"),
        ("error", "Error chord barrier (pp)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in METHODS:
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
        fig.savefig(destination / f"training_stage_{metric}.pdf")
        fig.savefig(destination / f"training_stage_{metric}.png", dpi=180)
        plt.close(fig)

    final = int(cfg["epochs"])
    final_rows = [
        row
        for row in aggregates
        if row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["epoch"] == final
        and row["method"] in METHODS[:3]
    ]
    final_rows.sort(key=lambda row: METHODS.index(row["method"]))
    hierarchy_by_pair = []
    raw_rows = [
        row
        for row in rows
        if row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["epoch"] == final
    ]
    for replicate in range(len(cfg["seed_pairs"])):
        values = {
            row["method"]: row["chord"]
            for row in raw_rows
            if row["replicate"] == replicate
        }
        hierarchy_by_pair.append(
            dict(
                replicate=replicate,
                seeds=cfg["seed_pairs"][replicate],
                barriers={key: values[key] for key in METHODS[:3]},
                strict_hierarchy=(
                    values["raw"] > values["permutation"] > values["permutation_scale"]
                ),
            )
        )
    summary = dict(
        experiment_4=dict(
            stages=cfg["stages"],
            methods=METHODS,
            figure_loss=str(destination / "training_stage_loss.pdf"),
            figure_error=str(destination / "training_stage_error.pdf"),
        ),
        experiment_5=dict(
            epoch=final,
            methods=METHODS[:3],
            aggregate_loss_barriers=final_rows,
            hierarchy_by_pair=hierarchy_by_pair,
            strict_hierarchy_count=sum(
                row["strict_hierarchy"] for row in hierarchy_by_pair
            ),
            pair_count=len(hierarchy_by_pair),
        ),
    )
    write_json(destination / "summary.json", summary)
    return summary
