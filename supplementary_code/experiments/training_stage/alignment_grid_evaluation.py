"""Held-out evaluation and reporting for the selected VGG11 alignment grid."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mode_connectivity.training_stage.evaluation import evaluate_pair
from mode_connectivity.training_stage.protocol import StopFlag, verify_protocol, write_json
from mode_connectivity.training_stage.runner import validate_config

from .alignment_grid import configure_torch, pairs


METHODS = ("raw", "base", "scale")
SUBSETS = ("train_eval", "test_eval")


def selected_config(source_root: Path, sweep_root: Path) -> tuple[dict, Path]:
    selected = sweep_root.resolve() / "selected"
    protocol_path = selected / "protocol.json"
    selection_path = selected / "selection.json"
    if not protocol_path.exists() or not selection_path.exists():
        raise FileNotFoundError(
            "Complete the Sinkhorn and scale grid selection before evaluation."
        )
    cfg = json.loads(protocol_path.read_text())["config"]
    cfg["output_root"] = str(selected)
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    validate_config(cfg)
    verify_protocol(cfg)
    selection = json.loads(selection_path.read_text())
    if Path(selection["source_root"]).resolve() != source_root.resolve():
        raise ValueError("Selected grid artifacts belong to a different source root.")
    return cfg, selected


def task_index(value: int | None) -> int:
    if value is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("Set --task-id or run inside a Slurm array task.")
    return int(value)


def evaluate_selected(args) -> None:
    cfg, selected = selected_config(args.source_root, args.sweep_root)
    pair_grid = pairs(cfg)
    index = task_index(args.task_id)
    if index < 0 or index >= len(pair_grid):
        raise IndexError(f"Task {index} is outside the {len(pair_grid)}-pair grid.")
    configure_torch(cfg)
    replicate, left, right = pair_grid[index]
    status_dir = selected / "evaluation_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{replicate}_{left:03d}_{right:03d}.json"
    profile_path = (
        selected / "pairs" / str(replicate) / f"{left:03d}_{right:03d}" / "profiles.json"
    )
    with status_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists() and profile_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") == "complete":
                print(f"Already complete: pair={(replicate, left, right)}")
                return
        write_json(
            status_path,
            dict(status="running", pair=[replicate, left, right]),
        )
        try:
            result = evaluate_pair(
                cfg,
                replicate,
                left,
                right,
                StopFlag(),
                methods=list(METHODS),
                subsets=list(SUBSETS),
                destination=profile_path,
            )
            write_json(
                status_path,
                dict(
                    status="complete",
                    pair=[replicate, left, right],
                    profile=str(profile_path),
                    entries=len(result),
                    test_access="after validation-only hyperparameter selection",
                ),
            )
        except BaseException as exc:
            write_json(
                status_path,
                dict(
                    status="failed",
                    pair=[replicate, left, right],
                    error=repr(exc),
                ),
            )
            raise


def collect_rows(cfg: dict, selected: Path) -> tuple[list[dict], list[dict]]:
    rows, profiles = [], []
    for replicate, left, right in pairs(cfg):
        path = selected / "pairs" / str(replicate) / f"{left:03d}_{right:03d}" / "profiles.json"
        if not path.exists():
            raise FileNotFoundError(f"Selected-grid evaluation is incomplete: {path}")
        value = json.loads(path.read_text())
        for method in METHODS:
            for subset in SUBSETS:
                profile = value[f"{method}/{subset}"]
                profiles.append(
                    {
                        **profile,
                        "replicate": replicate,
                        "epochs": [left, right],
                        "method": method,
                        "subset": subset,
                    }
                )
                for metric in ("loss", "error"):
                    barrier = profile[metric]
                    rows.append(
                        dict(
                            replicate=replicate,
                            left_epoch=left,
                            right_epoch=right,
                            method=method,
                            subset=subset,
                            metric=metric,
                            chord=barrier["chord"],
                            worse=barrier["worse"],
                            git_rebasin=barrier["git_rebasin"],
                            mean=barrier["mean"],
                            peak_alpha=barrier["peak_alpha"],
                        )
                    )
    return rows, profiles


def matrix(cfg: dict, rows: list[dict], method: str, subset: str, metric: str, field: str):
    stages = list(cfg["stages"])
    lookup = {
        (row["left_epoch"], row["right_epoch"]): float(row[field])
        for row in rows
        if row["method"] == method
        and row["subset"] == subset
        and row["metric"] == metric
    }
    return np.asarray([[lookup[(left, right)] for right in stages] for left in stages])


def write_matrix_figures(cfg: dict, rows: list[dict], destination: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = list(cfg["stages"])
    outputs = []
    for metric, metric_label in (("loss", "Loss"), ("error", "Error (pp)")):
        for field, field_label in (("chord", "chord"), ("worse", "worse-endpoint")):
            arrays = [matrix(cfg, rows, method, "test_eval", metric, field) for method in METHODS]
            vmax = max(float(array.max()) for array in arrays)
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
            image = None
            for axis, method, values in zip(axes, METHODS, arrays):
                image = axis.imshow(values, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
                axis.set(
                    title={"raw": "Raw", "base": "Sinkhorn", "scale": "Sinkhorn + scale"}[method],
                    xlabel="Right completed epoch",
                    ylabel="Left completed epoch",
                    xticks=range(len(stages)),
                    yticks=range(len(stages)),
                    xticklabels=stages,
                    yticklabels=stages,
                )
            for axis in axes:
                axis.tick_params(axis="x", rotation=45)
            fig.colorbar(image, ax=axes, label=f"{metric_label} {field_label} barrier")
            stem = f"test_{metric}_{field}_matrices"
            fig.savefig(destination / f"{stem}.pdf")
            fig.savefig(destination / f"{stem}.png", dpi=180)
            plt.close(fig)
            outputs.append(stem)

    raw = matrix(cfg, rows, "raw", "test_eval", "loss", "chord")
    base = matrix(cfg, rows, "base", "test_eval", "loss", "chord")
    scale = matrix(cfg, rows, "scale", "test_eval", "loss", "chord")
    improvements = [raw - base, base - scale, raw - scale]
    titles = ["Raw − Sinkhorn", "Sinkhorn − scale", "Raw − scale"]
    bound = max(abs(float(array.min())) for array in improvements)
    bound = max(bound, max(abs(float(array.max())) for array in improvements), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    image = None
    for axis, title, values in zip(axes, titles, improvements):
        image = axis.imshow(values, origin="lower", vmin=-bound, vmax=bound, cmap="coolwarm")
        axis.set(
            title=title,
            xlabel="Right completed epoch",
            ylabel="Left completed epoch",
            xticks=range(len(stages)),
            yticks=range(len(stages)),
            xticklabels=stages,
            yticklabels=stages,
        )
        axis.tick_params(axis="x", rotation=45)
    fig.colorbar(image, ax=axes, label="Test loss chord-barrier reduction")
    fig.savefig(destination / "test_loss_chord_improvements.pdf")
    fig.savefig(destination / "test_loss_chord_improvements.png", dpi=180)
    plt.close(fig)
    outputs.append("test_loss_chord_improvements")
    return outputs


def write_absolute_profiles(profiles: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    index = {
        (tuple(row["epochs"]), row["method"], row["subset"]): row
        for row in profiles
    }
    pairs_found = sorted({tuple(row["epochs"]) for row in profiles})
    with PdfPages(destination / "absolute_profiles.pdf") as pdf:
        for epochs in pairs_found:
            fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
            for row_index, subset in enumerate(SUBSETS):
                for column_index, (values_key, ylabel) in enumerate(
                    (("losses", "Cross-entropy"), ("errors", "Error (%)"))
                ):
                    axis = axes[row_index, column_index]
                    for method, label in zip(METHODS, ("raw", "sinkhorn", "scale")):
                        row = index[(epochs, method, subset)]
                        axis.plot(row["alphas"], row[values_key], label=label)
                    axis.set(
                        title=f"{subset}: {ylabel}",
                        xlabel="Interpolation coefficient α",
                        ylabel=ylabel,
                    )
                    axis.grid(alpha=0.2)
            axes[0, 0].legend()
            fig.suptitle(f"Completed epochs {epochs[0]} → {epochs[1]}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def report_selected(args) -> None:
    cfg, selected = selected_config(args.source_root, args.sweep_root)
    rows, profiles = collect_rows(cfg, selected)
    destination = selected / "report"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "barriers.json", rows)
    write_json(destination / "profiles.json", profiles)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    grouped = defaultdict(list)
    for row in rows:
        for field in ("chord", "worse"):
            grouped[(row["method"], row["subset"], row["metric"], field)].append(
                float(row[field])
            )
    for (method, subset, metric, field), values in sorted(grouped.items()):
        aggregates.append(
            dict(
                method=method,
                subset=subset,
                metric=metric,
                barrier=field,
                n=len(values),
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                maximum=max(values),
            )
        )
    write_json(destination / "aggregates.json", aggregates)
    figures = write_matrix_figures(cfg, rows, destination)
    write_absolute_profiles(profiles, destination)
    base = json.loads((args.sweep_root.resolve() / "selected_base.json").read_text())
    scale = json.loads((args.sweep_root.resolve() / "selected_scale.json").read_text())
    summary = dict(
        source_root=str(args.source_root.resolve()),
        selected_root=str(selected),
        pair_count=len(pairs(cfg)),
        stages=cfg["stages"],
        methods=list(METHODS),
        selected_base=base["selected"],
        selected_scale=scale["selected"],
        aggregates=aggregates,
        figures=figures + ["absolute_profiles"],
        selection_data="validation selection subset",
        reported_data=["disjoint train evaluation subset", "frozen test evaluation subset"],
    )
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["count", "evaluate", "report"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    cfg, _ = selected_config(args.source_root, args.sweep_root)
    if args.operation == "count":
        print(len(pairs(cfg)))
    elif args.operation == "evaluate":
        evaluate_selected(args)
    else:
        report_selected(args)


if __name__ == "__main__":
    main()
