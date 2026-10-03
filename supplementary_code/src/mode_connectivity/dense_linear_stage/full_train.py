"""Post-hoc full endpoint-training-split evaluation of frozen alignments."""

from __future__ import annotations

import csv
import fcntl
import gc
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .alignment import METHODS, aligned_state, artifact_path, pair_models
from .models import device_state, interpolated_profile
from .protocol import (
    Data,
    StopFlag,
    digest,
    file_hash,
    root,
    verify_protocol,
    write_json,
)
from .reporting import METHOD_LABELS, METRICS


METHOD_SHARDS = (
    ("raw", "wm"),
    ("wm_scale", "sinkhorn"),
    ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"),
)


def load_completed_config(experiment_root: str | Path) -> dict:
    """Load the immutable config recorded by a completed final benchmark."""

    experiment_root = Path(experiment_root).resolve()
    manifest_path = experiment_root / "tasks.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")
    cfg = json.loads(manifest_path.read_text())["config"]
    cfg["output_root"] = str(experiment_root)
    if cfg.get("benchmark_mode") != "final_alignment":
        raise ValueError("Full-train evaluation is restricted to final_alignment roots.")
    if len(cfg["stages"]) != 1 or len(cfg["seed_pairs"]) != 3:
        raise ValueError("Expected one final epoch and three endpoint pairs.")
    # The evaluation code was added after the frozen benchmark. All scientific
    # inputs remain checked, but a newer reporting-code hash is intentionally
    # allowed and recorded in each output's provenance.
    verify_protocol(cfg, require_current_code=False)
    report = experiment_root / "report" / "summary.json"
    if not report.exists():
        raise RuntimeError(f"Original final benchmark is not complete: {report}")
    return cfg


def evaluation_tasks() -> list[dict]:
    """Return nine independent pair/method-shard tasks."""

    return [
        dict(index=3 * replicate + shard, replicate=replicate, methods=list(methods))
        for replicate in range(3)
        for shard, methods in enumerate(METHOD_SHARDS)
    ]


def profile_path(cfg, replicate: int, method: str) -> Path:
    return root(cfg) / "full_train_profiles" / f"r{replicate}" / f"{method}.json"


def _source_hash(cfg, replicate: int, epoch: int, method: str) -> str | None:
    if method == "raw":
        return None
    path = artifact_path(cfg, replicate, epoch, epoch, method)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen alignment artifact: {path}")
    return file_hash(path)


def _is_complete(cfg, replicate: int, method: str) -> bool:
    path = profile_path(cfg, replicate, method)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        metadata = payload["metadata"]
        return (
            metadata["method"] == method
            and int(metadata["replicate"]) == int(replicate)
            and int(metadata["examples"]) == int(cfg["expected_train_examples"])
            and metadata["subset_hash"]
            == digest(verify_protocol(cfg, require_current_code=False)["indices"]["train_fit"])
            and metadata.get("alignment_sha256")
            == _source_hash(cfg, replicate, int(cfg["stages"][0]), method)
            and len(payload["profile"]["alphas"])
            == int(cfg["profile_points_selected"])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def evaluate_task(experiment_root: str | Path, task_index: int) -> dict:
    """Evaluate one pair and two methods, resuming at method granularity."""

    cfg = load_completed_config(experiment_root)
    tasks = {task["index"]: task for task in evaluation_tasks()}
    if int(task_index) not in tasks:
        raise ValueError(f"task_index must be one of {sorted(tasks)}")
    task = tasks[int(task_index)]
    replicate, methods = int(task["replicate"]), task["methods"]
    lock_path = root(cfg) / "full_train_profiles" / f"task_{int(task_index)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _evaluate_locked(cfg, int(task_index), replicate, methods)


def _evaluate_locked(cfg, task_index: int, replicate: int, methods: list[str]) -> dict:
    """Run a full-train shard while its task lock is held."""

    epoch = int(cfg["stages"][0])
    missing = [method for method in methods if not _is_complete(cfg, replicate, method)]
    if not missing:
        return dict(task=task_index, replicate=replicate, skipped=methods)

    stop = StopFlag()
    data = Data(cfg, require_current_code=False)
    train_indices = data.subsets["indices"]["train_fit"]
    if len(train_indices) != int(cfg["expected_train_examples"]):
        raise ValueError("Frozen train_fit split does not match expected_train_examples.")
    loader = data.loader("train_fit")
    left, _, endpoint_paths = pair_models(cfg, replicate, epoch, epoch)
    left_state = dict(left.named_parameters())
    protocol = json.loads((root(cfg) / "protocol.json").read_text())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    produced = []
    for method in missing:
        stop.check()
        state, artifact = aligned_state(cfg, replicate, epoch, epoch, method)
        state = device_state(state, cfg["device"])
        profile = interpolated_profile(
            left,
            left_state,
            state,
            loader,
            np.linspace(0.0, 1.0, int(cfg["profile_points_selected"])).tolist(),
            stop,
        )
        destination = profile_path(cfg, replicate, method)
        write_json(
            destination,
            dict(
                profile=profile,
                metadata=dict(
                    method=method,
                    subset="train_fit",
                    examples=len(train_indices),
                    subset_hash=digest(train_indices),
                    replicate=replicate,
                    seeds=cfg["seed_pairs"][replicate],
                    epoch=epoch,
                    endpoints=[
                        dict(path=str(path), sha256=file_hash(path))
                        for path in endpoint_paths
                    ],
                    alignment_sha256=_source_hash(cfg, replicate, epoch, method),
                    source_protocol_hash=protocol["hash"],
                    source_code_hash=protocol["code_hash"],
                    evaluation_revision=revision,
                    test_data_used=False,
                ),
            ),
        )
        produced.append(str(destination))
        del state, artifact
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(task=task_index, replicate=replicate, produced=produced)


def missing_task_indices(experiment_root: str | Path) -> list[int]:
    cfg = load_completed_config(experiment_root)
    return [
        task["index"]
        for task in evaluation_tasks()
        if any(
            not _is_complete(cfg, int(task["replicate"]), method)
            for method in task["methods"]
        )
    ]


def _rows_and_profiles(cfg):
    rows, profiles = [], {}
    epoch = int(cfg["stages"][0])
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        for method in METHODS:
            path = profile_path(cfg, replicate, method)
            if not _is_complete(cfg, replicate, method):
                raise FileNotFoundError(f"Incomplete full-train profile: {path}")
            profile = json.loads(path.read_text())["profile"]
            profiles[(replicate, method)] = profile
            for metric in METRICS:
                rows.append(
                    dict(
                        replicate=replicate,
                        left_seed=int(seeds[0]),
                        right_seed=int(seeds[1]),
                        epoch=epoch,
                        method=method,
                        subset="train_fit",
                        examples=int(cfg["expected_train_examples"]),
                        metric=metric,
                        barrier="chord",
                        value=float(profile[metric]["chord"]),
                        points=len(profile["alphas"]),
                    )
                )
    return rows, profiles


def _plot(cfg, rows, profiles, destination: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(METHODS))
    loss_values = {
        method: [
            row["value"]
            for row in rows
            if row["method"] == method and row["metric"] == "loss"
        ]
        for method in METHODS
    }
    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.bar(
        positions,
        [np.mean(loss_values[method]) for method in METHODS],
        yerr=[np.std(loss_values[method], ddof=1) for method in METHODS],
        capsize=4,
    )
    axis.set(
        ylabel="Full training loss barrier",
        xticks=positions,
        xticklabels=[METHOD_LABELS[method] for method in METHODS],
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    barrier_stem = "full_train_loss_barriers"
    fig.savefig(destination / f"{barrier_stem}.pdf")
    fig.savefig(destination / f"{barrier_stem}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for axis, (metric, title, values_key) in zip(
        axes,
        (
            ("loss", "Full endpoint-training loss", "losses"),
            ("error", "Full endpoint-training error (%)", "errors"),
        ),
    ):
        for method in METHODS:
            curves = [profiles[(replicate, method)] for replicate in range(3)]
            grid = np.asarray(curves[0]["alphas"], dtype=float)
            matrix = np.asarray([curve[values_key] for curve in curves], dtype=float)
            axis.plot(grid, matrix.mean(axis=0), label=METHOD_LABELS[method])
            axis.fill_between(
                grid,
                matrix.mean(axis=0) - matrix.std(axis=0, ddof=1),
                matrix.mean(axis=0) + matrix.std(axis=0, ddof=1),
                alpha=0.13,
            )
        axis.set(title=title, xlabel="Interpolation coefficient α")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=7)
    profile_stem = "full_train_absolute_profiles"
    fig.savefig(destination / f"{profile_stem}.pdf")
    fig.savefig(destination / f"{profile_stem}.png", dpi=180)
    plt.close(fig)
    return [barrier_stem, profile_stem]


def report(experiment_root: str | Path) -> dict:
    cfg = load_completed_config(experiment_root)
    rows, profiles = _rows_and_profiles(cfg)
    destination = root(cfg) / "report_full_train"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "barriers.json", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["metric"])].append(row["value"])
    aggregates = []
    for (method, metric), values in sorted(grouped.items()):
        aggregates.append(
            dict(
                method=method,
                subset="train_fit",
                examples=int(cfg["expected_train_examples"]),
                metric=metric,
                barrier="chord",
                count=len(values),
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)),
                values=[float(value) for value in values],
            )
        )
    write_json(destination / "aggregates.json", aggregates)
    summary = dict(
        benchmark_mode="posthoc_full_train_evaluation",
        source_root=str(root(cfg)),
        dataset=cfg["dataset"],
        model=cfg.get("model"),
        epoch=int(cfg["stages"][0]),
        seed_pairs=cfg["seed_pairs"],
        methods=list(METHODS),
        subset="complete endpoint-training split",
        examples=int(cfg["expected_train_examples"]),
        interpolation_points=int(cfg["profile_points_selected"]),
        statistic="mean and sample standard deviation across endpoint pairs",
        test_data_used=False,
        figures=_plot(cfg, rows, profiles, destination),
    )
    write_json(destination / "summary.json", summary)
    return summary


def status(experiment_root: str | Path) -> dict:
    cfg = load_completed_config(experiment_root)
    completed = sum(
        _is_complete(cfg, replicate, method)
        for replicate in range(3)
        for method in METHODS
    )
    result = dict(
        root=str(root(cfg)),
        profiles_complete=completed,
        profiles_total=3 * len(METHODS),
        missing_tasks=missing_task_indices(experiment_root),
        report_complete=(root(cfg) / "report_full_train" / "summary.json").exists(),
    )
    return result
