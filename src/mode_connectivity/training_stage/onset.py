"""Git Re-Basin Figure 3 replication and true-initialization extension."""

from __future__ import annotations

import gc
import json

import numpy as np

from .alignment import weight_artifact
from .geometry import read_model, transformed, equivalence, profile
from .protocol import (
    Data,
    artifact_provenance,
    checkpoint,
    file_hash,
    load,
    root,
    save,
    write_json,
)


def onset_dir(cfg, replicate):
    return root(cfg) / "onset" / str(replicate)


def onset_paths(cfg, replicate, epoch):
    directory = onset_dir(cfg, replicate)
    return (
        directory / f"epoch_{epoch:03d}.json",
        directory / f"permutation_{epoch:03d}.pt",
    )


def evaluate_onset(cfg, replicate, epoch, stop):
    """Fit fresh same-stage WM and evaluate the complete official data grids."""
    directory = onset_dir(cfg, replicate)
    directory.mkdir(parents=True, exist_ok=True)
    destination, permutation_path = onset_paths(cfg, replicate, epoch)
    seeds = cfg["seed_pairs"][replicate]
    endpoint_paths = [
        checkpoint(cfg, seeds[0], epoch),
        checkpoint(cfg, seeds[1], epoch),
    ]
    a, b = [read_model(path, cfg) for path in endpoint_paths]
    if not permutation_path.exists():
        artifact = weight_artifact(a, b, cfg)
        artifact.update(
            transform="weight_matching",
            matching_seed=cfg["onset_matching_seed"],
            **artifact_provenance(cfg, endpoint_paths),
        )
        save(permutation_path, artifact)
    fresh_artifact = load(permutation_path)
    final_epoch = max(cfg["onset_epochs"])
    _, final_permutation_path = onset_paths(cfg, replicate, final_epoch)
    if not final_permutation_path.exists():
        raise FileNotFoundError(
            "The final permutation must be frozen before transfer evaluation."
        )
    final_artifact = load(final_permutation_path)
    data = Data(cfg, allow_test=True)
    alphas = np.linspace(0, 1, cfg["onset_points"]).tolist()
    result = dict(
        replicate=replicate,
        seeds=seeds,
        completed_epoch=epoch,
        # Official checkpoint k is after completed epoch k+1. Initialization
        # has no official checkpoint counterpart.
        official_checkpoint=None if epoch == 0 else epoch - 1,
        exact_figure3_member=1 <= epoch <= 100,
        alphas=alphas,
        endpoint_sha256=[file_hash(path) for path in endpoint_paths],
        methods={},
    )
    selection = data.loader("selection")
    for method, artifact in [
        ("raw", None),
        ("wm", fresh_artifact),
        ("transfer_final_wm", final_artifact),
    ]:
        stop.check()
        aligned = b if artifact is None else transformed(b, artifact, cfg)
        difference = (
            0.0
            if artifact is None
            else equivalence(b, aligned, selection, cfg["device"], cfg)
        )
        method_result = dict(
            alignment_sha256=None
            if artifact is None
            else file_hash(
                permutation_path if method == "wm" else final_permutation_path
            ),
            max_logit_difference=difference,
        )
        for subset in ["train_full", "test_full"]:
            method_result[subset] = profile(
                a,
                aligned,
                data.loader(
                    subset, batch_size=cfg["onset_eval_batch_size"]
                ),
                cfg["device"],
                alphas,
                stop=stop,
            )
        result["methods"][method] = method_result
        if aligned is not b:
            del aligned
        gc.collect()
    write_json(destination, result)
    return dict(
        completed_epoch=epoch,
        official_checkpoint=result["official_checkpoint"],
        methods=list(result["methods"]),
    )


def report_onset(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = root(cfg) / "onset_report"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for replicate in range(len(cfg["seed_pairs"])):
        for epoch in cfg["onset_epochs"]:
            path, _ = onset_paths(cfg, replicate, epoch)
            records.append(json.loads(path.read_text()))
    write_json(directory / "profiles.json", records)

    # Direct counterpart of Figure 3: fresh WM, published scalar, first 25
    # completed epochs. The paper used one pair, so show each replicate rather
    # than hiding variation behind an aggregate.
    fig, axes = plt.subplots(
        len(cfg["seed_pairs"]), 1, figsize=(7, 4 * len(cfg["seed_pairs"])), squeeze=False
    )
    for replicate, ax in enumerate(axes[:, 0]):
        rows = [
            r
            for r in records
            if r["replicate"] == replicate and 1 <= r["completed_epoch"] <= 25
        ]
        rows.sort(key=lambda r: r["completed_epoch"])
        for subset, style in [("train_full", "-"), ("test_full", "--")]:
            ax.plot(
                [r["completed_epoch"] for r in rows],
                [r["methods"]["wm"][subset]["loss"]["git_rebasin"] for r in rows],
                style,
                marker="o",
                markersize=3,
                label=subset.replace("_full", "").title(),
            )
        ax.axhspan(ax.get_ylim()[0], 0, color="grey", alpha=0.15, hatch="//")
        ax.set(
            title=f"Git Re-Basin Figure 3 replication, seed pair {replicate}",
            xlabel="Completed epoch",
            ylabel="Published loss barrier",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "figure3_replication.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for method in ["raw", "wm", "transfer_final_wm"]:
        for ax, subset in zip(axes, ["train_full", "test_full"]):
            for replicate in range(len(cfg["seed_pairs"])):
                rows = sorted(
                    (r for r in records if r["replicate"] == replicate),
                    key=lambda r: r["completed_epoch"],
                )
                ax.plot(
                    [r["completed_epoch"] for r in rows],
                    [r["methods"][method][subset]["loss"]["chord"] for r in rows],
                    label=f"{method}, pair {replicate}",
                )
            ax.set(
                title=f"{subset}: chord barrier",
                xlabel="Completed epoch (0 is true initialization)",
                ylabel="Loss barrier",
            )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "onset_full_chord.png", dpi=180)
    plt.close(fig)
    summary = dict(
        records=len(records),
        replicates=len(cfg["seed_pairs"]),
        epochs=cfg["onset_epochs"],
        paper_mapping="completed epoch e corresponds to official checkpoint e-1",
        published_metric="max(loss)-0.5*(endpoint_a_loss+endpoint_b_loss)",
    )
    write_json(directory / "summary.json", summary)
    return summary
