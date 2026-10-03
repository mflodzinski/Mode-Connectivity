"""Machine-readable summaries and thesis-ready nonlinear-stage figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import validation_path
from .protocol import load, path_dir, primary_pairs, root, selected_dir, write_json


def _profile_row(pair, subset, method, profile, selection, geometry):
    return dict(
        pair=pair["id"],
        replicate=pair["replicate"],
        kind=pair["kind"],
        n=pair["n"],
        left_seed=pair["left_seed"],
        left_epoch=pair["left_epoch"],
        right_seed=pair["right_seed"],
        right_epoch=pair["right_epoch"],
        subset=subset,
        method=method,
        family="linear" if method == "linear" else selection["family"],
        restart=-1 if method == "linear" else selection["restart"],
        loss_chord=profile["loss"]["chord"],
        loss_worse=profile["loss"]["worse"],
        error_chord=profile["error"]["chord"],
        error_worse=profile["error"]["worse"],
        max_loss=profile["max_loss"],
        max_error=profile["max_error"],
        normalized_path_length=(1.0 if method == "linear" else geometry["normalized_path_length"]),
        control_displacement=(0.0 if method == "linear" else geometry["control_displacement"]),
    )


def report(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    destination = root(cfg) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    rows, profile_records, restart_rows = [], [], []
    confirmations = json.loads((root(cfg) / "confirmation_targets.json").read_text())
    for pair in primary_pairs(cfg):
        selection = json.loads((selected_dir(cfg, pair["id"]) / "selection.json").read_text())
        test = json.loads((selected_dir(cfg, pair["id"]) / "test.json").read_text())
        chosen = load(selection["artifact"])
        validation = json.loads(
            validation_path(cfg, pair["id"], selection["family"], selection["restart"]).read_text()
        )
        for subset, profiles in [
            ("train_eval", validation["subsets"]["train_eval"]),
            ("validation_audit", validation["subsets"]["validation_audit"]),
            ("test_eval", dict(linear=test["linear"], curve=test["curve"])),
        ]:
            for method in ["linear", "curve"]:
                rows.append(
                    _profile_row(
                        pair, subset, method, profiles[method], selection, chosen["geometry"]
                    )
                )
                profile_records.append(
                    dict(
                        pair=pair, subset=subset, method=method,
                        family=("linear" if method == "linear" else selection["family"]),
                        restart=(-1 if method == "linear" else selection["restart"]),
                        profile=profiles[method],
                    )
                )
        for candidate in selection["candidates"]:
            value = json.loads(
                validation_path(
                    cfg, pair["id"], candidate["family"], candidate["restart"]
                ).read_text()
            )["subsets"]["validation_audit"]["curve"]
            restart_rows.append(
                dict(
                    pair=pair["id"], replicate=pair["replicate"], kind=pair["kind"], n=pair["n"],
                    family=candidate["family"], restart=candidate["restart"],
                    loss_chord=value["loss"]["chord"], error_chord=value["error"]["chord"],
                    selected=(candidate["family"] == selection["family"] and candidate["restart"] == selection["restart"]),
                )
            )
    fields = list(rows[0])
    with (destination / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "profiles.json", profile_records)
    write_json(destination / "restart_results.json", restart_rows)
    write_json(
        destination / "initialization_controls.json",
        [row for row in rows if row["n"] == 0],
    )

    def plot_kind(filename, kinds, titles):
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4), squeeze=False)
        for ax, kind, title in zip(axes[0], kinds, titles):
            subset_rows = [
                row for row in rows if row["subset"] == "test_eval" and row["kind"] == kind
            ]
            for method, color in [("linear", "tab:orange"), ("curve", "tab:blue")]:
                by_rep = []
                for rep in range(len(cfg["source_pairs"])):
                    values = sorted(
                        [r for r in subset_rows if r["method"] == method and r["replicate"] == rep],
                        key=lambda r: r["n"],
                    )
                    x = np.asarray([r["n"] for r in values])
                    y = np.asarray([r["loss_chord"] for r in values])
                    by_rep.append(y)
                    ax.plot(x, y, color=color, alpha=0.22, linewidth=1)
                matrix = np.asarray(by_rep)
                mean = matrix.mean(0)
                sd = matrix.std(0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
                ax.plot(x, mean, color=color, marker="o", label=method)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14)
            ax.axvspan(-1, 0.5, color="0.9", alpha=0.6)
            ax.axvline(0, color="black", linestyle=":", linewidth=0.7)
            ax.set(title=title, xlabel="intermediate epoch n", ylabel="test loss chord barrier")
            ax.legend()
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=180)
        plt.close(fig)

    plot_kind("same_stage.png", ["same"], ["Independent models: n–n"])
    plot_kind(
        "cross_stage.png", ["final_left", "final_right"],
        ["A(200)–B(n)", "A(n)–B(200)"],
    )
    plot_kind(
        "within_run.png", ["within_a", "within_b"],
        ["A(n)–A(200)", "B(n)–B(200)"],
    )

    kinds = ["same", "final_left", "final_right", "within_a", "within_b"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(24, 4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        gains = []
        for rep in range(len(cfg["source_pairs"])):
            current = [
                row for row in rows
                if row["subset"] == "test_eval" and row["kind"] == kind
                and row["replicate"] == rep
            ]
            linear = {row["n"]: row["loss_chord"] for row in current if row["method"] == "linear"}
            curve = {row["n"]: row["loss_chord"] for row in current if row["method"] == "curve"}
            x = np.asarray(sorted(linear))
            gain = np.asarray([linear[n] - curve[n] for n in x])
            gains.append(gain)
            ax.plot(x, gain, color="tab:green", alpha=0.22, linewidth=1)
        matrix = np.asarray(gains)
        mean = matrix.mean(0)
        sd = matrix.std(0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        ax.plot(x, mean, color="tab:green", marker="o")
        ax.fill_between(x, mean - sd, mean + sd, color="tab:green", alpha=0.14)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(title=kind, xlabel="n", ylabel="linear − nonlinear loss barrier")
    fig.tight_layout()
    fig.savefig(destination / "barrier_reduction.png", dpi=180)
    plt.close(fig)

    test_curves = [r for r in rows if r["subset"] == "test_eval" and r["method"] == "curve"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(
        [r["n"] for r in test_curves], [r["normalized_path_length"] for r in test_curves],
        c=[r["replicate"] for r in test_curves], s=14,
    )
    axes[1].scatter(
        [r["n"] for r in test_curves], [r["control_displacement"] for r in test_curves],
        c=[r["replicate"] for r in test_curves], s=14,
    )
    axes[0].set(xlabel="n", ylabel="path length / endpoint distance", title="Path length")
    axes[1].set(xlabel="n", ylabel="control displacement", title="Control geometry")
    fig.tight_layout()
    fig.savefig(destination / "path_geometry.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for family, marker in [("bezier", "o"), ("polygon", "s")]:
        values = [row for row in restart_rows if row["family"] == family]
        ax.scatter(
            [row["loss_chord"] for row in values],
            [row["error_chord"] for row in values],
            marker=marker, alpha=0.55, label=family,
        )
    ax.axvline(cfg["loss_confirmation_threshold"], color="black", linestyle=":")
    ax.axhline(cfg["error_confirmation_threshold"], color="black", linestyle=":")
    ax.set(xlabel="validation loss chord barrier", ylabel="validation error barrier (pp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "restart_results.png", dpi=180)
    plt.close(fig)

    representative = {
        ("same", 0), ("same", 1), ("same", 60), ("same", 200),
        ("final_left", 1), ("final_right", 1), ("within_a", 1),
    }
    with PdfPages(destination / "absolute_profiles.pdf") as pdf:
        for pair in primary_pairs(cfg):
            if (pair["kind"], pair["n"]) not in representative:
                continue
            records = [
                record for record in profile_records
                if record["pair"]["id"] == pair["id"] and record["subset"] == "test_eval"
            ]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for record in records:
                profile = record["profile"]
                label = record["method"] if record["method"] == "linear" else record["family"]
                axes[0].plot(profile["alphas"], profile["losses"], label=label)
                axes[1].plot(profile["alphas"], profile["errors"], label=label)
            axes[0].set(title="test cross-entropy", xlabel="t")
            axes[1].set(title="test error (%)", xlabel="t")
            axes[0].legend()
            fig.suptitle(
                f"replicate {pair['replicate']}: {pair['left_seed']}@{pair['left_epoch']} → "
                f"{pair['right_seed']}@{pair['right_epoch']} ({pair['kind']})"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    selected_successes = sum(
        row["loss_chord"] <= cfg["loss_confirmation_threshold"]
        and row["error_chord"] <= cfg["error_confirmation_threshold"]
        for row in test_curves
    )
    endpoint_results = []
    for rep in range(len(cfg["source_pairs"])):
        endpoint_results += json.loads(
            (root(cfg) / "endpoint_test" / f"replicate_{rep}.json").read_text()
        )
    runtime_rows, issue_rows = [], []
    for status_path in sorted((root(cfg) / "status").glob("*.json")):
        status = json.loads(status_path.read_text())
        for attempt_number, attempt in enumerate(status.get("attempts", []), start=1):
            runtime_rows.append(
                dict(
                    task=status_path.stem,
                    operation=status.get("task", {}).get("operation"),
                    attempt=attempt_number,
                    status=attempt.get("status"),
                    seconds=attempt.get("seconds"),
                    host_peak_bytes=attempt.get("host_peak_bytes_upper_bound"),
                    gpu_peak_allocated_bytes=attempt.get("gpu_peak_allocated_bytes"),
                    slurm_job_id=attempt.get("slurm_job_id"),
                )
            )
        if status.get("status") not in {"complete", "running"}:
            issue_rows.append(
                dict(task=status_path.stem, status=status.get("status"), error=status.get("error", ""))
            )
    runtime_fields = list(runtime_rows[0]) if runtime_rows else ["task"]
    with (destination / "runtime.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime_fields)
        writer.writeheader()
        writer.writerows(runtime_rows)
    with (destination / "issues.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "status", "error"])
        writer.writeheader()
        writer.writerows(issue_rows)
    restart_successes = sum(
        row["loss_chord"] <= cfg["loss_confirmation_threshold"]
        and row["error_chord"] <= cfg["error_confirmation_threshold"]
        for row in restart_rows
    )
    summary = dict(
        architecture="VGG11",
        raw_endpoints=True,
        stages=cfg["stage_epochs"],
        replicates=len(cfg["source_pairs"]),
        primary_paths=len(primary_pairs(cfg)),
        confirmation_targets=len(confirmations["targets"]),
        selected_paths_below_test_threshold=selected_successes,
        validation_restart_successes=restart_successes,
        validation_restarts=len(restart_rows),
        note=(
            "Epoch zero is an initialization traversal. Residual barriers are optimization "
            "failures for the tested path families, not proofs of disconnectedness."
        ),
        endpoint_full_test=endpoint_results,
    )
    write_json(destination / "summary.json", summary)
    return summary
