"""Standalone scientific figures and machine-readable aggregate results."""

import csv
import json
import shutil
import subprocess

import numpy as np

from .evaluation import METHODS
from .protocol import root, write_json
from .scheduling import build_dag, completed


def report(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    tasks = build_dag(cfg)
    missing = [
        t["id"] for t in tasks if t["operation"] != "report" and not completed(cfg, t)
    ]
    if missing:
        raise RuntimeError(f"Report requires every expected task: {missing}")
    directory = root(cfg) / "report"
    directory.mkdir(parents=True, exist_ok=True)
    stages, replicates = cfg["stages"], len(cfg["seed_pairs"])
    rows = []
    for rep in range(replicates):
        for ea in stages:
            for eb in stages:
                path = (
                    root(cfg)
                    / "pairs"
                    / str(rep)
                    / f"{ea:03d}_{eb:03d}"
                    / "profiles.json"
                )
                profiles = json.loads(path.read_text())
                for method in METHODS:
                    for subset in ["train_eval", "test_eval"]:
                        row = profiles[f"{method}/{subset}"]
                        if row["epochs"] != [ea, eb] or row["replicate"] != rep:
                            raise ValueError(f"Mislabeled profile: {path}")
                        rows.append(row)
    write_json(directory / "profiles.json", rows)
    aggregates = []
    for subset in ["train_eval", "test_eval"]:
        for metric in ["loss", "error"]:
            for barrier in ["chord", "worse"]:
                values = np.empty((replicates, len(METHODS), len(stages), len(stages)))
                for row in rows:
                    if row["subset"] == subset:
                        i, j = map(stages.index, row["epochs"])
                        values[row["replicate"], METHODS.index(row["method"]), i, j] = (
                            row[metric][barrier]
                        )
                mean = values.mean(axis=0)
                std = (
                    values.std(axis=0, ddof=1)
                    if replicates > 1
                    else np.zeros_like(mean)
                )
                aggregates.append(
                    dict(
                        subset=subset,
                        metric=metric,
                        barrier=barrier,
                        methods=METHODS,
                        mean=mean.tolist(),
                        std=std.tolist(),
                    )
                )
                fig, axes = plt.subplots(
                    replicates + 1,
                    len(METHODS),
                    figsize=(22, 3 * (replicates + 1)),
                    squeeze=False,
                )
                vmax = max(float(values.max()), 1e-8)
                for r in range(replicates + 1):
                    for m, method in enumerate(METHODS):
                        ax = axes[r, m]
                        mat = values[r, m] if r < replicates else mean[m]
                        im = ax.imshow(mat, vmin=0, vmax=vmax, cmap="viridis")
                        ax.set(
                            xticks=range(len(stages)),
                            xticklabels=stages,
                            yticks=range(len(stages)),
                            yticklabels=stages,
                            title=f"{method}: "
                            + (f"pair {r}" if r < replicates else "mean"),
                            xlabel="B epoch",
                            ylabel="A epoch",
                        )
                        for i in range(len(stages)):
                            for j in range(len(stages)):
                                label = f"{mat[i,j]:.3f}"
                                if r == replicates:
                                    label += f"\n±{std[m,i,j]:.3f}"
                                ax.text(
                                    j,
                                    i,
                                    label,
                                    ha="center",
                                    va="center",
                                    fontsize=7,
                                    color="white",
                                )
                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
                fig.suptitle(
                    f"{subset}: {metric} {barrier} barrier (error in percentage points)"
                )
                fig.savefig(
                    directory / f"matrix_{subset}_{metric}_{barrier}.png",
                    dpi=160,
                    bbox_inches="tight",
                )
                plt.close(fig)
                gains = [
                    ("raw", "base"),
                    ("wm", "base"),
                    ("base", "scale"),
                    ("continue", "scale"),
                ]
                fig, axes = plt.subplots(1, len(gains), figsize=(14, 3))
                for ax, (left, right) in zip(axes, gains):
                    gain = (
                        values[:, METHODS.index(left)] - values[:, METHODS.index(right)]
                    )
                    limit = max(float(np.abs(gain).max()), 1e-8)
                    im = ax.imshow(
                        gain.mean(0), cmap="coolwarm", vmin=-limit, vmax=limit
                    )
                    ax.set(
                        title=f"{left} minus {right}",
                        xticks=range(len(stages)),
                        xticklabels=stages,
                        yticks=range(len(stages)),
                        yticklabels=stages,
                        xlabel="B epoch",
                        ylabel="A epoch",
                    )
                    fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(
                    directory / f"gains_{subset}_{metric}_{barrier}.png", dpi=160
                )
                plt.close(fig)
    write_json(directory / "aggregates.json", aggregates)
    # Full path shapes reveal unequal endpoint losses and dips hidden by scalar barriers.
    with PdfPages(directory / "absolute_profiles.pdf") as pdf:
        for rep in range(replicates):
            for ea in stages:
                for eb in stages:
                    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
                    for row in rows:
                        if row["replicate"] != rep or row["epochs"] != [ea, eb]:
                            continue
                        i = ["train_eval", "test_eval"].index(row["subset"])
                        for j, metric in enumerate(["losses", "errors"]):
                            axes[i, j].plot(
                                row["alphas"], row[metric], label=row["method"]
                            )
                            axes[i, j].set(
                                title=f"{row['subset']} {metric}", xlabel="alpha"
                            )
                    axes[0, 0].legend(fontsize=7)
                    fig.suptitle(f"Seed pair {rep}, epochs {ea} → {eb}")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
    for subset in ["train_eval", "test_eval"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for method in ["raw", "wm", "transfer_wm", "transfer_base"]:
            for ax, metric in zip(axes, ["loss", "error"]):
                curves = []
                for rep in range(replicates):
                    values = []
                    for epoch in cfg.get("control_checkpoints", cfg["checkpoints"]):
                        path = (
                            root(cfg)
                            / "controls"
                            / str(rep)
                            / f"diagonal_{epoch:03d}.json"
                        )
                        values.append(
                            json.loads(path.read_text())[f"{method}/{subset}"][metric][
                                "chord"
                            ]
                        )
                    curves.append(values)
                    control_epochs = cfg.get("control_checkpoints", cfg["checkpoints"])
                    ax.plot(control_epochs, values, alpha=0.2, linewidth=0.7)
                curves = np.asarray(curves)
                mean = curves.mean(0)
                sd = curves.std(0, ddof=1) if replicates > 1 else np.zeros_like(mean)
                control_epochs = cfg.get("control_checkpoints", cfg["checkpoints"])
                ax.plot(control_epochs, mean, label=method)
                ax.fill_between(control_epochs, mean - sd, mean + sd, alpha=0.12)
                ax.set(
                    title=f"{subset} {metric} chord barrier", xlabel="completed epoch"
                )
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(directory / f"diagonal_{subset}.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for rep, seeds in enumerate(cfg["seed_pairs"]):
            for seed in seeds:
                curves = [
                    json.loads(
                        (
                            root(cfg)
                            / "controls"
                            / str(rep)
                            / f"within_{seed}_{e:03d}.json"
                        ).read_text()
                    )[f"raw/{subset}"]
                    for e in stages[:-1]
                ]
                for ax, metric in zip(axes, ["loss", "error"]):
                    ax.plot(
                        stages[:-1],
                        [c[metric]["chord"] for c in curves],
                        marker="o",
                        label=f"seed {seed}",
                    )
                    ax.set(
                        title=f"{subset}: within-run {metric}", xlabel="starting epoch"
                    )
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(directory / f"within_{subset}.png", dpi=160)
        plt.close(fig)
    audit = json.loads((root(cfg) / "audit/discrepancies.json").read_text())
    write_json(directory / "audit.json", audit)
    endpoint_results = []
    for rep, seeds in enumerate(cfg["seed_pairs"]):
        for seed in seeds:
            metrics = json.loads(
                (
                    root(cfg)
                    / "controls"
                    / str(rep)
                    / f"endpoint_{seed}_full_test.json"
                ).read_text()
            )
            endpoint_results.append(
                dict(replicate=rep, seed=seed, epoch=cfg["epochs"], **metrics)
            )
    write_json(directory / "final_endpoints_full_test.json", endpoint_results)
    runtime = []
    for task in tasks:
        if task["operation"] == "report":
            continue
        status = json.loads((root(cfg) / "status" / f"{task['id']}.json").read_text())
        for attempt, timing in enumerate(status["attempts"]):
            runtime.append(
                dict(
                    task=task["id"],
                    operation=task["operation"],
                    attempt=attempt,
                    **timing,
                )
            )
    fields = sorted(set().union(*(r.keys() for r in runtime)))
    with (directory / "runtime.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runtime)
    job_ids = sorted({r["slurm_job_id"] for r in runtime if r.get("slurm_job_id")})
    accounting = dict(available=False)
    if shutil.which("sacct") and job_ids:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "-P",
                "--format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES",
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        (directory / "sacct.txt").write_text(result.stdout + result.stderr)
        accounting = dict(available=result.returncode == 0)
    limitations = [
        "Sampled grids/subsets do not establish an everywhere low-loss path.",
        "Heuristic alignment failure does not prove disconnected solutions.",
        "Runtime and memory must be measured on DAIC; local tests do not validate allocations.",
    ]
    if cfg.get("experiment_profile") == "git_rebasin_cifar10_mlp":
        limitations += [
            "This is a PyTorch protocol replication; JAX and PyTorch RNG streams and floating-point trajectories are not bitwise identical.",
            "The paper trains on all 50,000 examples, so alignment selection data was seen during endpoint training; no test result selects an alignment.",
            "The literal reference implementation uses run-seeded augmentation as well as run-seeded initialization and batch order.",
        ]
    summary = dict(
        profile_count=len(rows),
        replicates=replicates,
        final_endpoint_full_test=endpoint_results,
        accounting=accounting,
        reference=cfg.get("reference"),
        limitations=limitations,
    )
    write_json(directory / "summary.json", summary)
    return summary
