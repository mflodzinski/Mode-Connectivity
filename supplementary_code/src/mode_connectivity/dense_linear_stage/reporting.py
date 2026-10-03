"""Paper-oriented matrices, stage slices, and absolute interpolation profiles."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from .alignment import METHODS
from .protocol import pair_dir, root, write_json


METHOD_LABELS = {
    "raw": "Raw",
    "wm": "WM",
    "wm_scale": "WM+S",
    "sinkhorn": "Sinkhorn",
    "sinkhorn_scale_joint": "Sinkhorn+S (joint)",
    "sinkhorn_scale_finetune": "Sinkhorn+S (finetune)",
}
SELECTION_KINDS = ("permutation_only", "overall")
SUBSETS = ("train_report", "test_full")
METRICS = ("loss", "error")
BARRIERS = ("chord", "worse")


def _cell_choices(cell):
    """Read the two-choice schema, with a conservative legacy fallback."""
    if "choices" in cell:
        return {
            kind: cell["choices"][kind]["method"] for kind in SELECTION_KINDS
        }
    overall = cell["method"]
    permutation_only = overall if overall in ("wm", "sinkhorn") else "wm"
    return {"permutation_only": permutation_only, "overall": overall}


def _collect(cfg):
    selections = json.loads((root(cfg) / "selections.json").read_text())["selections"]
    rows, profiles = [], {}
    for replicate in range(len(cfg["seed_pairs"])):
        for left in cfg["stages"]:
            for right in cfg["stages"]:
                path = pair_dir(cfg, replicate, left, right) / "full_profiles.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing full profile: {path}")
                value = json.loads(path.read_text())
                selected = _cell_choices(selections[f"{left:03d}_{right:03d}"])
                selected_union = set(selected.values())
                methods = METHODS if replicate == 0 else tuple(
                    method for method in METHODS if method in selected_union
                )
                for method in methods:
                    for subset in SUBSETS:
                        if replicate == 0:
                            resolutions = ["coarse"]
                            if method in selected_union:
                                resolutions.append("selected_dense")
                        else:
                            resolutions = ["selected_dense"]
                        for resolution in resolutions:
                            key = f"{method}/{subset}/{resolution}"
                            if key not in value:
                                raise KeyError(f"Missing {key} in {path}")
                            profile = value[key]
                            profiles[(replicate, left, right, method, subset, resolution)] = profile
                            for metric in METRICS:
                                for barrier in BARRIERS:
                                    rows.append(
                                        dict(
                                            replicate=replicate,
                                            left_epoch=left,
                                            right_epoch=right,
                                            method=method,
                                            selected=method == selected["overall"],
                                            selected_overall=method == selected["overall"],
                                            selected_permutation_only=(
                                                method == selected["permutation_only"]
                                            ),
                                            resolution=resolution,
                                            subset=subset,
                                            metric=metric,
                                            barrier=barrier,
                                            value=float(profile[metric][barrier]),
                                            points=len(profile["alphas"]),
                                        )
                                    )
    return rows, profiles, selections


def _selected_lookup(rows, selection_kind):
    field = f"selected_{selection_kind}"
    return {
        (
            row["replicate"], row["left_epoch"], row["right_epoch"],
            row["subset"], row["metric"], row["barrier"],
        ): row["value"]
        for row in rows
        if row[field] and row["resolution"] == "selected_dense"
    }


def _panel_specs():
    return [
        ("train_report", "loss", "Training-subset loss"),
        ("train_report", "error", "Training-subset error (pp)"),
        ("test_full", "loss", "Test loss"),
        ("test_full", "error", "Test error (pp)"),
    ]


def _replicate_values(cfg, lookup, pairs, subset, metric, barrier):
    return np.asarray([
        [lookup[(replicate, left, right, subset, metric, barrier)] for left, right in pairs]
        for replicate in range(len(cfg["seed_pairs"]))
    ])


def _mean_std(values):
    ddof = 1 if values.shape[0] > 1 else 0
    return values.mean(axis=0), values.std(axis=0, ddof=ddof)


def _plot_stage_lines(cfg, rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = cfg["stages"]
    progress = np.asarray(cfg["stage_progress"], dtype=float)
    outputs = []
    for selection_kind in SELECTION_KINDS:
        lookup = _selected_lookup(rows, selection_kind)
        selection_title = selection_kind.replace("_", " ")
        for barrier in BARRIERS:
            for kind in ("diagonal", "final_slices"):
                fig, axes = plt.subplots(
                    2, 2, figsize=(11.2, 7.5), sharex=True,
                    constrained_layout=True,
                )
                for axis, (subset, metric, title) in zip(axes.flat, _panel_specs()):
                    if kind == "diagonal":
                        pairs = [(epoch, epoch) for epoch in stages]
                        values = _replicate_values(
                            cfg, lookup, pairs, subset, metric, barrier
                        )
                        mean, std = _mean_std(values)
                        axis.plot(progress, mean, marker="o", label="A n – B n")
                        axis.fill_between(progress, mean - std, mean + std, alpha=.2)
                    else:
                        final = stages[-1]
                        for label, pairs, style in (
                            ("A final – B n", [(final, epoch) for epoch in stages], "-"),
                            ("A n – B final", [(epoch, final) for epoch in stages], "--"),
                        ):
                            values = _replicate_values(
                                cfg, lookup, pairs, subset, metric, barrier
                            )
                            mean, std = _mean_std(values)
                            axis.plot(
                                progress, mean, marker="o", linestyle=style, label=label
                            )
                            axis.fill_between(
                                progress, mean - std, mean + std, alpha=.15
                            )
                    axis.set_title(title)
                    axis.grid(alpha=.2)
                    axis.set_xticks(progress, stages, rotation=45)
                    axis.set_xlabel("Training progress (epoch label)")
                    axis.set_ylabel(f"$B_{{\\mathrm{{{barrier}}}}}$")
                axes[0, 0].legend(fontsize=8)
                fig.suptitle(f"Validation-selected {selection_title}")
                stem = f"selected_{selection_kind}_{kind}_{barrier}"
                fig.savefig(destination / f"{stem}.pdf")
                fig.savefig(destination / f"{stem}.png", dpi=180)
                plt.close(fig)
                outputs.append(stem)
    return outputs


def _plot_matrices(cfg, rows, selections, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = cfg["stages"]
    outputs = []
    for selection_kind in SELECTION_KINDS:
        lookup = _selected_lookup(rows, selection_kind)
        selection_title = selection_kind.replace("_", " ")
        for subset, metric, label in _panel_specs():
            for barrier in BARRIERS:
                matrices = np.asarray([
                    [
                        [
                            lookup[(replicate, left, right, subset, metric, barrier)]
                            for right in stages
                        ]
                        for left in stages
                    ]
                    for replicate in range(len(cfg["seed_pairs"]))
                ])
                values, deviations = _mean_std(matrices)
                fig, axis = plt.subplots(
                    figsize=(11.2, 9.4), constrained_layout=True
                )
                image = axis.imshow(
                    values, origin="lower", cmap="viridis", aspect="equal"
                )
                spread = float(values.max() - values.min())
                threshold = float(values.min() + .58 * spread)
                for i, left in enumerate(stages):
                    for j, right in enumerate(stages):
                        selected = _cell_choices(
                            selections[f"{left:03d}_{right:03d}"]
                        )[selection_kind]
                        color = "white" if values[i, j] > threshold else "black"
                        axis.text(
                            j, i,
                            f"{values[i, j]:.3f}\n±{deviations[i, j]:.3f}\n"
                            f"{METHOD_LABELS[selected]}",
                            ha="center", va="center", fontsize=5.4, color=color,
                        )
                axis.set(
                    title=(
                        f"{label}: {selection_title} {barrier} barrier "
                        "(mean ± SD, 3 seed pairs)"
                    ),
                    xlabel="B completed epoch",
                    ylabel="A completed epoch",
                    xticks=range(len(stages)),
                    yticks=range(len(stages)),
                    xticklabels=stages,
                    yticklabels=stages,
                )
                axis.tick_params(axis="x", rotation=45)
                fig.colorbar(
                    image, ax=axis, label=f"{label} $B_{{\\mathrm{{{barrier}}}}}$"
                )
                stem = f"matrix_{selection_kind}_{subset}_{metric}_{barrier}"
                fig.savefig(destination / f"{stem}.pdf")
                fig.savefig(destination / f"{stem}.png", dpi=180)
                plt.close(fig)
                outputs.append(stem)
    return outputs


def _role_label(roles):
    labels = {
        "permutation_only": "permutation-only",
        "overall": "overall",
    }
    return " & ".join(labels[role] for role in SELECTION_KINDS if role in roles)


def _plot_absolute_profiles(cfg, profiles, selections, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output = destination / "absolute_profiles_all_stage_pairs.pdf"
    with PdfPages(output) as pdf:
        for left in cfg["stages"]:
            for right in cfg["stages"]:
                selected = _cell_choices(selections[f"{left:03d}_{right:03d}"])
                roles = defaultdict(list)
                for kind, method in selected.items():
                    roles[method].append(kind)
                fig, axes = plt.subplots(
                    2, 2, figsize=(11.0, 7.6), constrained_layout=True
                )
                for axis, (subset, metric, title) in zip(axes.flat, _panel_specs()):
                    values_key = "losses" if metric == "loss" else "errors"
                    for method in METHODS:
                        if method in roles:
                            curves = [
                                profiles[
                                    (replicate, left, right, method, subset, "selected_dense")
                                ]
                                for replicate in range(len(cfg["seed_pairs"]))
                            ]
                            grid = np.asarray(curves[0]["alphas"], dtype=float)
                            matrix = np.asarray(
                                [curve[values_key] for curve in curves], dtype=float
                            )
                            mean, std = _mean_std(matrix)
                            role = _role_label(roles[method])
                            linestyle = "--" if roles[method] == ["permutation_only"] else "-"
                            axis.plot(
                                grid, mean,
                                label=f"{METHOD_LABELS[method]} ({role}, mean)",
                                linewidth=2.4, linestyle=linestyle,
                            )
                            axis.fill_between(grid, mean - std, mean + std, alpha=.16)
                        else:
                            curve = profiles[(0, left, right, method, subset, "coarse")]
                            axis.plot(
                                curve["alphas"], curve[values_key],
                                label=f"{METHOD_LABELS[method]} (seed pair 0)",
                                linewidth=1.0, alpha=.55,
                            )
                    axis.set(
                        title=title,
                        xlabel="Interpolation coefficient α",
                        ylabel="Cross-entropy" if metric == "loss" else "Error (%)",
                    )
                    axis.grid(alpha=.2)
                axes[0, 0].legend(ncol=2, fontsize=6.8)
                fig.suptitle(
                    f"A epoch {left} ↔ B epoch {right}; "
                    f"permutation-only: {METHOD_LABELS[selected['permutation_only']]}; "
                    f"overall: {METHOD_LABELS[selected['overall']]}"
                )
                pdf.savefig(fig)
                plt.close(fig)
    return output.name


def report(cfg):
    destination = root(cfg) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    rows, profiles, selections = _collect(cfg)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "barriers.json", rows)

    aggregates = []
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["selected_permutation_only"], row["selected_overall"],
            row["resolution"], row["subset"], row["metric"], row["barrier"],
        )
        grouped[key].append(row["value"])
    for key, values in sorted(grouped.items(), key=str):
        (
            method, selected_permutation_only, selected_overall,
            resolution, subset, metric, barrier,
        ) = key
        aggregates.append(dict(
            method=method,
            selected_permutation_only=selected_permutation_only,
            selected_overall=selected_overall,
            resolution=resolution,
            subset=subset,
            metric=metric,
            barrier=barrier,
            count=len(values),
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        ))
    write_json(destination / "aggregates.json", aggregates)

    selection_counts = {
        kind: dict(Counter(
            _cell_choices(cell)[kind] for cell in selections.values()
        ))
        for kind in SELECTION_KINDS
    }
    write_json(destination / "selection_counts.json", selection_counts)

    figures = _plot_stage_lines(cfg, rows, destination)
    figures += _plot_matrices(cfg, rows, selections, destination)
    figures.append(_plot_absolute_profiles(cfg, profiles, selections, destination))
    reuse_path = root(cfg) / "reuse_inventory.json"
    reuse = json.loads(reuse_path.read_text()) if reuse_path.exists() else None
    summary = dict(
        dataset=cfg["dataset"],
        model=cfg.get("model", "FashionMLP10x512"),
        stages=cfg["stages"],
        stage_progress=cfg["stage_progress"],
        pair_cells=len(cfg["stages"]) ** 2,
        replicates=len(cfg["seed_pairs"]),
        methods=list(METHODS),
        alignment_train_examples=int(cfg["expected_train_examples"]),
        train_report_examples=int(cfg["train_report_size"]),
        test_examples=10000,
        all_method_points=int(cfg["profile_points_all"]),
        selected_method_points=int(cfg["profile_points_selected"]),
        selection=(
            "For each ordered cell, seed pair (0,1) independently selects the best "
            "permutation-only method from {WM, Sinkhorn} and the best overall method "
            "from all six candidates on validation loss B_worse, with validation loss "
            "B_chord, maximum loss, mean loss, and fixed method order as tie breakers. "
            "Both choices are frozen and replicated on two held-out seed pairs; test "
            "data are untouched until final evaluation."
        ),
        selection_counts=selection_counts,
        reuse=reuse,
        figures=figures,
    )
    write_json(destination / "summary.json", summary)
    return summary
