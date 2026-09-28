#!/usr/bin/env python3
"""Generate compact figures for the VGG11 training-stage supervisor report."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURES = HERE / "figures"
LINEAR_CSV = REPO / "results/dense_linear_stage_vgg11_10k/report/barriers.csv"
NONLINEAR_CSV = REPO / "results/nonlinear_stage_vgg11_fullfit/report/results.csv"
STAGES = [0, 1, 2, 3, 5, 10, 20, 30, 60, 100, 150, 200]


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def barrier_schematic() -> None:
    t = np.linspace(0.0, 1.0, 501)
    left, right = 0.28, 0.67
    chord = (1.0 - t) * left + t * right
    bump = 0.76 * 4.0 * t * (1.0 - t) * np.exp(-((t - 0.58) / 0.27) ** 2)
    path = chord + bump
    chord_gap = path - chord
    i_chord = int(np.argmax(chord_gap))
    i_path = int(np.argmax(path))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
    for ax in axes:
        ax.plot(t, path, color="#d95f02", linewidth=2.5, label=r"path cost $C(\theta(\lambda))$")
        ax.scatter([0, 1], [left, right], color="black", zorder=4, label="endpoint costs")
        ax.set(xlabel=r"interpolation position $\lambda$", ylabel="cost")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.1, 1.28)
        ax.grid(alpha=0.18)

    ax = axes[0]
    ax.plot(t, chord, color="#1b9e77", linestyle="--", linewidth=2,
            label="endpoint chord")
    ax.fill_between(t, chord, path, where=path >= chord, color="#1b9e77", alpha=0.12)
    ax.annotate("", xy=(t[i_chord], path[i_chord]), xytext=(t[i_chord], chord[i_chord]),
                arrowprops=dict(arrowstyle="<->", color="#1b9e77", linewidth=2))
    ax.text(t[i_chord] + 0.035, (path[i_chord] + chord[i_chord]) / 2,
            r"$B_{\mathrm{chord}}$", color="#13795b", va="center")
    ax.set_title("Deviation from the endpoint chord")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    worse = max(left, right)
    ax.axhline(worse, color="#7570b3", linestyle="--", linewidth=2,
               label="worse endpoint")
    ax.fill_between(t, worse, path, where=path >= worse, color="#7570b3", alpha=0.12)
    ax.annotate("", xy=(t[i_path], path[i_path]), xytext=(t[i_path], worse),
                arrowprops=dict(arrowstyle="<->", color="#7570b3", linewidth=2))
    ax.text(t[i_path] + 0.035, (path[i_path] + worse) / 2,
            r"$B_{\mathrm{worse}}$", color="#5c55a0", va="center")
    ax.set_title("Excess over the worse endpoint")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "barrier_definitions")


def load_linear() -> list[dict]:
    rows = []
    with LINEAR_CSV.open() as stream:
        for row in csv.DictReader(stream):
            for key in ("replicate", "left_epoch", "right_epoch", "points"):
                row[key] = int(row[key])
            row["value"] = float(row["value"])
            for key in ("selected", "selected_overall", "selected_permutation_only"):
                row[key] = row[key] == "True"
            rows.append(row)
    return rows


def linear_lookup(rows: list[dict], selection: str, subset: str, metric: str, barrier: str):
    field = f"selected_{selection}"
    return {
        (r["replicate"], r["left_epoch"], r["right_epoch"]): r["value"]
        for r in rows
        if r[field] and r["resolution"] == "selected_dense"
        and r["subset"] == subset and r["metric"] == metric and r["barrier"] == barrier
    }


def _series(lookup, pairs):
    values = np.asarray([[lookup[(rep, a, b)] for a, b in pairs] for rep in range(3)])
    return values.mean(0), values.std(0, ddof=1)


def _stage_axis(ax):
    ax.set_xscale("symlog", linthresh=2.0, linscale=1.0)
    ax.set_xticks(STAGES, [str(x) for x in STAGES], rotation=45)
    ax.set_xlabel("completed epoch")
    ax.grid(alpha=0.2)


def linear_summary() -> None:
    rows = load_linear()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), constrained_layout=True)
    colors = {"permutation_only": "#1f77b4", "overall": "#d62728"}
    labels = {"permutation_only": "best permutation only", "overall": "best overall"}

    specifications = [
        (axes[0], "worse", r"Excess over the worse endpoint"),
        (axes[1], "chord", r"Deviation from the endpoint chord"),
    ]
    pairs = [(e, e) for e in STAGES]
    for ax, barrier, title in specifications:
        for selection in ("permutation_only", "overall"):
            lookup = linear_lookup(rows, selection, "test_full", "loss", barrier)
            mean, std = _series(lookup, pairs)
            ax.plot(STAGES, mean, marker="o", color=colors[selection], label=labels[selection])
            ax.fill_between(STAGES, np.maximum(0.0, mean - std), mean + std,
                            color=colors[selection], alpha=0.15)
        ax.set_title(title)
        ax.set_ylabel(rf"test-loss $B_{{\mathrm{{{barrier}}}}}$")
        _stage_axis(ax)
    axes[0].legend(fontsize=8)
    _save(fig, "linear_stage_summary")


def linear_cross_stage() -> None:
    rows = load_linear()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), constrained_layout=True)
    for ax, barrier, title in (
        (axes[0], "worse", "Excess over the worse endpoint"),
        (axes[1], "chord", "Deviation from the endpoint chord"),
    ):
        lookup = linear_lookup(rows, "overall", "test_full", "loss", barrier)
        for label, pairs_, style, color in (
            (r"$A_{200}$--$B_n$", [(200, e) for e in STAGES], "-", "#2ca02c"),
            (r"$A_n$--$B_{200}$", [(e, 200) for e in STAGES], "--", "#9467bd"),
        ):
            mean, std = _series(lookup, pairs_)
            ax.plot(STAGES, mean, marker="o", linestyle=style, color=color, label=label)
            ax.fill_between(STAGES, np.maximum(0.0, mean - std), mean + std,
                            color=color, alpha=0.14)
        ax.set_title(title)
        ax.set_ylabel(rf"test-loss $B_{{\mathrm{{{barrier}}}}}$")
        _stage_axis(ax)
    axes[0].legend(fontsize=8)
    _save(fig, "linear_cross_stage")


def load_nonlinear() -> list[dict]:
    rows = []
    with NONLINEAR_CSV.open() as stream:
        for row in csv.DictReader(stream):
            for key in ("replicate", "n", "left_epoch", "right_epoch", "restart"):
                row[key] = int(row[key])
            for key in ("loss_chord", "loss_worse", "error_chord", "error_worse"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _nonlinear_series(rows, kind, method, subset="test_eval", metric="loss_chord"):
    stages = STAGES if kind == "same" else STAGES[:-1]
    grouped = defaultdict(list)
    for row in rows:
        if row["kind"] == kind and row["method"] == method and row["subset"] == subset:
            grouped[row["n"]].append(row[metric])
    matrix = np.asarray([grouped[e] for e in stages])
    return stages, matrix.mean(1), matrix.std(1, ddof=1)


def nonlinear_summary() -> None:
    rows = load_nonlinear()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45), constrained_layout=True)

    ax = axes[0]
    for method, color in (("linear", "#ff7f0e"), ("curve", "#1f77b4")):
        x, mean, std = _nonlinear_series(rows, "same", method)
        ax.plot(x, mean, marker="o", color=color, label=method)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title(r"Independent models: $A_n$--$B_n$")
    ax.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
    _stage_axis(ax)
    ax.legend(fontsize=8)

    ax = axes[1]
    for kind, direction, linestyle in (
        ("final_left", r"$A_{200}$--$B_n$", "-"),
        ("final_right", r"$A_n$--$B_{200}$", "--"),
    ):
        for method, color in (("linear", "#ff7f0e"), ("curve", "#1f77b4")):
            x, mean, std = _nonlinear_series(rows, kind, method)
            ax.plot(x, mean, marker="o", color=color, linestyle=linestyle,
                    label=f"{direction}, {method}")
            ax.fill_between(x, np.maximum(0.0, mean - std), mean + std,
                            color=color, alpha=0.08)
    ax.set_title("Independent models: cross stage")
    ax.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
    _stage_axis(ax)
    ax.legend(fontsize=7)

    ax = axes[2]
    for method, color in (("linear", "#ff7f0e"), ("curve", "#1f77b4")):
        values = []
        for kind in ("within_a", "within_b"):
            x, mean, _ = _nonlinear_series(rows, kind, method)
            values.append(mean)
        mean = np.mean(values, axis=0)
        ax.plot(STAGES[:-1], mean, marker="o", color=color, label=method)
    ax.set_title(r"Within run: average of $A_n$--$A_{200}$ and $B_n$--$B_{200}$")
    ax.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
    _stage_axis(ax)
    ax.legend(fontsize=8)

    _save(fig, "nonlinear_stage_summary")


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.frameon": True,
        "figure.dpi": 140,
    })
    barrier_schematic()
    linear_summary()
    linear_cross_stage()
    nonlinear_summary()


if __name__ == "__main__":
    main()
