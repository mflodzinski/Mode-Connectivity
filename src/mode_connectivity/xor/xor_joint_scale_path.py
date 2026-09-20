"""Joint positive-scaling and quadratic-Bezier optimization for 2-2-1 XOR."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_connectivity.common.paths import PROJECT_ROOT
from mode_connectivity.xor.xor_curve_fitting import (
    XOR_DATA,
    apply_permutation_to_state,
    filter_eligible_seeds,
    load_or_train_models,
    logits_from_param_vector,
    parse_pairs,
    parse_seed_list,
    state_to_vector,
    xor_loss_and_accuracy_from_logits,
)


PERMUTATIONS = {"identity": (0, 1), "swap": (1, 0)}


def scale_state(state, log_scales):
    """Apply the exact positive ReLU scaling symmetry to a 2-H-1 state."""
    if log_scales.ndim != 1:
        raise ValueError("log_scales must be a one-dimensional tensor")
    hidden_size = state["fc1.weight"].shape[0]
    if len(log_scales) != hidden_size:
        raise ValueError(
            f"Expected {hidden_size} log scales, received {len(log_scales)}"
        )
    scales = torch.exp(log_scales)
    inverse_scales = torch.exp(-log_scales)
    return OrderedDict(
        [
            ("fc1.weight", state["fc1.weight"] * scales[:, None]),
            ("fc1.bias", state["fc1.bias"] * scales),
            ("fc2.weight", state["fc2.weight"] * inverse_scales[None, :]),
            ("fc2.bias", state["fc2.bias"].clone()),
        ]
    )


def function_check_inputs():
    """Return XOR points plus a fixed grid that probes the whole input plane."""
    axis = torch.linspace(-2.0, 2.0, 17)
    grid = torch.cartesian_prod(axis, axis)
    return torch.cat([XOR_DATA, grid], dim=0)


def state_logits(state, inputs=None):
    if inputs is None:
        inputs = XOR_DATA
    hidden_size = state["fc1.weight"].shape[0]
    output_size = state["fc2.weight"].shape[0]
    return logits_from_param_vector(
        inputs,
        state_to_vector(state),
        hidden_size,
        output_size,
    )


def preservation_error(original_state, transformed_state):
    with torch.no_grad():
        inputs = function_check_inputs()
        original = state_logits(original_state, inputs)
        transformed = state_logits(transformed_state, inputs)
        difference = torch.abs(original - transformed)
    return {
        "max_abs_logit_difference": float(difference.max().item()),
        "mean_abs_logit_difference": float(difference.mean().item()),
    }


def assert_function_preserved(original_state, transformed_state):
    inputs = function_check_inputs()
    original = state_logits(original_state, inputs)
    transformed = state_logits(transformed_state, inputs)
    torch.testing.assert_close(original, transformed, atol=1e-5, rtol=1e-4)


def quadratic_bezier(t, endpoint_a, midpoint, endpoint_b):
    """Standard quadratic Bezier curve with fixed endpoints."""
    return (
        ((1.0 - t) ** 2) * endpoint_a
        + 2.0 * t * (1.0 - t) * midpoint
        + (t**2) * endpoint_b
    )


def midpoint_initialization(endpoint_a, endpoint_b, restart, restart_std, base_seed):
    midpoint = 0.5 * (endpoint_a + endpoint_b)
    if restart == 0 or restart_std == 0.0:
        return midpoint.clone(), None
    generator = torch.Generator(device="cpu")
    restart_seed = int(base_seed + restart)
    generator.manual_seed(restart_seed)
    endpoint_rms_distance = torch.sqrt(torch.mean((endpoint_b - endpoint_a) ** 2))
    noise_scale = restart_std * max(float(endpoint_rms_distance.item()), 1e-6)
    noise = torch.randn(midpoint.shape, generator=generator, dtype=midpoint.dtype)
    return midpoint + noise_scale * noise, restart_seed


def evaluate_path(endpoint_a, endpoint_b, midpoint, hidden_size, eval_points):
    ts = torch.linspace(0.0, 1.0, eval_points)
    losses, accuracies = [], []
    with torch.no_grad():
        for t in ts:
            parameters = quadratic_bezier(t, endpoint_a, midpoint, endpoint_b)
            logits = logits_from_param_vector(
                XOR_DATA, parameters, hidden_size, output_size=1
            )
            loss, accuracy = xor_loss_and_accuracy_from_logits(logits, output_size=1)
            losses.append(float(loss.item()))
            accuracies.append(float(accuracy))
    max_loss = max(losses)
    endpoint_average = 0.5 * (losses[0] + losses[-1])
    min_accuracy = min(accuracies)
    return {
        "t": ts.tolist(),
        "loss": losses,
        "accuracy": accuracies,
        "max_loss": float(max_loss),
        "endpoint_average_loss": float(endpoint_average),
        "loss_barrier": float(max_loss - endpoint_average),
        "min_accuracy": float(min_accuracy),
        "accuracy_barrier": float(100.0 - min_accuracy),
    }


def fit_path(
    state_a,
    state_b,
    *,
    method,
    initial_midpoint,
    steps,
    lr,
    num_t_samples,
    eval_points,
    scale_penalty,
    max_abs_log_scale,
    alternating_block_size,
    verbose=False,
):
    """Fit path-only, joint, or alternating scale/path parameters."""
    if method not in {"path_only", "joint", "alternating"}:
        raise ValueError(f"Unsupported method: {method}")
    hidden_size = state_a["fc1.weight"].shape[0]
    endpoint_a = state_to_vector(state_a).detach()
    midpoint = nn.Parameter(initial_midpoint.detach().clone())
    log_scales = nn.Parameter(torch.zeros(hidden_size, dtype=endpoint_a.dtype))
    t_samples = torch.linspace(0.0, 1.0, num_t_samples)

    if method == "path_only":
        optimizer = torch.optim.Adam([midpoint], lr=lr)
        path_optimizer = scale_optimizer = None
    elif method == "joint":
        optimizer = torch.optim.Adam([midpoint, log_scales], lr=lr)
        path_optimizer = scale_optimizer = None
    else:
        optimizer = None
        path_optimizer = torch.optim.Adam([midpoint], lr=lr)
        scale_optimizer = torch.optim.Adam([log_scales], lr=lr)

    best_objective = float("inf")
    best_midpoint = midpoint.detach().clone()
    best_log_scales = log_scales.detach().clone()

    for step in range(steps):
        if method == "alternating":
            path_optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

        if method == "path_only":
            scaled_state_b = state_b
        else:
            scaled_state_b = scale_state(state_b, log_scales)
        endpoint_b = state_to_vector(scaled_state_b)

        sampled_losses = []
        for t in t_samples:
            parameters = quadratic_bezier(t, endpoint_a, midpoint, endpoint_b)
            logits = logits_from_param_vector(
                XOR_DATA, parameters, hidden_size, output_size=1
            )
            loss, _ = xor_loss_and_accuracy_from_logits(logits, output_size=1)
            sampled_losses.append(loss)
        maximum_loss = torch.stack(sampled_losses).max()
        penalty = (
            scale_penalty * torch.sum(log_scales**2)
            if method != "path_only"
            else torch.zeros((), dtype=maximum_loss.dtype)
        )
        objective = maximum_loss + penalty
        objective.backward()

        # The objective was computed before the optimizer update, so retain the
        # parameters from that same state when tracking the best iterate.
        current_objective = float(objective.item())
        if current_objective < best_objective:
            best_objective = current_objective
            best_midpoint = midpoint.detach().clone()
            best_log_scales = log_scales.detach().clone()

        if method == "alternating":
            block = step // alternating_block_size
            if block % 2 == 0:
                path_optimizer.step()
            else:
                scale_optimizer.step()
        else:
            optimizer.step()

        if method != "path_only":
            with torch.no_grad():
                log_scales.clamp_(-max_abs_log_scale, max_abs_log_scale)

        if verbose and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
            print(
                f"      {method} step {step + 1}/{steps}: "
                f"max_loss={float(maximum_loss.item()):.6f}, "
                f"objective={current_objective:.6f}"
            )

    final_state_b = (
        state_b
        if method == "path_only"
        else scale_state(state_b, best_log_scales)
    )
    if method != "path_only":
        assert_function_preserved(state_b, final_state_b)
    metrics = evaluate_path(
        endpoint_a,
        state_to_vector(final_state_b).detach(),
        best_midpoint,
        hidden_size,
        eval_points,
    )
    metrics.update(
        {
            "best_training_objective": float(best_objective),
            "midpoint": best_midpoint.tolist(),
            "log_scales": best_log_scales.tolist(),
            "scales": torch.exp(best_log_scales).tolist(),
            "inverse_scales": torch.exp(-best_log_scales).tolist(),
            "function_preservation": preservation_error(state_b, final_state_b),
        }
    )
    return metrics


def mark_success(metrics, threshold):
    metrics["accuracy_success"] = metrics["min_accuracy"] >= 99.999
    metrics["low_loss_success"] = bool(
        metrics["accuracy_success"] and metrics["loss_barrier"] <= threshold
    )
    return metrics


def run_trial(
    state_a,
    state_b,
    *,
    methods,
    restarts,
    restart_std,
    base_seed,
    success_loss_barrier,
    fit_kwargs,
):
    endpoint_a = state_to_vector(state_a)
    endpoint_b = state_to_vector(state_b)
    results = {method: [] for method in methods}
    for restart in range(restarts):
        initial_midpoint, restart_seed = midpoint_initialization(
            endpoint_a, endpoint_b, restart, restart_std, base_seed
        )
        for method in methods:
            metrics = fit_path(
                state_a,
                state_b,
                method=method,
                initial_midpoint=initial_midpoint,
                **fit_kwargs,
            )
            mark_success(metrics, success_loss_barrier)
            metrics["restart"] = int(restart)
            metrics["restart_seed"] = restart_seed
            results[method].append(metrics)

    summarized = {}
    for method, runs in results.items():
        best = min(
            range(len(runs)),
            key=lambda index: (
                runs[index]["max_loss"],
                runs[index]["loss_barrier"],
                -runs[index]["min_accuracy"],
            ),
        )
        summarized[method] = {
            "restarts": runs,
            "best_restart": int(best),
            "best": runs[best],
            "accuracy_success_rate": float(
                np.mean([run["accuracy_success"] for run in runs])
            ),
            "low_loss_success_rate": float(
                np.mean([run["low_loss_success"] for run in runs])
            ),
        }
    return summarized


def plot_trial(trial, title, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"path_only": "tab:blue", "joint": "tab:purple", "alternating": "tab:green"}
    labels = {"path_only": "path only", "joint": "joint path + scale", "alternating": "alternating"}
    for method, result in trial["methods"].items():
        best = result["best"]
        axes[0].plot(best["t"], best["loss"], label=labels[method], color=colors[method])
        axes[1].plot(best["t"], best["accuracy"], label=labels[method], color=colors[method])
    axes[0].set(xlabel="t", ylabel="Binary cross-entropy", title="Best-restart loss")
    axes[1].set(xlabel="t", ylabel="Accuracy (%)", title="Best-restart accuracy", ylim=(0, 102))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_restart_csv(pair_results, positive_control, output):
    fields = [
        "kind", "seed_a", "seed_b", "permutation", "method", "restart",
        "max_loss", "loss_barrier", "min_accuracy", "accuracy_barrier",
        "accuracy_success", "low_loss_success", "log_scale_0", "log_scale_1",
        "scale_0", "scale_1", "max_abs_logit_difference",
    ]
    rows = []

    def append_trial(kind, seed_a, seed_b, permutation, trial):
        for method, result in trial["methods"].items():
            for run in result["restarts"]:
                rows.append(
                    {
                        "kind": kind,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "permutation": permutation,
                        "method": method,
                        "restart": run["restart"],
                        "max_loss": run["max_loss"],
                        "loss_barrier": run["loss_barrier"],
                        "min_accuracy": run["min_accuracy"],
                        "accuracy_barrier": run["accuracy_barrier"],
                        "accuracy_success": run["accuracy_success"],
                        "low_loss_success": run["low_loss_success"],
                        "log_scale_0": run["log_scales"][0],
                        "log_scale_1": run["log_scales"][1],
                        "scale_0": run["scales"][0],
                        "scale_1": run["scales"][1],
                        "max_abs_logit_difference": run["function_preservation"][
                            "max_abs_logit_difference"
                        ],
                    }
                )

    for pair in pair_results:
        for trial in pair["permutations"]:
            append_trial("pair", pair["seed_a"], pair["seed_b"], trial["permutation"], trial)
    if positive_control is not None:
        append_trial(
            "positive_control_unscaled",
            positive_control["seed_a"],
            positive_control["seed_b"],
            positive_control["permutation"],
            positive_control["unscaled_baseline"],
        )
        append_trial(
            "positive_control_scaled",
            positive_control["seed_a"],
            positive_control["seed_b"],
            positive_control["permutation"],
            positive_control,
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(pair_results, methods):
    summary = {}
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    for method in methods:
        all_runs = [run for trial in trials for run in trial["methods"][method]["restarts"]]
        best_runs = [trial["methods"][method]["best"] for trial in trials]
        summary[method] = {
            "num_pair_permutation_trials": len(trials),
            "num_restarts": len(all_runs),
            "restart_accuracy_success_rate": float(
                np.mean([run["accuracy_success"] for run in all_runs])
            ),
            "restart_low_loss_success_rate": float(
                np.mean([run["low_loss_success"] for run in all_runs])
            ),
            "trial_best_accuracy_success_rate": float(
                np.mean([run["accuracy_success"] for run in best_runs])
            ),
            "trial_best_low_loss_success_rate": float(
                np.mean([run["low_loss_success"] for run in best_runs])
            ),
            "mean_best_loss_barrier": float(np.mean([run["loss_barrier"] for run in best_runs])),
            "median_best_loss_barrier": float(
                np.median([run["loss_barrier"] for run in best_runs])
            ),
            "mean_best_max_loss": float(np.mean([run["max_loss"] for run in best_runs])),
        }
    return summary


def assess_positive_control(positive_control, success_loss_barrier):
    """State whether the control supports interpreting a negative pair result."""
    unscaled = positive_control["unscaled_baseline"]["methods"]["path_only"][
        "best"
    ]
    path_only = positive_control["methods"]["path_only"]["best"]
    joint = positive_control["methods"]["joint"]["best"]
    baseline_connectable = bool(unscaled["low_loss_success"])
    path_only_hardened = bool(
        baseline_connectable and not path_only["low_loss_success"]
    )
    joint_recovered = bool(joint["low_loss_success"])
    if baseline_connectable and path_only_hardened and joint_recovered:
        status = "passed"
    elif not baseline_connectable:
        status = "failed_unscaled_baseline"
    elif joint_recovered:
        status = "joint_recovered_but_scaling_did_not_harden_path_only"
    else:
        status = "failed_joint_recovery"
    return {
        "status": status,
        "negative_pair_results_interpretable": bool(
            baseline_connectable and path_only_hardened and joint_recovered
        ),
        "criterion": (
            "the unscaled known-connectable pair succeeds, extreme scaling "
            "makes path-only fail, and joint optimization recovers it"
        ),
        "success_loss_barrier": float(success_loss_barrier),
        "unscaled_baseline_connectable": baseline_connectable,
        "path_only_hardened": bool(path_only_hardened),
        "joint_recovered": bool(joint_recovered),
        "unscaled_best_loss_barrier": float(unscaled["loss_barrier"]),
        "unscaled_best_min_accuracy": float(unscaled["min_accuracy"]),
        "path_only_best_loss_barrier": float(path_only["loss_barrier"]),
        "joint_best_loss_barrier": float(joint["loss_barrier"]),
        "path_only_best_min_accuracy": float(path_only["min_accuracy"]),
        "joint_best_min_accuracy": float(joint["min_accuracy"]),
    }


def plot_summary(pair_results, methods, output):
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    labels = {"path_only": "path only", "joint": "joint path + scale", "alternating": "alternating"}
    values = [
        [trial["methods"][method]["best"]["loss_barrier"] for trial in trials]
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(methods))
    ax.boxplot(values, positions=positions, widths=0.5, showmeans=True)
    for index, method_values in enumerate(values):
        jitter = (
            np.linspace(-0.08, 0.08, len(method_values))
            if len(method_values) > 1
            else np.array([0.0])
        )
        ax.scatter(index + jitter, method_values, s=22, alpha=0.65)
    ax.set(
        xticks=positions,
        xticklabels=[labels[method] for method in methods],
        ylabel="Best-restart loss barrier",
        title="Joint scaling and nonlinear XOR paths",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--output", default="results/xor/xor_2h_joint_scale_path")
    parser.add_argument("--seeds", default="2,4,5,9,10,11,12,14")
    parser.add_argument("--pairs", default="4-10,4-12,4-5,9-11,2-11")
    parser.add_argument("--max-endpoint-loss", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-t-samples", type=int, default=11)
    parser.add_argument("--eval-points", type=int, default=121)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--restart-std", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--scale-penalty", type=float, default=1e-4)
    parser.add_argument("--max-abs-log-scale", type=float, default=8.0)
    parser.add_argument("--success-loss-barrier", type=float, default=0.05)
    parser.add_argument("--include-alternating", action="store_true")
    parser.add_argument("--alternating-block-size", type=int, default=50)
    parser.add_argument("--positive-control-log-scale", type=float, default=5.0)
    parser.add_argument("--positive-control-pair", default="2-9")
    parser.add_argument("--skip-positive-control", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0 or args.restarts <= 0:
        raise ValueError("steps and restarts must be positive")
    if args.num_t_samples < 3 or args.eval_points < 3:
        raise ValueError("num-t-samples and eval-points must be at least 3")
    if args.lr <= 0 or args.scale_penalty < 0 or args.restart_std < 0:
        raise ValueError("lr must be positive; penalties and restart std must be nonnegative")
    if args.alternating_block_size <= 0 or args.max_abs_log_scale <= 0:
        raise ValueError("alternating block size and max log scale must be positive")

    checkpoint_dir = Path(args.checkpoints_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(args.seeds, num_networks=0)
    models, model_info, run_checkpoint_dir = load_or_train_models(
        seeds,
        hidden_size=2,
        checkpoints_dir=str(checkpoint_dir),
        output_dir=str(output_dir),
        train_max_epochs=None,
        train_lr=None,
        verbose=args.verbose,
    )
    eligible = filter_eligible_seeds(model_info, args.max_endpoint_loss)
    pairs = parse_pairs(args.pairs, eligible)
    methods = ["path_only", "joint"]
    if args.include_alternating:
        methods.append("alternating")
    fit_kwargs = {
        "steps": args.steps,
        "lr": args.lr,
        "num_t_samples": args.num_t_samples,
        "eval_points": args.eval_points,
        "scale_penalty": args.scale_penalty,
        "max_abs_log_scale": args.max_abs_log_scale,
        "alternating_block_size": args.alternating_block_size,
        "verbose": args.verbose,
    }

    pair_results = []
    plots_dir = output_dir / "plots"
    for pair_index, (seed_a, seed_b) in enumerate(pairs):
        print(f"Pair {seed_a}-{seed_b}")
        pair_record = {"seed_a": seed_a, "seed_b": seed_b, "permutations": []}
        state_a = models[seed_a].state_dict()
        original_state_b = models[seed_b].state_dict()
        for permutation_index, (permutation_name, permutation) in enumerate(PERMUTATIONS.items()):
            state_b = apply_permutation_to_state(original_state_b, permutation)
            assert_function_preserved(original_state_b, state_b)
            trial = {
                "permutation": permutation_name,
                "permutation_indices": list(permutation),
                "permutation_preservation": preservation_error(original_state_b, state_b),
                "methods": run_trial(
                    state_a,
                    state_b,
                    methods=methods,
                    restarts=args.restarts,
                    restart_std=args.restart_std,
                    base_seed=args.base_seed + 1000 * pair_index + 100 * permutation_index,
                    success_loss_barrier=args.success_loss_barrier,
                    fit_kwargs=fit_kwargs,
                ),
            }
            pair_record["permutations"].append(trial)
            if not args.skip_plots:
                plot_trial(
                    trial,
                    f"XOR {seed_a}-{seed_b}, {permutation_name} permutation",
                    plots_dir / f"pair_{seed_a}_{seed_b}_{permutation_name}.png",
                )
        pair_results.append(pair_record)

    positive_control = None
    if not args.skip_positive_control:
        control_pairs = parse_pairs(args.positive_control_pair, eligible)
        if len(control_pairs) != 1:
            raise ValueError("positive-control-pair must specify exactly one pair")
        control_seed_a, control_seed_b = control_pairs[0]
        control_state_a = models[control_seed_a].state_dict()
        control_state_b = models[control_seed_b].state_dict()
        imposed_log_scales = torch.tensor(
            [args.positive_control_log_scale, -args.positive_control_log_scale],
            dtype=torch.float32,
        )
        extreme_state = scale_state(control_state_b, imposed_log_scales)
        assert_function_preserved(control_state_b, extreme_state)
        control_base_seed = args.base_seed + 999000
        unscaled_baseline = {
            "permutation": "identity",
            "methods": run_trial(
                control_state_a,
                control_state_b,
                methods=["path_only"],
                restarts=args.restarts,
                restart_std=args.restart_std,
                base_seed=control_base_seed,
                success_loss_barrier=args.success_loss_barrier,
                fit_kwargs=fit_kwargs,
            ),
        }
        positive_control = {
            "seed_a": control_seed_a,
            "seed_b": control_seed_b,
            "permutation": "identity",
            "unscaled_baseline": unscaled_baseline,
            "imposed_log_scales": imposed_log_scales.tolist(),
            "imposed_scales": torch.exp(imposed_log_scales).tolist(),
            "imposed_inverse_scales": torch.exp(-imposed_log_scales).tolist(),
            "imposed_function_preservation": preservation_error(
                control_state_b, extreme_state
            ),
            "methods": run_trial(
                control_state_a,
                extreme_state,
                methods=methods,
                restarts=args.restarts,
                restart_std=args.restart_std,
                base_seed=control_base_seed,
                success_loss_barrier=args.success_loss_barrier,
                fit_kwargs=fit_kwargs,
            ),
        }
        positive_control["assessment"] = assess_positive_control(
            positive_control, args.success_loss_barrier
        )
        for method_result in positive_control["methods"].values():
            for run in method_result["restarts"]:
                run["net_log_scales_from_original"] = [
                    imposed + learned
                    for imposed, learned in zip(
                        positive_control["imposed_log_scales"], run["log_scales"]
                    )
                ]
                run["net_scales_from_original"] = [
                    math.exp(value) for value in run["net_log_scales_from_original"]
                ]
        if not args.skip_plots:
            plot_trial(
                positive_control,
                (
                    f"Positive control: {control_seed_a}-{control_seed_b}, "
                    f"imposed log scales ±{args.positive_control_log_scale:g}"
                ),
                plots_dir / "positive_control.png",
            )
            plot_trial(
                unscaled_baseline,
                f"Positive-control baseline: {control_seed_a}-{control_seed_b}",
                plots_dir / "positive_control_unscaled.png",
            )

    summary = aggregate_summary(pair_results, methods)
    results = {
        "config": {
            "architecture": "2-2-1 ReLU MLP",
            "checkpoints_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "run_checkpoints_dir": run_checkpoint_dir,
            "seeds": seeds,
            "eligible_seeds": eligible,
            "pairs": [list(pair) for pair in pairs],
            "permutations": {name: list(value) for name, value in PERMUTATIONS.items()},
            "methods": methods,
            "optimizer": "Adam",
            "steps": args.steps,
            "lr": args.lr,
            "num_t_samples": args.num_t_samples,
            "eval_points": args.eval_points,
            "restarts": args.restarts,
            "restart_std": args.restart_std,
            "restart_zero_is_exact_midpoint": True,
            "scale_penalty": args.scale_penalty,
            "max_abs_log_scale": args.max_abs_log_scale,
            "success_loss_barrier": args.success_loss_barrier,
            "alternating_block_size": args.alternating_block_size,
            "positive_control_log_scale": args.positive_control_log_scale,
            "positive_control_pair": [
                int(value) for value in args.positive_control_pair.split("-")
            ],
            "bezier_formula": "(1-t)^2 A + 2t(1-t) M + t^2 B_scaled(u)",
        },
        "model_info": model_info,
        "pair_results": pair_results,
        "positive_control": positive_control,
        "negative_result_interpretability": (
            None
            if positive_control is None
            else positive_control["assessment"]
        ),
        "summary": summary,
    }
    results_path = output_dir / "joint_scale_path_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    write_restart_csv(
        pair_results,
        positive_control,
        output_dir / "restart_results.csv",
    )
    if not args.skip_plots:
        plot_summary(pair_results, methods, plots_dir / "summary_loss_barrier.png")

    print(json.dumps(summary, indent=2))
    if positive_control is not None:
        control_summary = {
            method: {
                "best_loss_barrier": record["best"]["loss_barrier"],
                "best_min_accuracy": record["best"]["min_accuracy"],
                "low_loss_success_rate": record["low_loss_success_rate"],
                "best_scales": record["best"]["scales"],
            }
            for method, record in positive_control["methods"].items()
        }
        print("Positive control:")
        print(json.dumps(control_summary, indent=2))
        print("Positive-control assessment:")
        print(json.dumps(positive_control["assessment"], indent=2))
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
