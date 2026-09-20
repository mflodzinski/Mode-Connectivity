"""Joint positive-scaling and polygonal-path optimization for 2-2-1 XOR."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from mode_connectivity.xor.xor_joint_scale_path import (
    PERMUTATIONS,
    assess_positive_control,
    assert_function_preserved,
    mark_success,
    plot_summary,
    plot_trial,
    preservation_error,
    scale_state,
)


def parse_bend_counts(value):
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count < 1 for count in counts):
        raise ValueError("polygon-bends must contain positive integers")
    return list(dict.fromkeys(counts))


def initialize_internal_points(
    endpoint_a,
    endpoint_b,
    num_internal_bends,
    restart,
    restart_std,
    base_seed,
):
    points = []
    for index in range(1, num_internal_bends + 1):
        alpha = index / (num_internal_bends + 1)
        points.append((1.0 - alpha) * endpoint_a + alpha * endpoint_b)
    if restart == 0 or restart_std == 0.0:
        return [point.clone() for point in points], None

    restart_seed = int(base_seed + restart)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(restart_seed)
    endpoint_rms_distance = torch.sqrt(torch.mean((endpoint_b - endpoint_a) ** 2))
    noise_scale = restart_std * max(float(endpoint_rms_distance.item()), 1e-6)
    perturbed = [
        point
        + noise_scale
        * torch.randn(point.shape, generator=generator, dtype=point.dtype)
        for point in points
    ]
    return perturbed, restart_seed


def polygonal_path(t, control_points):
    """Evaluate a uniformly parameterized piecewise-linear control polygon."""
    scaled_t = t * (len(control_points) - 1)
    result = torch.zeros_like(control_points[0])
    for index, point in enumerate(control_points):
        coefficient = torch.clamp(
            1.0 - torch.abs(scaled_t - float(index)), min=0.0, max=1.0
        )
        result = result + coefficient * point
    return result


def fitting_grid(num_t_samples, num_internal_bends):
    """Use the old grid and add every polygon knot so bends cannot be missed."""
    regular = torch.linspace(0.0, 1.0, num_t_samples).tolist()
    knots = [
        index / (num_internal_bends + 1)
        for index in range(1, num_internal_bends + 1)
    ]
    return torch.tensor(sorted(set(regular + knots)), dtype=torch.float32)


def evaluate_path(endpoint_a, endpoint_b, internal_points, hidden_size, eval_points):
    control_points = [endpoint_a, *internal_points, endpoint_b]
    ts = torch.linspace(0.0, 1.0, eval_points)
    losses, accuracies = [], []
    with torch.no_grad():
        for t in ts:
            parameters = polygonal_path(t, control_points)
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
    initial_internal_points,
    steps,
    lr,
    num_t_samples,
    eval_points,
    scale_penalty,
    max_abs_log_scale,
    alternating_block_size,
    verbose=False,
):
    if method not in {"path_only", "joint", "alternating"}:
        raise ValueError(f"Unsupported method: {method}")
    hidden_size = state_a["fc1.weight"].shape[0]
    endpoint_a = state_to_vector(state_a).detach()
    internal_points = nn.ParameterList(
        [nn.Parameter(point.detach().clone()) for point in initial_internal_points]
    )
    log_scales = nn.Parameter(torch.zeros(hidden_size, dtype=endpoint_a.dtype))
    t_samples = fitting_grid(num_t_samples, len(internal_points))

    if method == "path_only":
        optimizer = torch.optim.Adam(internal_points.parameters(), lr=lr)
        path_optimizer = scale_optimizer = None
    elif method == "joint":
        optimizer = torch.optim.Adam(
            [*internal_points.parameters(), log_scales], lr=lr
        )
        path_optimizer = scale_optimizer = None
    else:
        optimizer = None
        path_optimizer = torch.optim.Adam(internal_points.parameters(), lr=lr)
        scale_optimizer = torch.optim.Adam([log_scales], lr=lr)

    best_objective = float("inf")
    best_internal_points = [point.detach().clone() for point in internal_points]
    best_log_scales = log_scales.detach().clone()

    for step in range(steps):
        if method == "alternating":
            path_optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

        scaled_state_b = (
            state_b if method == "path_only" else scale_state(state_b, log_scales)
        )
        endpoint_b = state_to_vector(scaled_state_b)
        control_points = [endpoint_a, *internal_points, endpoint_b]
        sampled_losses = []
        for t in t_samples:
            parameters = polygonal_path(t, control_points)
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

        current_objective = float(objective.item())
        if current_objective < best_objective:
            best_objective = current_objective
            best_internal_points = [
                point.detach().clone() for point in internal_points
            ]
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
        best_internal_points,
        hidden_size,
        eval_points,
    )
    metrics.update(
        {
            "best_training_objective": float(best_objective),
            "num_internal_bends": len(best_internal_points),
            "num_control_points": len(best_internal_points) + 2,
            "fitting_t": t_samples.tolist(),
            "internal_points": [point.tolist() for point in best_internal_points],
            "log_scales": best_log_scales.tolist(),
            "scales": torch.exp(best_log_scales).tolist(),
            "inverse_scales": torch.exp(-best_log_scales).tolist(),
            "function_preservation": preservation_error(state_b, final_state_b),
        }
    )
    return metrics


def run_trial(
    state_a,
    state_b,
    *,
    methods,
    num_internal_bends,
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
        initial_points, restart_seed = initialize_internal_points(
            endpoint_a,
            endpoint_b,
            num_internal_bends,
            restart,
            restart_std,
            base_seed,
        )
        for method in methods:
            metrics = fit_path(
                state_a,
                state_b,
                method=method,
                initial_internal_points=initial_points,
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


def aggregate_summary(pair_results, methods):
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    summary = {}
    for method in methods:
        all_runs = [
            run
            for trial in trials
            for run in trial["methods"][method]["restarts"]
        ]
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
            "mean_best_loss_barrier": float(
                np.mean([run["loss_barrier"] for run in best_runs])
            ),
            "median_best_loss_barrier": float(
                np.median([run["loss_barrier"] for run in best_runs])
            ),
            "mean_best_max_loss": float(
                np.mean([run["max_loss"] for run in best_runs])
            ),
        }
    return summary


def run_for_bend_count(
    bend_count,
    pairs,
    eligible,
    models,
    methods,
    args,
    fit_kwargs,
    output_dir,
):
    pair_results = []
    plots_dir = output_dir / f"bends_{bend_count}" / "plots"
    for pair_index, (seed_a, seed_b) in enumerate(pairs):
        print(f"Bends {bend_count}, pair {seed_a}-{seed_b}")
        pair_record = {"seed_a": seed_a, "seed_b": seed_b, "permutations": []}
        state_a = models[seed_a].state_dict()
        original_state_b = models[seed_b].state_dict()
        for permutation_index, (permutation_name, permutation) in enumerate(
            PERMUTATIONS.items()
        ):
            state_b = apply_permutation_to_state(original_state_b, permutation)
            assert_function_preserved(original_state_b, state_b)
            trial = {
                "permutation": permutation_name,
                "permutation_indices": list(permutation),
                "permutation_preservation": preservation_error(
                    original_state_b, state_b
                ),
                "methods": run_trial(
                    state_a,
                    state_b,
                    methods=methods,
                    num_internal_bends=bend_count,
                    restarts=args.restarts,
                    restart_std=args.restart_std,
                    base_seed=(
                        args.base_seed
                        + 100000 * bend_count
                        + 1000 * pair_index
                        + 100 * permutation_index
                    ),
                    success_loss_barrier=args.success_loss_barrier,
                    fit_kwargs=fit_kwargs,
                ),
            }
            pair_record["permutations"].append(trial)
            if not args.skip_plots:
                plot_trial(
                    trial,
                    (
                        f"XOR {seed_a}-{seed_b}, {permutation_name}, "
                        f"polygon with {bend_count} internal bend(s)"
                    ),
                    plots_dir / f"pair_{seed_a}_{seed_b}_{permutation_name}.png",
                )
        pair_results.append(pair_record)

    positive_control = None
    if not args.skip_positive_control:
        control_seed_a, control_seed_b = parse_pairs(
            args.positive_control_pair,
            eligible,
        )[0]
        state_a = models[control_seed_a].state_dict()
        state_b = models[control_seed_b].state_dict()
        imposed_log_scales = torch.tensor(
            [args.positive_control_log_scale, -args.positive_control_log_scale],
            dtype=torch.float32,
        )
        extreme_state = scale_state(state_b, imposed_log_scales)
        assert_function_preserved(state_b, extreme_state)
        control_seed = args.base_seed + 100000 * bend_count + 999000
        unscaled_baseline = {
            "permutation": "identity",
            "methods": run_trial(
                state_a,
                state_b,
                methods=["path_only"],
                num_internal_bends=bend_count,
                restarts=args.restarts,
                restart_std=args.restart_std,
                base_seed=control_seed,
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
                state_b, extreme_state
            ),
            "methods": run_trial(
                state_a,
                extreme_state,
                methods=methods,
                num_internal_bends=bend_count,
                restarts=args.restarts,
                restart_std=args.restart_std,
                base_seed=control_seed,
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
                        imposed_log_scales.tolist(), run["log_scales"]
                    )
                ]
                run["net_scales_from_original"] = [
                    math.exp(value) for value in run["net_log_scales_from_original"]
                ]
        if not args.skip_plots:
            plot_trial(
                positive_control,
                f"Positive control, polygon with {bend_count} internal bend(s)",
                plots_dir / "positive_control.png",
            )
            plot_trial(
                unscaled_baseline,
                (
                    f"Positive-control baseline, polygon with "
                    f"{bend_count} internal bend(s)"
                ),
                plots_dir / "positive_control_unscaled.png",
            )

    summary = aggregate_summary(pair_results, methods)
    if not args.skip_plots:
        plot_summary(
            pair_results,
            methods,
            plots_dir / "summary_loss_barrier.png",
        )
    return {
        "num_internal_bends": bend_count,
        "num_control_points": bend_count + 2,
        "pair_results": pair_results,
        "positive_control": positive_control,
        "negative_result_interpretability": (
            None
            if positive_control is None
            else positive_control["assessment"]
        ),
        "summary": summary,
    }


def write_restart_csv(bend_results, output):
    fields = [
        "num_internal_bends",
        "kind",
        "seed_a",
        "seed_b",
        "permutation",
        "method",
        "restart",
        "max_loss",
        "loss_barrier",
        "min_accuracy",
        "accuracy_barrier",
        "accuracy_success",
        "low_loss_success",
        "log_scale_0",
        "log_scale_1",
        "scale_0",
        "scale_1",
        "max_abs_logit_difference",
    ]
    rows = []

    def append_trial(bends, kind, seed_a, seed_b, permutation, trial):
        for method, method_result in trial["methods"].items():
            for run in method_result["restarts"]:
                rows.append(
                    {
                        "num_internal_bends": bends,
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

    for bend_result in bend_results:
        bends = bend_result["num_internal_bends"]
        for pair in bend_result["pair_results"]:
            for trial in pair["permutations"]:
                append_trial(
                    bends,
                    "pair",
                    pair["seed_a"],
                    pair["seed_b"],
                    trial["permutation"],
                    trial,
                )
        control = bend_result["positive_control"]
        if control is not None:
            append_trial(
                bends,
                "positive_control_unscaled",
                control["seed_a"],
                control["seed_b"],
                control["permutation"],
                control["unscaled_baseline"],
            )
            append_trial(
                bends,
                "positive_control_scaled",
                control["seed_a"],
                control["seed_b"],
                control["permutation"],
                control,
            )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_combined_summary(bend_results, methods, output):
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods):
        for bend_result in bend_results:
            bends = bend_result["num_internal_bends"]
            trials = [
                trial
                for pair in bend_result["pair_results"]
                for trial in pair["permutations"]
            ]
            values = [
                trial["methods"][method]["best"]["loss_barrier"]
                for trial in trials
            ]
            axis.scatter([bends] * len(values), values, alpha=0.55, s=24)
            axis.plot(
                bends,
                np.mean(values),
                marker="D",
                color="black",
                markersize=6,
            )
        axis.set(
            xlabel="Number of internal bends",
            ylabel="Best-restart loss barrier",
            title=method.replace("_", " "),
            xticks=[result["num_internal_bends"] for result in bend_results],
        )
        axis.grid(alpha=0.25)
    fig.suptitle("Width-2 XOR polygonal paths")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument(
        "--output", default="results/xor/xor_2h_joint_scale_polygonal"
    )
    parser.add_argument("--seeds", default="2,4,5,9,10,11,12,14")
    parser.add_argument("--pairs", default="4-10,4-12,4-5,9-11,2-11")
    parser.add_argument("--polygon-bends", default="1,2,4,6")
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

    bend_counts = parse_bend_counts(args.polygon_bends)
    if args.steps <= 0 or args.restarts <= 0:
        raise ValueError("steps and restarts must be positive")
    if args.num_t_samples < 3 or args.eval_points < 3:
        raise ValueError("num-t-samples and eval-points must be at least 3")
    if args.lr <= 0 or args.scale_penalty < 0 or args.restart_std < 0:
        raise ValueError(
            "lr must be positive; penalties and restart std must be nonnegative"
        )
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

    bend_results = [
        run_for_bend_count(
            bend_count,
            pairs,
            eligible,
            models,
            methods,
            args,
            fit_kwargs,
            output_dir,
        )
        for bend_count in bend_counts
    ]
    results = {
        "config": {
            "architecture": "2-2-1 ReLU MLP",
            "path_type": "uniform piecewise-linear polygonal chain",
            "num_internal_bends": bend_counts,
            "checkpoints_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "run_checkpoints_dir": run_checkpoint_dir,
            "seeds": seeds,
            "eligible_seeds": eligible,
            "pairs": [list(pair) for pair in pairs],
            "permutations": {
                name: list(value) for name, value in PERMUTATIONS.items()
            },
            "methods": methods,
            "optimizer": "Adam",
            "steps": args.steps,
            "lr": args.lr,
            "base_num_t_samples": args.num_t_samples,
            "polygon_knots_always_in_fitting_grid": True,
            "eval_points": args.eval_points,
            "restarts": args.restarts,
            "restart_std": args.restart_std,
            "restart_zero_is_straight_polygon": True,
            "scale_penalty": args.scale_penalty,
            "max_abs_log_scale": args.max_abs_log_scale,
            "success_loss_barrier": args.success_loss_barrier,
            "alternating_block_size": args.alternating_block_size,
            "positive_control_log_scale": args.positive_control_log_scale,
            "positive_control_pair": [
                int(value) for value in args.positive_control_pair.split("-")
            ],
        },
        "model_info": model_info,
        "bend_results": bend_results,
    }
    results_path = output_dir / "joint_scale_polygonal_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    write_restart_csv(bend_results, output_dir / "restart_results.csv")
    if not args.skip_plots:
        plot_combined_summary(
            bend_results,
            methods,
            output_dir / "summary_by_bends.png",
        )

    compact_summary = {
        str(result["num_internal_bends"]): result["summary"]
        for result in bend_results
    }
    print(json.dumps(compact_summary, indent=2))
    for result in bend_results:
        control = result["positive_control"]
        if control is not None:
            print(
                f"Positive control ({result['num_internal_bends']} bends): "
                f"{json.dumps(control['assessment'])}"
            )
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
