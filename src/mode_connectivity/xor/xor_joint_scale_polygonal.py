"""Joint positive-scaling and nonlinear-path optimization for 2-H-1 XOR."""

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
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_connectivity.common.paths import PROJECT_ROOT
from mode_connectivity.xor.xor_curve_fitting import (
    XOR_DATA,
    XOR_LABELS,
    SimpleMLP,
    apply_permutation_to_state,
    evaluate_model,
    filter_eligible_seeds,
    parse_pairs,
    state_to_vector,
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


def checkpoint_seed(path):
    stem = path.stem
    return int(stem.removeprefix("seed_").removeprefix("seed"))


def load_saved_models(checkpoint_dir, requested_seeds, hidden_size):
    """Load either retained XOR checkpoint schema without retraining endpoints."""
    paths = sorted(
        set(checkpoint_dir.glob("seed_*.pt")) | set(checkpoint_dir.glob("seed*.pt")),
        key=checkpoint_seed,
    )
    if not paths:
        raise FileNotFoundError(f"No seed checkpoints found in {checkpoint_dir}")

    requested = None if requested_seeds is None else set(requested_seeds)
    models = {}
    model_info = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        seed = int(payload.get("seed", checkpoint_seed(path)))
        if requested is not None and seed not in requested:
            continue
        state = payload.get("state_dict", payload.get("model_state"))
        if state is None:
            raise ValueError(f"Checkpoint {path} has no state_dict or model_state")
        checkpoint_width = int(state["fc1.weight"].shape[0])
        if checkpoint_width != hidden_size:
            raise ValueError(
                f"Checkpoint {path} has hidden size {checkpoint_width}, "
                f"expected {hidden_size}"
            )
        model = SimpleMLP(hidden_size=hidden_size, output_size=1)
        model.load_state_dict(state)
        model.eval()
        metrics = evaluate_model(model)
        models[seed] = model
        model_info[seed] = {
            "accuracy": float(metrics["accuracy"]),
            "loss": float(metrics["loss"]),
            "source": "checkpoint",
            "loaded_checkpoint": str(path),
            "hidden_size": hidden_size,
            "output_size": 1,
        }

    if requested is not None:
        missing = sorted(requested - set(models))
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoints for requested seeds {missing} in {checkpoint_dir}"
            )
    if not models:
        raise RuntimeError(f"No requested checkpoints loaded from {checkpoint_dir}")
    return models, model_info


def load_fixed_permutations(results_path, method, hidden_size):
    """Read one previously selected permutation for every endpoint pair."""
    payload = json.loads(results_path.read_text())
    result = {}
    for pair in payload.get("pair_results", []):
        key = (int(pair["seed_a"]), int(pair["seed_b"]))
        record = pair.get(method)
        if record is None:
            raise KeyError(f"Alignment method {method!r} is absent for pair {key}")
        if isinstance(record, list):
            permutation = record
        else:
            permutation = record.get("best_perm", record.get("hard_perm"))
        if permutation is None:
            raise KeyError(
                f"Alignment method {method!r} has no best_perm/hard_perm for pair {key}"
            )
        permutation = tuple(int(index) for index in permutation)
        if sorted(permutation) != list(range(hidden_size)):
            raise ValueError(f"Invalid width-{hidden_size} permutation for pair {key}")
        result[key] = permutation
    return result


def parse_bend_counts(value):
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count < 1 for count in counts):
        raise ValueError("internal-points must contain positive integers")
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


def polygonal_path_batch(ts, control_points):
    """Evaluate a control polygon at all path locations in one tensor operation."""
    points = torch.stack(list(control_points), dim=0)
    knot_indices = torch.arange(
        len(control_points), dtype=ts.dtype, device=ts.device
    )
    coefficients = torch.clamp(
        1.0
        - torch.abs(ts[:, None] * (len(control_points) - 1) - knot_indices[None, :]),
        min=0.0,
        max=1.0,
    )
    return coefficients @ points


def bezier_path_batch(ts, control_points):
    """Evaluate a Bezier curve at all path locations in one tensor operation."""
    points = torch.stack(list(control_points), dim=0)
    degree = len(control_points) - 1
    indices = torch.arange(
        len(control_points), dtype=ts.dtype, device=ts.device
    )
    binomial = torch.tensor(
        [math.comb(degree, index) for index in range(degree + 1)],
        dtype=ts.dtype,
        device=ts.device,
    )
    coefficients = (
        binomial[None, :]
        * (1.0 - ts[:, None]) ** (degree - indices[None, :])
        * ts[:, None] ** indices[None, :]
    )
    return coefficients @ points


def nonlinear_path_batch(ts, control_points, curve_type):
    if curve_type == "polygonal":
        return polygonal_path_batch(ts, control_points)
    if curve_type == "bezier":
        return bezier_path_batch(ts, control_points)
    raise ValueError(f"Unsupported curve type: {curve_type}")


def logits_from_param_matrix(inputs, param_matrix, hidden_size):
    """Vectorized 2-H-1 forward pass for a matrix of flattened parameters."""
    index = 0
    first_weight_size = hidden_size * 2
    first_weight = param_matrix[:, index : index + first_weight_size].reshape(
        -1, hidden_size, 2
    )
    index += first_weight_size
    first_bias = param_matrix[:, index : index + hidden_size]
    index += hidden_size
    second_weight = param_matrix[:, index : index + hidden_size].reshape(
        -1, 1, hidden_size
    )
    index += hidden_size
    second_bias = param_matrix[:, index : index + 1]

    hidden = torch.relu(
        torch.einsum("nd,thd->tnh", inputs, first_weight)
        + first_bias[:, None, :]
    )
    return (
        torch.einsum("tnh,toh->tno", hidden, second_weight)
        + second_bias[:, None, :]
    )


def path_losses_and_accuracies(param_matrix, hidden_size):
    logits = logits_from_param_matrix(XOR_DATA, param_matrix, hidden_size)
    labels = XOR_LABELS[None, :, :].expand_as(logits)
    losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ).mean(dim=(1, 2))
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    accuracies = (predictions == labels).float().mean(dim=(1, 2)) * 100.0
    return losses, accuracies


def fitting_grid(num_t_samples, num_internal_bends):
    """Use the old grid and add every polygon knot so bends cannot be missed."""
    regular = torch.linspace(0.0, 1.0, num_t_samples).tolist()
    knots = [
        index / (num_internal_bends + 1)
        for index in range(1, num_internal_bends + 1)
    ]
    return torch.tensor(sorted(set(regular + knots)), dtype=torch.float32)


def stochastic_fitting_grid(
    num_samples,
    num_internal_bends,
    curve_type,
    previous_worst,
    generator,
):
    """Stratified random locations plus the last worst point and polygon knots."""
    strata = (
        torch.arange(num_samples, dtype=torch.float32)
        + torch.rand(num_samples, generator=generator)
    ) / num_samples
    locations = strata.tolist() + [float(previous_worst)]
    if curve_type == "polygonal":
        locations.extend(
            index / (num_internal_bends + 1)
            for index in range(1, num_internal_bends + 1)
        )
    return torch.tensor(sorted(set(locations)), dtype=torch.float32)


def evaluate_path(
    endpoint_a,
    endpoint_b,
    internal_points,
    hidden_size,
    eval_points,
    curve_type="polygonal",
):
    control_points = [endpoint_a, *internal_points, endpoint_b]
    ts = torch.linspace(0.0, 1.0, eval_points)
    with torch.no_grad():
        parameters = nonlinear_path_batch(ts, control_points, curve_type)
        loss_tensor, accuracy_tensor = path_losses_and_accuracies(
            parameters, hidden_size
        )
    losses = loss_tensor.tolist()
    accuracies = accuracy_tensor.tolist()
    max_loss = max(losses)
    endpoint_average = 0.5 * (losses[0] + losses[-1])
    chord = (
        (1.0 - ts) * loss_tensor[0]
        + ts * loss_tensor[-1]
    )
    excess_losses = loss_tensor - chord
    loss_barrier = float(torch.clamp(excess_losses.max(), min=0.0).item())
    min_accuracy = min(accuracies)
    return {
        "t": ts.tolist(),
        "loss": losses,
        "endpoint_loss_chord": chord.tolist(),
        "excess_loss": excess_losses.tolist(),
        "accuracy": accuracies,
        "max_loss": float(max_loss),
        "endpoint_average_loss": float(endpoint_average),
        "loss_barrier": loss_barrier,
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
    curve_type="polygonal",
    stochastic_max_samples=0,
    grid_refresh_every=25,
    grid_refresh_points=31,
    sampling_seed=0,
    scale_endpoints="second",
    internal_parameterization="affine_residual",
    objective_name="max_excess",
    verbose=False,
):
    if method not in {"path_only", "joint", "alternating"}:
        raise ValueError(f"Unsupported method: {method}")
    hidden_size = state_a["fc1.weight"].shape[0]
    if scale_endpoints not in {"second", "both"}:
        raise ValueError(f"Unsupported scale endpoint mode: {scale_endpoints}")
    if internal_parameterization not in {"affine_residual", "absolute"}:
        raise ValueError(
            f"Unsupported internal parameterization: {internal_parameterization}"
        )
    if objective_name not in {"max_excess", "mean_loss"}:
        raise ValueError(f"Unsupported optimization objective: {objective_name}")
    endpoint_a_unscaled = state_to_vector(state_a).detach()
    endpoint_b_unscaled = state_to_vector(state_b).detach()
    internal_alphas = [
        index / (len(initial_internal_points) + 1)
        for index in range(1, len(initial_internal_points) + 1)
    ]
    initial_offsets = [
        point.detach()
        - ((1.0 - alpha) * endpoint_a_unscaled + alpha * endpoint_b_unscaled)
        for point, alpha in zip(initial_internal_points, internal_alphas)
    ]
    initial_internal_parameters = (
        initial_offsets
        if internal_parameterization == "affine_residual"
        else [point.detach() for point in initial_internal_points]
    )
    internal_parameters = nn.ParameterList(
        [nn.Parameter(value.clone()) for value in initial_internal_parameters]
    )
    log_scales_a = nn.Parameter(
        torch.zeros(hidden_size, dtype=endpoint_a_unscaled.dtype)
    )
    log_scales_b = nn.Parameter(
        torch.zeros(hidden_size, dtype=endpoint_a_unscaled.dtype)
    )
    active_scale_parameters = (
        [log_scales_b]
        if scale_endpoints == "second"
        else [log_scales_a, log_scales_b]
    )
    with torch.no_grad():
        endpoint_parameters = torch.stack(
            [endpoint_a_unscaled, endpoint_b_unscaled]
        )
        endpoint_losses, _ = path_losses_and_accuracies(
            endpoint_parameters, hidden_size
        )
    fixed_t_samples = fitting_grid(num_t_samples, len(internal_parameters))
    refresh_t_samples = fitting_grid(grid_refresh_points, len(internal_parameters))
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(int(sampling_seed))
    previous_worst = 0.5

    if method == "path_only":
        optimizer = torch.optim.Adam(internal_parameters.parameters(), lr=lr)
        path_optimizer = scale_optimizer = None
    elif method == "joint":
        optimizer = torch.optim.Adam(
            [*internal_parameters.parameters(), *active_scale_parameters], lr=lr
        )
        path_optimizer = scale_optimizer = None
    else:
        optimizer = None
        path_optimizer = torch.optim.Adam(internal_parameters.parameters(), lr=lr)
        scale_optimizer = torch.optim.Adam(active_scale_parameters, lr=lr)

    best_objective = float("inf")
    best_internal_parameters = [
        value.detach().clone() for value in internal_parameters
    ]
    best_log_scales_a = log_scales_a.detach().clone()
    best_log_scales_b = log_scales_b.detach().clone()

    for step in range(steps):
        if method == "alternating":
            path_optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

        scaled_state_a = (
            state_a
            if method == "path_only" or scale_endpoints == "second"
            else scale_state(state_a, log_scales_a)
        )
        scaled_state_b = (
            state_b if method == "path_only" else scale_state(state_b, log_scales_b)
        )
        endpoint_a = state_to_vector(scaled_state_a)
        endpoint_b = state_to_vector(scaled_state_b)
        if internal_parameterization == "affine_residual":
            internal_points = [
                (1.0 - alpha) * endpoint_a + alpha * endpoint_b + residual
                for alpha, residual in zip(internal_alphas, internal_parameters)
            ]
        else:
            internal_points = list(internal_parameters)
        control_points = [endpoint_a, *internal_points, endpoint_b]
        refresh_step = bool(
            stochastic_max_samples > 0
            and (
                step % grid_refresh_every == 0
                or step == steps - 1
            )
        )
        if stochastic_max_samples > 0 and not refresh_step:
            t_samples = stochastic_fitting_grid(
                stochastic_max_samples,
                len(internal_parameters),
                curve_type,
                previous_worst,
                random_generator,
            )
        elif stochastic_max_samples > 0:
            t_samples = refresh_t_samples
        else:
            t_samples = fixed_t_samples
        endpoint_loss_chord = (
            (1.0 - t_samples) * endpoint_losses[0]
            + t_samples * endpoint_losses[1]
        )
        parameters = nonlinear_path_batch(t_samples, control_points, curve_type)
        sampled_losses, _ = path_losses_and_accuracies(parameters, hidden_size)
        excess_losses = sampled_losses - endpoint_loss_chord
        worst_index = int(torch.argmax(excess_losses).item())
        previous_worst = float(t_samples[worst_index].item())
        maximum_excess_loss = torch.clamp(excess_losses[worst_index], min=0.0)
        path_objective = (
            maximum_excess_loss
            if objective_name == "max_excess"
            else sampled_losses.mean()
        )
        penalty = (
            scale_penalty
            * sum(torch.sum(parameter**2) for parameter in active_scale_parameters)
            if method != "path_only"
            else torch.zeros((), dtype=path_objective.dtype)
        )
        objective = path_objective + penalty
        objective.backward()

        current_objective = float(objective.item())
        # Stochastic minibatches are not directly comparable across steps.
        # Select checkpoints only on the common refresh grid.
        comparable_objective = stochastic_max_samples == 0 or refresh_step
        if comparable_objective and current_objective < best_objective:
            best_objective = current_objective
            best_internal_parameters = [
                value.detach().clone() for value in internal_parameters
            ]
            best_log_scales_a = log_scales_a.detach().clone()
            best_log_scales_b = log_scales_b.detach().clone()

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
                for parameter in active_scale_parameters:
                    parameter.clamp_(-max_abs_log_scale, max_abs_log_scale)

        if verbose and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
            print(
                f"      {method} step {step + 1}/{steps}: "
                f"{objective_name}={float(path_objective.item()):.6f}, "
                f"objective={current_objective:.6f}"
            )

    final_state_a = (
        state_a
        if method == "path_only" or scale_endpoints == "second"
        else scale_state(state_a, best_log_scales_a)
    )
    final_state_b = (
        state_b if method == "path_only" else scale_state(state_b, best_log_scales_b)
    )
    if method != "path_only":
        assert_function_preserved(state_a, final_state_a)
        assert_function_preserved(state_b, final_state_b)
    final_endpoint_a = state_to_vector(final_state_a).detach()
    final_endpoint_b = state_to_vector(final_state_b).detach()
    if internal_parameterization == "affine_residual":
        best_internal_points = [
            (1.0 - alpha) * final_endpoint_a
            + alpha * final_endpoint_b
            + residual
            for alpha, residual in zip(
                internal_alphas, best_internal_parameters
            )
        ]
    else:
        best_internal_points = best_internal_parameters
    metrics = evaluate_path(
        final_endpoint_a,
        final_endpoint_b,
        best_internal_points,
        hidden_size,
        eval_points,
        curve_type=curve_type,
    )
    metrics.update(
        {
            "best_training_objective": float(best_objective),
            "num_internal_bends": len(best_internal_points),
            "num_control_points": len(best_internal_points) + 2,
            "curve_type": curve_type,
            "optimization_objective": objective_name,
            "fitting_mode": (
                "fixed_grid" if stochastic_max_samples == 0 else "stochastic_max"
            ),
            "fitting_t": (
                fixed_t_samples.tolist() if stochastic_max_samples == 0 else None
            ),
            "stochastic_max_samples": int(stochastic_max_samples),
            "grid_refresh_every": int(grid_refresh_every),
            "grid_refresh_points": int(grid_refresh_points),
            "sampling_seed": int(sampling_seed),
            "internal_points": [point.tolist() for point in best_internal_points],
            "internal_offsets": (
                [value.tolist() for value in best_internal_parameters]
                if internal_parameterization == "affine_residual"
                else None
            ),
            "internal_parameterization": internal_parameterization,
            "scale_endpoints": scale_endpoints,
            "log_scales": best_log_scales_b.tolist(),
            "scales": torch.exp(best_log_scales_b).tolist(),
            "inverse_scales": torch.exp(-best_log_scales_b).tolist(),
            "log_scales_a": best_log_scales_a.tolist(),
            "scales_a": torch.exp(best_log_scales_a).tolist(),
            "inverse_scales_a": torch.exp(-best_log_scales_a).tolist(),
            "log_scales_b": best_log_scales_b.tolist(),
            "scales_b": torch.exp(best_log_scales_b).tolist(),
            "inverse_scales_b": torch.exp(-best_log_scales_b).tolist(),
            "function_preservation": preservation_error(state_b, final_state_b),
            "function_preservation_a": preservation_error(state_a, final_state_a),
            "function_preservation_b": preservation_error(state_b, final_state_b),
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
    compact_restarts=False,
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
                sampling_seed=base_seed + restart,
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
                runs[index]["loss_barrier"],
                runs[index]["max_loss"],
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
        if compact_restarts:
            profile_fields = {
                "t",
                "loss",
                "endpoint_loss_chord",
                "excess_loss",
                "accuracy",
                "internal_points",
                "internal_offsets",
            }
            for index, run in enumerate(runs):
                if index != best:
                    for field in profile_fields:
                        run.pop(field, None)
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


def paired_scale_summary(pair_results, tolerance=1e-8):
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    path_barriers = np.asarray(
        [trial["methods"]["path_only"]["best"]["loss_barrier"] for trial in trials],
        dtype=np.float64,
    )
    joint_barriers = np.asarray(
        [trial["methods"]["joint"]["best"]["loss_barrier"] for trial in trials],
        dtype=np.float64,
    )
    deltas = joint_barriers - path_barriers
    max_abs_log_scales = np.asarray(
        [
            max(
                abs(value)
                for field in ("log_scales_a", "log_scales_b")
                for value in trial["methods"]["joint"]["best"].get(field, [])
            )
            for trial in trials
        ],
        dtype=np.float64,
    )
    return {
        "delta_definition": "joint loss barrier minus path-only loss barrier",
        "comparison_tolerance": float(tolerance),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "num_joint_better": int(np.sum(deltas < -tolerance)),
        "num_path_only_better": int(np.sum(deltas > tolerance)),
        "num_tied": int(np.sum(np.abs(deltas) <= tolerance)),
        "median_max_abs_log_scale": float(np.median(max_abs_log_scales)),
        "max_abs_log_scale": float(np.max(max_abs_log_scales)),
    }


def run_for_bend_count(
    bend_count,
    pairs,
    eligible,
    models,
    methods,
    args,
    fit_kwargs,
    output_dir,
    fixed_permutations,
):
    pair_results = []
    plots_dir = output_dir / f"bends_{bend_count}" / "plots"
    for pair_index, (seed_a, seed_b) in enumerate(pairs):
        print(f"Bends {bend_count}, pair {seed_a}-{seed_b}")
        pair_record = {"seed_a": seed_a, "seed_b": seed_b, "permutations": []}
        state_a = models[seed_a].state_dict()
        original_state_b = models[seed_b].state_dict()
        if args.no_permutation:
            permutation_items = [
                ("raw_endpoint", tuple(range(args.hidden_size)))
            ]
        elif fixed_permutations is None:
            permutation_items = list(PERMUTATIONS.items())
        else:
            pair_key = (seed_a, seed_b)
            if pair_key not in fixed_permutations:
                raise KeyError(f"No fixed permutation available for pair {pair_key}")
            permutation_items = [
                ("fixed_best_permutation", fixed_permutations[pair_key])
            ]
        for permutation_index, (permutation_name, permutation) in enumerate(
            permutation_items
        ):
            if args.no_permutation:
                state_b = original_state_b
                permutation_metadata = {
                    "permutation": None,
                    "permutation_indices": None,
                    "permutation_preservation": None,
                }
            else:
                state_b = apply_permutation_to_state(original_state_b, permutation)
                assert_function_preserved(original_state_b, state_b)
                permutation_metadata = {
                    "permutation": permutation_name,
                    "permutation_indices": list(permutation),
                    "permutation_preservation": preservation_error(
                        original_state_b, state_b
                    ),
                }
            trial = {
                **permutation_metadata,
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
                    compact_restarts=args.compact_restarts,
                ),
            }
            pair_record["permutations"].append(trial)
            if not args.skip_plots:
                plot_trial(
                    trial,
                    (
                        f"XOR {seed_a}-{seed_b}, {permutation_name}, "
                        f"{args.curve_type} with {bend_count} internal point(s)"
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
        imposed_log_scales = torch.zeros(args.hidden_size, dtype=torch.float32)
        imposed_log_scales[0] = args.positive_control_log_scale
        if args.hidden_size > 1:
            imposed_log_scales[1] = -args.positive_control_log_scale
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
                compact_restarts=args.compact_restarts,
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
                compact_restarts=args.compact_restarts,
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
                (
                    f"Positive control, {args.curve_type} with "
                    f"{bend_count} internal point(s)"
                ),
                plots_dir / "positive_control.png",
            )
            plot_trial(
                unscaled_baseline,
                (
                    f"Positive-control baseline, {args.curve_type} with "
                    f"{bend_count} internal point(s)"
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
        plot_aggregate_loss_profiles(
            pair_results,
            methods,
            plots_dir / "aggregate_loss_profiles.png",
            hidden_size=args.hidden_size,
            bend_count=bend_count,
        )
        plot_paired_loss_barriers(
            pair_results,
            plots_dir / "paired_loss_barriers.png",
            hidden_size=args.hidden_size,
            bend_count=bend_count,
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
        "paired_scale_comparison": paired_scale_summary(pair_results),
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
        "log_scales_json",
        "scales_json",
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
                        "log_scales_json": json.dumps(run["log_scales"]),
                        "scales_json": json.dumps(run["scales"]),
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


def plot_combined_summary(
    bend_results, methods, output, hidden_size, curve_type="polygonal"
):
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
            xlabel="Number of internal control points",
            ylabel="Best-restart loss barrier",
            title=method.replace("_", " "),
            xticks=[result["num_internal_bends"] for result in bend_results],
        )
        axis.grid(alpha=0.25)
    fig.suptitle(f"Width-{hidden_size} XOR {curve_type} paths")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_aggregate_loss_profiles(
    pair_results, methods, output, *, hidden_size, bend_count
):
    """Plot all pair profiles faintly and the cross-pair median with an IQR band."""
    labels = {
        "path_only": "Nonlinear path",
        "joint": "Nonlinear path + scale",
        "alternating": "Alternating path + scale",
    }
    colors = {
        "path_only": "#0072B2",
        "joint": "#CC79A7",
        "alternating": "#009E73",
    }
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in methods:
        curves = np.asarray(
            [trial["methods"][method]["best"]["loss"] for trial in trials],
            dtype=np.float64,
        )
        path_locations = np.asarray(
            trials[0]["methods"][method]["best"]["t"], dtype=np.float64
        )
        for curve in curves:
            ax.plot(path_locations, curve, color=colors[method], alpha=0.10, lw=0.8)
        lower, median, upper = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(
            path_locations, lower, upper, color=colors[method], alpha=0.18
        )
        ax.plot(
            path_locations,
            median,
            color=colors[method],
            lw=2.4,
            label=labels[method],
        )
    ax.set(
        xlabel=r"$\lambda$",
        ylabel="Binary cross-entropy",
        title=(
            f"Width {hidden_size}: nonlinear paths with {bend_count} internal bends"
        ),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_paired_loss_barriers(
    pair_results, output, *, hidden_size, bend_count
):
    trials = [trial for pair in pair_results for trial in pair["permutations"]]
    path_only = np.asarray(
        [trial["methods"]["path_only"]["best"]["loss_barrier"] for trial in trials]
    )
    joint = np.asarray(
        [trial["methods"]["joint"]["best"]["loss_barrier"] for trial in trials]
    )
    limit = max(float(path_only.max()), float(joint.max()), 1e-6) * 1.05
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(path_only, joint, color="#CC79A7", edgecolor="white", linewidth=0.5)
    ax.plot([0.0, limit], [0.0, limit], "--", color="#777777", lw=1.2)
    ax.set(
        xlim=(0.0, limit),
        ylim=(0.0, limit),
        xlabel="Nonlinear-path loss barrier",
        ylabel="Path + scale loss barrier",
        title=f"Width {hidden_size}, {bend_count} internal bends",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--hidden-size", type=int, default=2)
    parser.add_argument(
        "--alignment-results",
        type=Path,
        default=None,
        help=(
            "Existing XOR alignment JSON from which one fixed permutation is "
            "read for each pair. Required for hidden sizes above two."
        ),
    )
    parser.add_argument(
        "--alignment-method",
        default="best_permutation",
        help="Pair-result key containing best_perm or hard_perm.",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help=(
            "Fit paths between the raw trained endpoints without searching for "
            "or applying a hidden-unit permutation."
        ),
    )
    parser.add_argument(
        "--output", default="results/xor/xor_2h_joint_scale_polygonal"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument(
        "--curve-type", choices=("polygonal", "bezier"), default="polygonal"
    )
    parser.add_argument(
        "--scale-endpoints",
        choices=("second", "both"),
        default="second",
        help="Optimize scale variables for the second endpoint or both endpoints.",
    )
    parser.add_argument(
        "--internal-parameterization",
        choices=("affine_residual", "absolute"),
        default="affine_residual",
        help=(
            "Represent interior controls relative to the changing endpoints or "
            "as independent absolute parameters."
        ),
    )
    parser.add_argument(
        "--polygon-bends",
        "--internal-points",
        dest="polygon_bends",
        default="1,2,4,6",
        help="Comma-separated numbers of trainable interior control points.",
    )
    parser.add_argument("--max-endpoint-loss", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--objective",
        choices=("max_excess", "mean_loss"),
        default="max_excess",
        help=(
            "Optimize the maximum excess loss or the mean path loss. "
            "The latter reproduces the original XOR nonlinear-path protocol."
        ),
    )
    parser.add_argument("--num-t-samples", type=int, default=11)
    parser.add_argument(
        "--stochastic-max-samples",
        type=int,
        default=0,
        help=(
            "Stratified random path locations per ordinary optimization step; "
            "0 retains the fixed-grid objective."
        ),
    )
    parser.add_argument("--grid-refresh-every", type=int, default=25)
    parser.add_argument("--grid-refresh-points", type=int, default=31)
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
    parser.add_argument(
        "--compact-restarts",
        action="store_true",
        help="Keep dense path profiles only for the selected restart.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bend_counts = parse_bend_counts(args.polygon_bends)
    if args.hidden_size < 2:
        raise ValueError("hidden-size must be at least 2")
    if (
        not args.no_permutation
        and args.hidden_size > 2
        and args.alignment_results is None
    ):
        raise ValueError(
            "--alignment-results is required above width 2 so permutation search "
            "is fixed rather than factorially repeated"
        )
    if args.steps <= 0 or args.restarts <= 0:
        raise ValueError("steps and restarts must be positive")
    if args.num_t_samples < 3 or args.eval_points < 3:
        raise ValueError("num-t-samples and eval-points must be at least 3")
    if args.stochastic_max_samples < 0:
        raise ValueError("stochastic-max-samples must be nonnegative")
    if args.objective == "mean_loss" and args.stochastic_max_samples > 0:
        raise ValueError(
            "--stochastic-max-samples is only defined for --objective max_excess"
        )
    if args.grid_refresh_every <= 0 or args.grid_refresh_points < 3:
        raise ValueError(
            "grid-refresh-every must be positive and grid-refresh-points at least 3"
        )
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

    seeds = (
        None
        if args.seeds is None
        else [int(value) for value in args.seeds.split(",") if value.strip()]
    )
    models, model_info = load_saved_models(
        checkpoint_dir, seeds, args.hidden_size
    )
    seeds = sorted(models)
    run_checkpoint_dir = None
    eligible = filter_eligible_seeds(model_info, args.max_endpoint_loss)
    pairs = parse_pairs(args.pairs, eligible)
    fixed_permutations = None
    if args.no_permutation and args.alignment_results is not None:
        raise ValueError(
            "--no-permutation and --alignment-results are mutually exclusive"
        )
    if args.alignment_results is not None:
        alignment_results = args.alignment_results
        if not alignment_results.is_absolute():
            alignment_results = PROJECT_ROOT / alignment_results
        fixed_permutations = load_fixed_permutations(
            alignment_results, args.alignment_method, args.hidden_size
        )
        missing_pairs = sorted(set(pairs) - set(fixed_permutations))
        if missing_pairs:
            raise KeyError(
                f"Alignment results {alignment_results} lack pairs {missing_pairs}"
            )
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
        "curve_type": args.curve_type,
        "stochastic_max_samples": args.stochastic_max_samples,
        "grid_refresh_every": args.grid_refresh_every,
        "grid_refresh_points": args.grid_refresh_points,
        "scale_endpoints": args.scale_endpoints,
        "internal_parameterization": args.internal_parameterization,
        "objective_name": args.objective,
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
            fixed_permutations,
        )
        for bend_count in bend_counts
    ]
    results = {
        "config": {
            "architecture": f"2-{args.hidden_size}-1 ReLU MLP",
            "hidden_size": args.hidden_size,
            "path_type": (
                "uniform piecewise-linear polygonal chain"
                if args.curve_type == "polygonal"
                else "Bezier curve"
            ),
            "curve_type": args.curve_type,
            "internal_control_parameterization": (
                args.internal_parameterization
            ),
            "scale_endpoints": args.scale_endpoints,
            "num_internal_bends": bend_counts,
            "checkpoints_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "run_checkpoints_dir": run_checkpoint_dir,
            "seeds": seeds,
            "eligible_seeds": eligible,
            "pairs": [list(pair) for pair in pairs],
            "alignment_results": (
                None
                if args.no_permutation or args.alignment_results is None
                else str(args.alignment_results)
            ),
            "alignment_method": (
                None
                if args.no_permutation or args.alignment_results is None
                else args.alignment_method
            ),
            "permutation_protocol": (
                "none; paths use raw trained endpoints"
                if args.no_permutation
                else (
                    "all width-2 permutations"
                    if fixed_permutations is None
                    else "one fixed precomputed permutation per pair"
                )
            ),
            "methods": methods,
            "optimizer": "Adam",
            "optimization_objective": (
                "mean binary cross-entropy along the path"
                if args.objective == "mean_loss"
                else "maximum excess binary cross-entropy above the endpoint-loss chord"
            ),
            "barrier_definition": (
                "max_lambda C(theta(lambda)) - "
                "[(1-lambda) C(theta_A) + lambda C(theta_B)]"
            ),
            "steps": args.steps,
            "lr": args.lr,
            "base_num_t_samples": args.num_t_samples,
            "fitting_mode": (
                "fixed_grid"
                if args.stochastic_max_samples == 0
                else "stochastic_max_with_periodic_grid_refresh"
            ),
            "stochastic_max_samples": args.stochastic_max_samples,
            "grid_refresh_every": args.grid_refresh_every,
            "grid_refresh_points": args.grid_refresh_points,
            "previous_worst_location_retained": bool(
                args.stochastic_max_samples > 0
            ),
            "polygon_knots_always_in_fitting_grid": bool(
                args.curve_type == "polygonal"
            ),
            "eval_points": args.eval_points,
            "restarts": args.restarts,
            "compact_restarts": args.compact_restarts,
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
    results_path = output_dir / f"joint_scale_{args.curve_type}_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    write_restart_csv(bend_results, output_dir / "restart_results.csv")
    if not args.skip_plots:
        plot_combined_summary(
            bend_results,
            methods,
            output_dir / "summary_by_bends.png",
            args.hidden_size,
            args.curve_type,
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
