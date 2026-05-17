"""XOR permutation-and-scale alignment experiment implementation.

This file contains the retained search, optimization, and reporting logic for
the XOR study that compares exact alignment, Sinkhorn alignment, and scaling.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from mode_connectivity.alignment.sinkhorn_utils import stable_sinkhorn
from mode_connectivity.xor.xor_curve_fitting import (
    XOR_DATA,
    XOR_LABELS,
    SimpleMLP,
    align_models_exhaustive,
    apply_permutation_to_state,
    compute_linear_path,
    compute_path_vectors_linear,
    save_curve_npz,
    summarize_barriers,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_seeds(seeds_arg: str | None, num_networks: int) -> list[int]:
    if not seeds_arg:
        return list(range(num_networks))
    return [int(item.strip()) for item in seeds_arg.split(",") if item.strip()]


def parse_pairs(pairs_arg: str | None, available_seeds: list[int]) -> list[tuple[int, int]]:
    if not pairs_arg:
        return list(combinations(sorted(available_seeds), 2))
    available = set(available_seeds)
    parsed: list[tuple[int, int]] = []
    for item in pairs_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair specification {item!r}; expected format like '1-7'.")
        left, right = item.split("-", 1)
        pair = (int(left), int(right))
        if pair[0] not in available or pair[1] not in available:
            raise ValueError(f"Pair {item!r} references a seed not in available seeds {sorted(available)}.")
        parsed.append(pair)
    return parsed


def parse_int_list(values_arg: str | None) -> list[int]:
    if not values_arg:
        return []
    return [int(item.strip()) for item in values_arg.split(",") if item.strip()]


def parse_float_list(values_arg: str | None) -> list[float]:
    if not values_arg:
        return []
    return [float(item.strip()) for item in values_arg.split(",") if item.strip()]


def load_optional_yaml_config(path_arg: str | None) -> dict[str, Any]:
    if not path_arg:
        return {}
    path = Path(path_arg)
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML config at {path}, got {type(data).__name__}.")
    return data


def normalize_int_list_from_config(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return parse_int_list(value)
    return [int(item) for item in value]


def normalize_float_list_from_config(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return parse_float_list(value)
    return [float(item) for item in value]


def load_models_from_checkpoints(
    checkpoints_dir: Path,
    *,
    requested_seeds: list[int] | None,
    hidden_size: int | None,
) -> tuple[dict[int, SimpleMLP], dict[int, dict[str, float]], int]:
    checkpoint_paths = sorted(checkpoints_dir.glob("seed_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files matching seed_*.pt found in {checkpoints_dir}.")

    seed_filter = None if requested_seeds is None else set(requested_seeds)
    models_by_seed: dict[int, SimpleMLP] = {}
    endpoint_results: dict[int, dict[str, float]] = {}
    inferred_hidden_size: int | None = hidden_size

    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location="cpu")
        seed = int(payload["seed"])
        if seed_filter is not None and seed not in seed_filter:
            continue
        checkpoint_hidden_size = int(payload["hidden_size"])
        if inferred_hidden_size is None:
            inferred_hidden_size = checkpoint_hidden_size
        elif checkpoint_hidden_size != inferred_hidden_size:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has hidden_size={checkpoint_hidden_size}, "
                f"expected {inferred_hidden_size}."
            )
        model = SimpleMLP(hidden_size=checkpoint_hidden_size, output_size=1)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models_by_seed[seed] = model
        endpoint_results[seed] = payload["eval"]

    if not models_by_seed:
        raise RuntimeError(
            f"No checkpoints loaded from {checkpoints_dir}. "
            f"Requested seeds: {sorted(seed_filter) if seed_filter is not None else '<all>'}"
        )
    assert inferred_hidden_size is not None
    return models_by_seed, endpoint_results, inferred_hidden_size


def state_to_device_tensors(state: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.detach().to(device=device, dtype=torch.float32) for key, value in state.items()}


def clone_state_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def hidden_scaled_state_from_base(base_state: dict[str, torch.Tensor], log_scales: torch.Tensor) -> dict[str, torch.Tensor]:
    scales = torch.exp(log_scales)
    return {
        "fc1.weight": scales[:, None] * base_state["fc1.weight"],
        "fc1.bias": scales * base_state["fc1.bias"],
        "fc2.weight": base_state["fc2.weight"] / scales[None, :],
        "fc2.bias": base_state["fc2.bias"],
    }


def apply_soft_permutation_to_state(state: dict[str, torch.Tensor], perm_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "fc1.weight": perm_matrix @ state["fc1.weight"],
        "fc1.bias": perm_matrix @ state["fc1.bias"],
        "fc2.weight": state["fc2.weight"] @ perm_matrix.t(),
        "fc2.bias": state["fc2.bias"],
    }


def logits_from_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = torch.relu(XOR_DATA.to(state["fc1.weight"].device) @ state["fc1.weight"].t() + state["fc1.bias"])
    return hidden @ state["fc2.weight"].t() + state["fc2.bias"]


def xor_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, XOR_LABELS.to(logits.device))


def linear_path_loss_barrier_objective(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    for t in ts:
        interp_state = {
            key: (1.0 - t) * state_a[key] + t * state_b[key]
            for key in state_a
        }
        losses.append(xor_loss_from_logits(logits_from_state(interp_state)))
    loss_tensor = torch.stack(losses)
    endpoint_avg = 0.5 * (loss_tensor[0] + loss_tensor[-1])
    loss_barrier = torch.max(loss_tensor - endpoint_avg)
    return loss_barrier, loss_tensor


def project_soft_perm_exhaustive(soft_perm: torch.Tensor) -> list[int]:
    soft_np = soft_perm.detach().cpu().numpy()
    best_perm: tuple[int, ...] | None = None
    best_score = -float("inf")
    for perm in permutations(range(soft_np.shape[0])):
        score = float(sum(soft_np[row, col] for row, col in enumerate(perm)))
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return list(best_perm)


def optimize_sinkhorn_permutation(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    *,
    opt_steps: int,
    opt_lr: float,
    opt_t_points: int,
    eval_points: int,
    tau: float,
    sinkhorn_iters: int,
    identity_strength: float,
    patience: int,
    min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    device = torch.device("cpu")
    state_a = state_to_device_tensors(model_a.state_dict(), device)
    base_state_b = state_to_device_tensors(model_b.state_dict(), device)
    hidden_size = int(base_state_b["fc1.weight"].shape[0])

    init_logits = float(identity_strength) * torch.eye(hidden_size, dtype=torch.float32, device=device)
    logits = torch.nn.Parameter(init_logits.clone())
    optimizer = torch.optim.Adam([logits], lr=opt_lr)
    ts = torch.linspace(0.0, 1.0, int(opt_t_points), device=device)

    best_objective = float("inf")
    best_step = 0
    best_soft_perm = stable_sinkhorn(logits.detach(), tau=float(tau), num_iters=int(sinkhorn_iters)).detach().cpu().clone()
    patience_counter = 0

    for step in range(1, opt_steps + 1):
        soft_perm = stable_sinkhorn(logits, tau=float(tau), num_iters=int(sinkhorn_iters))
        aligned_state = apply_soft_permutation_to_state(base_state_b, soft_perm)
        loss_barrier, _ = linear_path_loss_barrier_objective(state_a, aligned_state, ts)
        objective = loss_barrier

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        objective_value = float(objective.item())
        improved = objective_value < (best_objective - float(min_delta))
        if improved:
            best_objective = objective_value
            best_step = step
            best_soft_perm = soft_perm.detach().cpu().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (step == 1 or step % 100 == 0 or step == opt_steps):
            print(
                f"    [sinkhorn_perm] step={step:04d} objective={objective_value:.6f} "
                f"perm_max={float(soft_perm.max()):.4f} perm_min={float(soft_perm.min()):.4f}"
            )
        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"    [sinkhorn_perm] early stop at step {step:04d}; best step {best_step:04d}")
            break

    hard_perm = project_soft_perm_exhaustive(best_soft_perm)
    hard_state = apply_permutation_to_state(model_b.state_dict(), hard_perm)
    hard_model = SimpleMLP(hidden_size=hidden_size, output_size=1)
    hard_model.load_state_dict(hard_state)
    hard_model.eval()
    hard_metrics = compute_linear_path(model_a, hard_model, num_points=eval_points)

    return {
        "model": hard_model,
        "best_step": best_step,
        "best_objective": best_objective,
        "soft_perm": best_soft_perm.numpy(),
        "hard_perm": hard_perm,
        "soft_perm_stats": {
            "min": float(best_soft_perm.min().item()),
            "mean": float(best_soft_perm.mean().item()),
            "max": float(best_soft_perm.max().item()),
        },
        "hard_metrics": hard_metrics,
    }


def run_sinkhorn_search(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    *,
    eval_points: int,
    base_steps: int,
    base_lr: float,
    base_t_points: int,
    base_tau: float,
    base_sinkhorn_iters: int,
    base_identity_strength: float,
    base_patience: int,
    base_min_delta: float,
    barrier_epsilon: float,
    search_steps: list[int],
    search_lrs: list[float],
    search_taus: list[float],
    search_identity_strengths: list[float],
    search_patience: int,
    search_min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    def run_attempt(
        *,
        steps: int,
        lr: float,
        t_points: int,
        tau: float,
        sinkhorn_iters: int,
        identity_strength: float,
        patience: int,
        min_delta: float,
        label: str,
    ) -> dict[str, Any]:
        result = optimize_sinkhorn_permutation(
            model_a,
            model_b,
            opt_steps=steps,
            opt_lr=lr,
            opt_t_points=t_points,
            eval_points=eval_points,
            tau=tau,
            sinkhorn_iters=sinkhorn_iters,
            identity_strength=identity_strength,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
        )
        return {
            "label": label,
            "steps": int(steps),
            "lr": float(lr),
            "t_points": int(t_points),
            "tau": float(tau),
            "sinkhorn_iters": int(sinkhorn_iters),
            "identity_strength": float(identity_strength),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "result": result,
            "metrics": result["hard_metrics"],
        }

    initial_attempt = run_attempt(
        steps=base_steps,
        lr=base_lr,
        t_points=base_t_points,
        tau=base_tau,
        sinkhorn_iters=base_sinkhorn_iters,
        identity_strength=base_identity_strength,
        patience=base_patience,
        min_delta=base_min_delta,
        label="initial",
    )
    attempts.append(initial_attempt)
    best_attempt = initial_attempt

    if float(initial_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
        return {
            "selected": initial_attempt,
            "attempts": attempts,
            "search_triggered": False,
            "satisfied_epsilon": True,
        }

    candidate_steps = search_steps if search_steps else [base_steps]
    candidate_lrs = search_lrs if search_lrs else [base_lr]
    candidate_taus = search_taus if search_taus else [base_tau]
    candidate_identity_strengths = search_identity_strengths if search_identity_strengths else [base_identity_strength]
    search_patience_counter = 0

    for steps in candidate_steps:
        for lr in candidate_lrs:
            for tau in candidate_taus:
                for identity_strength in candidate_identity_strengths:
                    if (
                        int(steps) == int(base_steps)
                        and abs(float(lr) - float(base_lr)) <= 1e-12
                        and abs(float(tau) - float(base_tau)) <= 1e-12
                        and abs(float(identity_strength) - float(base_identity_strength)) <= 1e-12
                    ):
                        continue
                    attempt = run_attempt(
                        steps=int(steps),
                        lr=float(lr),
                        t_points=base_t_points,
                        tau=float(tau),
                        sinkhorn_iters=base_sinkhorn_iters,
                        identity_strength=float(identity_strength),
                        patience=base_patience,
                        min_delta=base_min_delta,
                        label="search",
                    )
                    attempts.append(attempt)
                    improved = float(attempt["metrics"]["loss_barrier"]) < (
                        float(best_attempt["metrics"]["loss_barrier"]) - float(search_min_delta)
                    )
                    if improved:
                        best_attempt = attempt
                        search_patience_counter = 0
                    else:
                        search_patience_counter += 1
                    if float(attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
                        return {
                            "selected": attempt,
                            "attempts": attempts,
                            "search_triggered": True,
                            "satisfied_epsilon": True,
                        }
                    if search_patience > 0 and search_patience_counter >= search_patience:
                        return {
                            "selected": best_attempt,
                            "attempts": attempts,
                            "search_triggered": True,
                            "satisfied_epsilon": float(best_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
                        }

    if float(best_attempt["metrics"]["loss_barrier"]) < float(initial_attempt["metrics"]["loss_barrier"]):
        selected = best_attempt
    else:
        selected = initial_attempt
    return {
        "selected": selected,
        "attempts": attempts,
        "search_triggered": True,
        "satisfied_epsilon": float(selected["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
    }


def optimize_scales_for_barrier(
    model_a: SimpleMLP,
    permuted_model_b: SimpleMLP,
    *,
    opt_steps: int,
    opt_lr: float,
    opt_t_points: int,
    scale_reg: float,
    patience: int,
    min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    device = torch.device("cpu")
    state_a = state_to_device_tensors(model_a.state_dict(), device)
    base_state_b = state_to_device_tensors(permuted_model_b.state_dict(), device)
    hidden_size = int(base_state_b["fc1.weight"].shape[0])

    log_scales = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([log_scales], lr=opt_lr)
    ts = torch.linspace(0.0, 1.0, int(opt_t_points), device=device)

    best_objective = float("inf")
    best_step = 0
    best_log_scales = log_scales.detach().cpu().clone()
    best_loss_tensor: torch.Tensor | None = None
    patience_counter = 0

    for step in range(1, opt_steps + 1):
        scaled_state = hidden_scaled_state_from_base(base_state_b, log_scales)
        loss_barrier, loss_tensor = linear_path_loss_barrier_objective(state_a, scaled_state, ts)
        objective = loss_barrier + float(scale_reg) * torch.sum(log_scales ** 2)

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        objective_value = float(objective.item())
        improved = objective_value < (best_objective - float(min_delta))
        if improved:
            best_objective = objective_value
            best_step = step
            best_log_scales = log_scales.detach().cpu().clone()
            best_loss_tensor = loss_tensor.detach().cpu().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (step == 1 or step % 100 == 0 or step == opt_steps):
            scales = torch.exp(log_scales.detach())
            print(
                f"    [scale_only] step={step:04d} objective={objective_value:.6f} "
                f"barrier={float(loss_barrier.item()):.6f} "
                f"scale[min={float(scales.min()):.4f}, mean={float(scales.mean()):.4f}, max={float(scales.max()):.4f}]"
            )
        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"    [scale_only] early stop at step {step:04d}; best step {best_step:04d}")
            break

    best_state = hidden_scaled_state_from_base(base_state_b, best_log_scales.to(device))
    scaled_model = SimpleMLP(hidden_size=hidden_size, output_size=1)
    scaled_model.load_state_dict({key: value.detach().cpu() for key, value in best_state.items()})
    scaled_model.eval()

    best_scales = torch.exp(best_log_scales).numpy()
    return {
        "model": scaled_model,
        "best_step": best_step,
        "best_objective": best_objective,
        "best_log_scales": best_log_scales.numpy(),
        "best_scales": best_scales,
        "scale_stats": {
            "min": float(best_scales.min()),
            "mean": float(best_scales.mean()),
            "max": float(best_scales.max()),
        },
        "optimization_curve_losses": None if best_loss_tensor is None else best_loss_tensor.numpy().tolist(),
    }


def optimize_sinkhorn_permutation_scale(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    *,
    opt_steps: int,
    opt_lr: float,
    opt_t_points: int,
    eval_points: int,
    tau: float,
    sinkhorn_iters: int,
    identity_strength: float,
    scale_reg: float,
    patience: int,
    min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    device = torch.device("cpu")
    state_a = state_to_device_tensors(model_a.state_dict(), device)
    base_state_b = state_to_device_tensors(model_b.state_dict(), device)
    hidden_size = int(base_state_b["fc1.weight"].shape[0])

    logits = torch.nn.Parameter(float(identity_strength) * torch.eye(hidden_size, dtype=torch.float32, device=device))
    log_scales = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([logits, log_scales], lr=opt_lr)
    ts = torch.linspace(0.0, 1.0, int(opt_t_points), device=device)

    best_objective = float("inf")
    best_step = 0
    best_soft_perm = stable_sinkhorn(logits.detach(), tau=float(tau), num_iters=int(sinkhorn_iters)).detach().cpu().clone()
    best_log_scales = log_scales.detach().cpu().clone()
    patience_counter = 0

    for step in range(1, opt_steps + 1):
        soft_perm = stable_sinkhorn(logits, tau=float(tau), num_iters=int(sinkhorn_iters))
        soft_aligned_state = apply_soft_permutation_to_state(base_state_b, soft_perm)
        scaled_state = hidden_scaled_state_from_base(soft_aligned_state, log_scales)
        loss_barrier, _ = linear_path_loss_barrier_objective(state_a, scaled_state, ts)
        objective = loss_barrier + float(scale_reg) * torch.sum(log_scales ** 2)

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        objective_value = float(objective.item())
        improved = objective_value < (best_objective - float(min_delta))
        if improved:
            best_objective = objective_value
            best_step = step
            best_soft_perm = soft_perm.detach().cpu().clone()
            best_log_scales = log_scales.detach().cpu().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (step == 1 or step % 100 == 0 or step == opt_steps):
            scales = torch.exp(log_scales.detach())
            print(
                f"    [sinkhorn_perm_scale] step={step:04d} objective={objective_value:.6f} "
                f"perm_max={float(soft_perm.max()):.4f} perm_min={float(soft_perm.min()):.4f} "
                f"scale[min={float(scales.min()):.4f}, mean={float(scales.mean()):.4f}, max={float(scales.max()):.4f}]"
            )
        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"    [sinkhorn_perm_scale] early stop at step {step:04d}; best step {best_step:04d}")
            break

    hard_perm = project_soft_perm_exhaustive(best_soft_perm)
    hard_state = apply_permutation_to_state(model_b.state_dict(), hard_perm)
    hard_state_device = state_to_device_tensors(hard_state, device)
    best_state = hidden_scaled_state_from_base(hard_state_device, best_log_scales.to(device))
    aligned_model = SimpleMLP(hidden_size=hidden_size, output_size=1)
    aligned_model.load_state_dict({key: value.detach().cpu() for key, value in best_state.items()})
    aligned_model.eval()
    hard_metrics = compute_linear_path(model_a, aligned_model, num_points=eval_points)

    best_scales = torch.exp(best_log_scales).numpy()
    return {
        "model": aligned_model,
        "best_step": best_step,
        "best_objective": best_objective,
        "soft_perm": best_soft_perm.numpy(),
        "hard_perm": hard_perm,
        "soft_perm_stats": {
            "min": float(best_soft_perm.min().item()),
            "mean": float(best_soft_perm.mean().item()),
            "max": float(best_soft_perm.max().item()),
        },
        "best_log_scales": best_log_scales.numpy(),
        "best_scales": best_scales,
        "scale_stats": {
            "min": float(best_scales.min()),
            "mean": float(best_scales.mean()),
            "max": float(best_scales.max()),
        },
        "hard_metrics": hard_metrics,
    }


def run_sinkhorn_perm_scale_search(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    *,
    eval_points: int,
    base_steps: int,
    base_lr: float,
    base_t_points: int,
    base_tau: float,
    base_sinkhorn_iters: int,
    base_identity_strength: float,
    base_scale_reg: float,
    base_patience: int,
    base_min_delta: float,
    barrier_epsilon: float | None,
    search_steps: list[int],
    search_lrs: list[float],
    search_taus: list[float],
    search_identity_strengths: list[float],
    search_regs: list[float],
    search_patience: int,
    search_min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    def run_attempt(
        *,
        steps: int,
        lr: float,
        t_points: int,
        tau: float,
        sinkhorn_iters: int,
        identity_strength: float,
        scale_reg: float,
        patience: int,
        min_delta: float,
        label: str,
    ) -> dict[str, Any]:
        result = optimize_sinkhorn_permutation_scale(
            model_a,
            model_b,
            opt_steps=steps,
            opt_lr=lr,
            opt_t_points=t_points,
            eval_points=eval_points,
            tau=tau,
            sinkhorn_iters=sinkhorn_iters,
            identity_strength=identity_strength,
            scale_reg=scale_reg,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
        )
        return {
            "label": label,
            "steps": int(steps),
            "lr": float(lr),
            "t_points": int(t_points),
            "tau": float(tau),
            "sinkhorn_iters": int(sinkhorn_iters),
            "identity_strength": float(identity_strength),
            "scale_reg": float(scale_reg),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "result": result,
            "metrics": result["hard_metrics"],
        }

    initial_attempt = run_attempt(
        steps=base_steps,
        lr=base_lr,
        t_points=base_t_points,
        tau=base_tau,
        sinkhorn_iters=base_sinkhorn_iters,
        identity_strength=base_identity_strength,
        scale_reg=base_scale_reg,
        patience=base_patience,
        min_delta=base_min_delta,
        label="initial",
    )
    attempts.append(initial_attempt)
    best_attempt = initial_attempt

    if barrier_epsilon is None or float(initial_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
        return {
            "selected": initial_attempt,
            "attempts": attempts,
            "search_triggered": False,
            "satisfied_epsilon": barrier_epsilon is None or float(initial_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
        }

    candidate_steps = search_steps if search_steps else [base_steps]
    candidate_lrs = search_lrs if search_lrs else [base_lr]
    candidate_taus = search_taus if search_taus else [base_tau]
    candidate_identity_strengths = search_identity_strengths if search_identity_strengths else [base_identity_strength]
    candidate_regs = search_regs if search_regs else [base_scale_reg]
    search_patience_counter = 0

    for steps in candidate_steps:
        for lr in candidate_lrs:
            for tau in candidate_taus:
                for identity_strength in candidate_identity_strengths:
                    for scale_reg in candidate_regs:
                        if (
                            int(steps) == int(base_steps)
                            and abs(float(lr) - float(base_lr)) <= 1e-12
                            and abs(float(tau) - float(base_tau)) <= 1e-12
                            and abs(float(identity_strength) - float(base_identity_strength)) <= 1e-12
                            and abs(float(scale_reg) - float(base_scale_reg)) <= 1e-12
                        ):
                            continue
                        attempt = run_attempt(
                            steps=int(steps),
                            lr=float(lr),
                            t_points=base_t_points,
                            tau=float(tau),
                            sinkhorn_iters=base_sinkhorn_iters,
                            identity_strength=float(identity_strength),
                            scale_reg=float(scale_reg),
                            patience=base_patience,
                            min_delta=base_min_delta,
                            label="search",
                        )
                        attempts.append(attempt)
                        improved = float(attempt["metrics"]["loss_barrier"]) < (
                            float(best_attempt["metrics"]["loss_barrier"]) - float(search_min_delta)
                        )
                        if improved:
                            best_attempt = attempt
                            search_patience_counter = 0
                        else:
                            search_patience_counter += 1
                        if float(attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
                            return {
                                "selected": attempt,
                                "attempts": attempts,
                                "search_triggered": True,
                                "satisfied_epsilon": True,
                            }
                        if search_patience > 0 and search_patience_counter >= search_patience:
                            return {
                                "selected": best_attempt,
                                "attempts": attempts,
                                "search_triggered": True,
                                "satisfied_epsilon": float(best_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
                            }

    if float(best_attempt["metrics"]["loss_barrier"]) < float(initial_attempt["metrics"]["loss_barrier"]):
        selected = best_attempt
    else:
        selected = initial_attempt
    return {
        "selected": selected,
        "attempts": attempts,
        "search_triggered": True,
        "satisfied_epsilon": float(selected["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
    }


def run_perm_scale_search(
    model_a: SimpleMLP,
    perm_model: SimpleMLP,
    *,
    eval_points: int,
    base_steps: int,
    base_lr: float,
    base_t_points: int,
    base_scale_reg: float,
    base_patience: int,
    base_min_delta: float,
    barrier_epsilon: float | None,
    search_steps: list[int],
    search_lrs: list[float],
    search_regs: list[float],
    search_patience: int,
    search_min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    def run_attempt(
        *,
        steps: int,
        lr: float,
        t_points: int,
        scale_reg: float,
        patience: int,
        min_delta: float,
        label: str,
    ) -> dict[str, Any]:
        scale_result = optimize_scales_for_barrier(
            model_a,
            perm_model,
            opt_steps=steps,
            opt_lr=lr,
            opt_t_points=t_points,
            scale_reg=scale_reg,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
        )
        metrics = compute_linear_path(model_a, scale_result["model"], num_points=eval_points)
        return {
            "label": label,
            "steps": int(steps),
            "lr": float(lr),
            "t_points": int(t_points),
            "scale_reg": float(scale_reg),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "scale_result": scale_result,
            "metrics": metrics,
        }

    initial_attempt = run_attempt(
        steps=base_steps,
        lr=base_lr,
        t_points=base_t_points,
        scale_reg=base_scale_reg,
        patience=base_patience,
        min_delta=base_min_delta,
        label="initial",
    )
    attempts.append(initial_attempt)
    best_attempt = initial_attempt

    if barrier_epsilon is None or float(initial_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
        return {
            "selected": initial_attempt,
            "attempts": attempts,
            "search_triggered": False,
            "satisfied_epsilon": barrier_epsilon is None or float(initial_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
        }

    candidate_steps = search_steps if search_steps else [base_steps]
    candidate_lrs = search_lrs if search_lrs else [base_lr]
    candidate_regs = search_regs if search_regs else [base_scale_reg]
    search_patience_counter = 0

    for steps in candidate_steps:
        for lr in candidate_lrs:
            for scale_reg in candidate_regs:
                if (
                    int(steps) == int(base_steps)
                    and abs(float(lr) - float(base_lr)) <= 1e-12
                    and abs(float(scale_reg) - float(base_scale_reg)) <= 1e-12
                ):
                    continue
                attempt = run_attempt(
                    steps=int(steps),
                    lr=float(lr),
                    t_points=base_t_points,
                    scale_reg=float(scale_reg),
                    patience=base_patience,
                    min_delta=base_min_delta,
                    label="search",
                )
                attempts.append(attempt)
                improved = float(attempt["metrics"]["loss_barrier"]) < (
                    float(best_attempt["metrics"]["loss_barrier"]) - float(search_min_delta)
                )
                if improved:
                    best_attempt = attempt
                    search_patience_counter = 0
                else:
                    search_patience_counter += 1
                if float(attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon):
                    return {
                        "selected": attempt,
                        "attempts": attempts,
                        "search_triggered": True,
                        "satisfied_epsilon": True,
                    }
                if search_patience > 0 and search_patience_counter >= search_patience:
                    return {
                        "selected": best_attempt,
                        "attempts": attempts,
                        "search_triggered": True,
                        "satisfied_epsilon": float(best_attempt["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
                    }

    if float(best_attempt["metrics"]["loss_barrier"]) < float(initial_attempt["metrics"]["loss_barrier"]):
        selected = best_attempt
    else:
        selected = initial_attempt
    return {
        "selected": selected,
        "attempts": attempts,
        "search_triggered": True,
        "satisfied_epsilon": float(selected["metrics"]["loss_barrier"]) <= float(barrier_epsilon),
    }


def align_models_exhaustive_with_scale(
    model_a: SimpleMLP,
    model_b: SimpleMLP,
    *,
    opt_steps: int,
    opt_lr: float,
    opt_t_points: int,
    eval_points: int,
    scale_reg: float,
    patience: int,
    min_delta: float,
    verbose: bool,
) -> dict[str, Any]:
    state_b = model_b.state_dict()
    hidden_size = int(state_b["fc1.weight"].shape[0])
    all_perms = list(permutations(range(hidden_size)))

    best_perm: list[int] | None = None
    best_scaled_model: SimpleMLP | None = None
    best_metrics: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []

    if verbose:
        print(f"    Exhaustive permutations + scale: {len(all_perms)}")

    for perm in all_perms:
        perm_state = apply_permutation_to_state(state_b, perm)
        perm_model = SimpleMLP(hidden_size=hidden_size, output_size=1)
        perm_model.load_state_dict(perm_state)
        scale_result = optimize_scales_for_barrier(
            model_a,
            perm_model,
            opt_steps=opt_steps,
            opt_lr=opt_lr,
            opt_t_points=opt_t_points,
            scale_reg=scale_reg,
            patience=patience,
            min_delta=min_delta,
            verbose=False,
        )
        scaled_model = scale_result["model"]
        metrics = compute_linear_path(model_a, scaled_model, num_points=eval_points)
        all_results.append(
            {
                "perm": list(perm),
                "barrier": float(metrics["barrier"]),
                "loss_barrier": float(metrics["loss_barrier"]),
                "mean_loss": float(np.mean(metrics["loss"])),
                "min_accuracy": float(metrics["min_accuracy"]),
                "scale_stats": scale_result["scale_stats"],
            }
        )

        if (
            best_metrics is None
            or metrics["barrier"] < best_metrics["barrier"]
            or (
                metrics["barrier"] == best_metrics["barrier"]
                and metrics["loss_barrier"] < best_metrics["loss_barrier"]
            )
        ):
            best_perm = list(perm)
            best_scaled_model = scaled_model
            best_metrics = metrics

    assert best_perm is not None and best_scaled_model is not None and best_metrics is not None
    return {
        "model": best_scaled_model,
        "best_perm": best_perm,
        "metrics": best_metrics,
        "all_results": all_results,
    }


def save_pair_curve(pair_dir: Path, name: str, model_a: SimpleMLP, model_b: SimpleMLP, metrics: dict[str, Any]) -> str:
    npz_path = pair_dir / name / "curve.npz"
    path_vectors = compute_path_vectors_linear(model_a.state_dict(), model_b.state_dict(), metrics["t"])
    save_curve_npz(str(npz_path), metrics["t"], metrics["loss"], metrics["accuracy"], path_vectors)
    return str(npz_path)


def get_plot_style() -> tuple[dict[str, str], dict[str, str]]:
    colors = {
        "no_alignment": "tab:gray",
        "best_permutation": "tab:blue",
        "sinkhorn_permutation": "tab:orange",
        "perm_plus_scale": "tab:green",
        "sinkhorn_perm_plus_scale": "tab:purple",
        "joint_perm_scale_exact": "tab:red",
    }
    labels = {
        "no_alignment": "No Alignment",
        "best_permutation": "Best Exhaustive Permutation",
        "sinkhorn_permutation": "Sinkhorn Permutation Only (From Scratch)",
        "sinkhorn_perm_plus_scale": "Sinkhorn Permutation + Scale (From Scratch)",
        "perm_plus_scale": "Best Exhaustive Permutation + Scale Refinement",
        "joint_perm_scale_exact": "Joint permutation + scale (exact)",
    }
    return colors, labels


def get_core_plot_methods() -> list[str]:
    return [
        "no_alignment",
        "sinkhorn_permutation",
        "best_permutation",
        "sinkhorn_perm_plus_scale",
        "perm_plus_scale",
    ]


def plot_pair_curves(pair_payload: dict[str, Any], output_path: Path, metric_key: str, ylabel: str, title: str) -> None:
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods()
    plt.figure()
    for method_key in ordered_methods:
        metrics = pair_payload[method_key]["metrics"]
        t = np.asarray(metrics["t"], dtype=np.float64)
        values = np.asarray(metrics["accuracy" if metric_key == "acc" else "loss"], dtype=np.float64)
        plt.plot(t, values, label=labels[method_key], color=colors[method_key])
    plt.xlabel("t (interpolation parameter)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pair_bar_metrics(pair_payload: dict[str, Any], output_path: Path, title: str) -> None:
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods()
    x = np.arange(len(ordered_methods), dtype=np.float64)
    loss_barriers = np.asarray(
        [float(pair_payload[method_key]["metrics"]["loss_barrier"]) for method_key in ordered_methods],
        dtype=np.float64,
    )
    min_accuracies = np.asarray(
        [float(pair_payload[method_key]["metrics"]["min_accuracy"]) for method_key in ordered_methods],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x, loss_barriers, color=[colors[method_key] for method_key in ordered_methods])
    axes[0].set_title("Loss Barrier")
    axes[0].set_ylabel("Loss Barrier")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[method_key] for method_key in ordered_methods], rotation=20, ha="right")

    axes[1].bar(x, min_accuracies, color=[colors[method_key] for method_key in ordered_methods])
    axes[1].set_title("Minimum Interpolation Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[method_key] for method_key in ordered_methods], rotation=20, ha="right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_method(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    loss_barriers = [float(item["loss_barrier"]) for item in metrics_list]
    acc_barriers = [float(item["barrier"]) for item in metrics_list]
    mean_losses = [float(np.mean(item["loss"])) for item in metrics_list]
    min_accs = [float(item["min_accuracy"]) for item in metrics_list]
    return {
        "loss_barrier_mean": float(np.mean(loss_barriers)),
        "loss_barrier_std": float(np.std(loss_barriers)),
        "barrier_mean": float(np.mean(acc_barriers)),
        "barrier_std": float(np.std(acc_barriers)),
        "mean_interp_loss_mean": float(np.mean(mean_losses)),
        "mean_interp_loss_std": float(np.std(mean_losses)),
        "min_interp_acc_mean": float(np.mean(min_accs)),
        "min_interp_acc_std": float(np.std(min_accs)),
    }


def aggregate_curves(pair_results: list[dict[str, Any]], method_key: str) -> dict[str, list[float]]:
    losses = np.asarray([pair[method_key]["metrics"]["loss"] for pair in pair_results], dtype=np.float64)
    accs = np.asarray([pair[method_key]["metrics"]["accuracy"] for pair in pair_results], dtype=np.float64)
    ts = np.asarray(pair_results[0][method_key]["metrics"]["t"], dtype=np.float64)
    return {
        "t": ts.tolist(),
        "loss_mean": losses.mean(axis=0).tolist(),
        "loss_std": losses.std(axis=0).tolist(),
        "acc_mean": accs.mean(axis=0).tolist(),
        "acc_std": accs.std(axis=0).tolist(),
    }


def plot_aggregate_curves(
    aggregates: dict[str, dict[str, list[float]]],
    output_path: Path,
    metric_key: str,
    ylabel: str,
    title: str | None,
    include_std: bool,
    show_legend: bool = True,
    title_fontsize: int = 16,
    ylabel_fontsize: int = 16,
    ylabel_fontweight: str = "bold",
    legend_fontsize: int = 14,
) -> None:
    plt.figure()
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods() + ["joint_perm_scale_exact"]
    for method_key in ordered_methods:
        if method_key not in aggregates:
            continue
        aggregate = aggregates[method_key]
        t = np.asarray(aggregate["t"], dtype=np.float64)
        mean = np.asarray(aggregate[f"{metric_key}_mean"], dtype=np.float64)
        std = np.asarray(aggregate[f"{metric_key}_std"], dtype=np.float64)
        color = colors[method_key]
        plt.plot(t, mean, label=labels[method_key], color=color)
        if include_std:
            plt.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)
    plt.xlabel("t (interpolation parameter)")
    plt.ylabel(ylabel, fontsize=ylabel_fontsize, fontweight=ylabel_fontweight)
    if title:
        plt.title(title, fontsize=title_fontsize)
    if show_legend:
        plt.legend(fontsize=legend_fontsize)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_aggregate_bar_metrics(
    pair_results: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods()
    x = np.arange(len(ordered_methods), dtype=np.float64)
    loss_barriers = np.asarray(
        [
            [float(pair[method_key]["metrics"]["loss_barrier"]) for method_key in ordered_methods]
            for pair in pair_results
        ],
        dtype=np.float64,
    )
    min_accuracies = np.asarray(
        [
            [float(pair[method_key]["metrics"]["min_accuracy"]) for method_key in ordered_methods]
            for pair in pair_results
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        x,
        loss_barriers.mean(axis=0),
        yerr=loss_barriers.std(axis=0),
        color=[colors[method_key] for method_key in ordered_methods],
        capsize=4,
    )
    axes[0].set_title("Mean Loss Barrier Across Pairs")
    axes[0].set_ylabel("Loss Barrier")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[method_key] for method_key in ordered_methods], rotation=20, ha="right")

    axes[1].bar(
        x,
        min_accuracies.mean(axis=0),
        yerr=min_accuracies.std(axis=0),
        color=[colors[method_key] for method_key in ordered_methods],
        capsize=4,
    )
    axes[1].set_title("Mean Minimum Interpolation Accuracy Across Pairs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[method_key] for method_key in ordered_methods], rotation=20, ha="right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_aggregate_stats_txt(
    aggregates: dict[str, dict[str, list[float]]],
    output_path: Path,
    metric_key: str,
) -> None:
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods() + ["joint_perm_scale_exact"]
    with open(output_path, "w") as handle:
        for method_key in ordered_methods:
            if method_key not in aggregates:
                continue
            aggregate = aggregates[method_key]
            t = np.asarray(aggregate["t"], dtype=np.float64)
            mean = np.asarray(aggregate[f"{metric_key}_mean"], dtype=np.float64)
            std = np.asarray(aggregate[f"{metric_key}_std"], dtype=np.float64)
            handle.write(f"[{method_key}] {labels[method_key]}\n")
            handle.write("t mean std\n")
            for t_value, mean_value, std_value in zip(t, mean, std):
                handle.write(f"{t_value:.8f} {mean_value:.8f} {std_value:.8f}\n")
            handle.write("\n")


def write_method_summary_markdown(summary: dict[str, Any], output_path: Path, include_joint_perm_scale: bool) -> None:
    colors, labels = get_plot_style()
    ordered_methods = get_core_plot_methods()
    if include_joint_perm_scale:
        ordered_methods.append("joint_perm_scale_exact")

    rows = [
        "| Method | Loss Barrier | Mean Interp Loss | Min Interp Acc | Accuracy Barrier |",
        "|---|---:|---:|---:|---:|",
    ]
    for method_key in ordered_methods:
        method_summary = summary[method_key]
        rows.append(
            "| {} | {}+{} | {}+{} | {}+{} | {}+{} |".format(
                labels[method_key],
                f"{method_summary['loss_barrier_mean']:.6f}",
                f"{method_summary['loss_barrier_std']:.6f}",
                f"{method_summary['mean_interp_loss_mean']:.6f}",
                f"{method_summary['mean_interp_loss_std']:.6f}",
                f"{method_summary['min_interp_acc_mean']:.6f}",
                f"{method_summary['min_interp_acc_std']:.6f}",
                f"{method_summary['barrier_mean']:.6f}",
                f"{method_summary['barrier_std']:.6f}",
            )
        )
    output_path.write_text("\n".join(rows) + "\n")


def write_pairwise_comparison_table(summary: dict[str, Any], num_pairs: int, output_path: Path) -> None:
    counts = summary["pairwise_comparison_counts"]
    rows = [
        f"comparison_tolerance: {counts['comparison_tolerance']:.1e}",
        "",
        "| Comparison | Count |",
        "|---|---:|",
        f"| perm+scale <= perm | {counts['perm_plus_scale_better_than_permutation']}/{num_pairs} |",
        f"| sinkhorn_perm+scale <= perm | {counts['sinkhorn_perm_plus_scale_better_than_permutation']}/{num_pairs} |",
        f"| sinkhorn_perm == perm | {counts['sinkhorn_permutation_equal_to_permutation']}/{num_pairs} |",
        f"| perm+scale <= sinkhorn_perm+scale | {counts['perm_plus_scale_better_than_sinkhorn_perm_plus_scale']}/{num_pairs} |",
    ]
    output_path.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="XOR exact permutation vs permutation+scale experiment")
    parser.add_argument("--output", type=Path, default=Path("results/xor/xor_5h_perm_vs_scale"))
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--curve-eval-points", type=int, default=61)
    parser.add_argument(
        "--sinkhorn-search-config",
        type=str,
        default="configs/experiments/xor/search/sinkhorn_permutation.yaml",
    )
    parser.add_argument("--sinkhorn-opt-steps", type=int, default=500)
    parser.add_argument("--sinkhorn-opt-lr", type=float, default=0.01)
    parser.add_argument("--sinkhorn-opt-t-points", type=int, default=31)
    parser.add_argument("--sinkhorn-tau", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--sinkhorn-identity-strength", type=float, default=1.0)
    parser.add_argument("--sinkhorn-patience", type=int, default=100)
    parser.add_argument("--sinkhorn-min-delta", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-search-steps", type=str, default="")
    parser.add_argument("--sinkhorn-search-lrs", type=str, default="")
    parser.add_argument("--sinkhorn-search-taus", type=str, default="")
    parser.add_argument("--sinkhorn-search-identity-strengths", type=str, default="")
    parser.add_argument("--sinkhorn-search-patience", type=int, default=0)
    parser.add_argument("--sinkhorn-search-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--perm-scale-search-config",
        type=str,
        default="configs/experiments/xor/search/permutation_scale.yaml",
    )
    parser.add_argument(
        "--sinkhorn-perm-scale-search-config",
        type=str,
        default="configs/experiments/xor/search/sinkhorn_permutation_scale.yaml",
    )
    parser.add_argument("--scale-opt-steps", type=int, default=500)
    parser.add_argument("--scale-opt-lr", type=float, default=0.01)
    parser.add_argument("--scale-opt-t-points", type=int, default=31)
    parser.add_argument("--scale-reg", type=float, default=0.001)
    parser.add_argument("--scale-patience", type=int, default=150)
    parser.add_argument("--scale-min-delta", type=float, default=1e-6)
    parser.add_argument("--perm-scale-target-epsilon", type=float, default=None)
    parser.add_argument("--perm-scale-search-steps", type=str, default="")
    parser.add_argument("--perm-scale-search-lrs", type=str, default="")
    parser.add_argument("--perm-scale-search-regs", type=str, default="")
    parser.add_argument("--perm-scale-search-patience", type=int, default=0)
    parser.add_argument("--perm-scale-search-min-delta", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-perm-scale-target-epsilon", type=float, default=None)
    parser.add_argument("--sinkhorn-perm-scale-search-steps", type=str, default="")
    parser.add_argument("--sinkhorn-perm-scale-search-lrs", type=str, default="")
    parser.add_argument("--sinkhorn-perm-scale-search-taus", type=str, default="")
    parser.add_argument("--sinkhorn-perm-scale-search-identity-strengths", type=str, default="")
    parser.add_argument("--sinkhorn-perm-scale-search-regs", type=str, default="")
    parser.add_argument("--sinkhorn-perm-scale-search-patience", type=int, default=0)
    parser.add_argument("--sinkhorn-perm-scale-search-min-delta", type=float, default=1e-6)
    parser.add_argument("--run-joint-perm-scale", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output.resolve())
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    evaluations_dir = ensure_dir(output_dir / "evaluations")
    plots_dir = ensure_dir(output_dir / "plots")

    sinkhorn_search_cfg = load_optional_yaml_config(args.sinkhorn_search_config)
    perm_scale_search_cfg = load_optional_yaml_config(args.perm_scale_search_config)
    sinkhorn_perm_scale_search_cfg = load_optional_yaml_config(args.sinkhorn_perm_scale_search_config)

    requested_seeds = None if args.seeds is None else parse_seeds(args.seeds, 0)
    print("=" * 80)
    print("XOR PERMUTATION VS PERMUTATION+SCALE")
    print("=" * 80)
    print(f"checkpoints_dir: {args.checkpoints_dir.resolve()}")
    print(f"requested_seeds: {requested_seeds if requested_seeds is not None else '<all>'}")
    print(f"curve_eval_points: {args.curve_eval_points}")
    print(f"sinkhorn_search_config: {args.sinkhorn_search_config or '<disabled>'}")
    print(f"sinkhorn_opt_steps: {args.sinkhorn_opt_steps}")
    print(f"sinkhorn_opt_lr: {args.sinkhorn_opt_lr}")
    print(f"sinkhorn_tau: {args.sinkhorn_tau}")
    print(f"perm_scale_search_config: {args.perm_scale_search_config or '<disabled>'}")
    print(f"sinkhorn_perm_scale_search_config: {args.sinkhorn_perm_scale_search_config or '<disabled>'}")
    print(f"scale_opt_steps: {args.scale_opt_steps}")
    print(f"scale_opt_lr: {args.scale_opt_lr}")
    print(f"run_joint_perm_scale: {args.run_joint_perm_scale}")
    print("")

    print("Step 1/4: loading saved XOR checkpoints")
    models_by_seed, endpoint_results, hidden_size = load_models_from_checkpoints(
        args.checkpoints_dir.resolve(),
        requested_seeds=requested_seeds,
        hidden_size=args.hidden_size,
    )
    available_seeds = sorted(models_by_seed)
    if len(available_seeds) < 2:
        raise RuntimeError(f"Need at least 2 trained XOR models, got {len(available_seeds)}.")

    pairs = parse_pairs(args.pairs, available_seeds)
    print(f"Loaded hidden_size: {hidden_size}")
    print(f"Step 2/4: evaluating {len(pairs)} pairs from seeds {available_seeds}")

    perm_scale_search_steps = (
        parse_int_list(args.perm_scale_search_steps)
        if args.perm_scale_search_steps
        else normalize_int_list_from_config(perm_scale_search_cfg.get("search_steps"))
    )
    perm_scale_search_lrs = (
        parse_float_list(args.perm_scale_search_lrs)
        if args.perm_scale_search_lrs
        else normalize_float_list_from_config(perm_scale_search_cfg.get("search_lrs"))
    )
    perm_scale_search_regs = (
        parse_float_list(args.perm_scale_search_regs)
        if args.perm_scale_search_regs
        else normalize_float_list_from_config(perm_scale_search_cfg.get("search_regs"))
    )
    perm_scale_target_epsilon = (
        float(args.perm_scale_target_epsilon)
        if args.perm_scale_target_epsilon is not None
        else (
            None
            if perm_scale_search_cfg.get("target_epsilon") is None
            else float(perm_scale_search_cfg.get("target_epsilon"))
        )
    )
    perm_scale_search_patience = (
        int(args.perm_scale_search_patience)
        if int(args.perm_scale_search_patience) > 0
        else int(perm_scale_search_cfg.get("search_patience", 0))
    )
    perm_scale_search_min_delta = (
        float(args.perm_scale_search_min_delta)
        if abs(float(args.perm_scale_search_min_delta) - 1e-6) > 1e-12
        else float(perm_scale_search_cfg.get("search_min_delta", 1e-6))
    )
    sinkhorn_search_steps = (
        parse_int_list(args.sinkhorn_search_steps)
        if args.sinkhorn_search_steps
        else normalize_int_list_from_config(sinkhorn_search_cfg.get("search_steps"))
    )
    sinkhorn_search_lrs = (
        parse_float_list(args.sinkhorn_search_lrs)
        if args.sinkhorn_search_lrs
        else normalize_float_list_from_config(sinkhorn_search_cfg.get("search_lrs"))
    )
    sinkhorn_search_taus = (
        parse_float_list(args.sinkhorn_search_taus)
        if args.sinkhorn_search_taus
        else normalize_float_list_from_config(sinkhorn_search_cfg.get("search_taus"))
    )
    sinkhorn_search_identity_strengths = (
        parse_float_list(args.sinkhorn_search_identity_strengths)
        if args.sinkhorn_search_identity_strengths
        else normalize_float_list_from_config(sinkhorn_search_cfg.get("search_identity_strengths"))
    )
    sinkhorn_search_patience = (
        int(args.sinkhorn_search_patience)
        if int(args.sinkhorn_search_patience) > 0
        else int(sinkhorn_search_cfg.get("search_patience", 0))
    )
    sinkhorn_search_min_delta = (
        float(args.sinkhorn_search_min_delta)
        if abs(float(args.sinkhorn_search_min_delta) - 1e-6) > 1e-12
        else float(sinkhorn_search_cfg.get("search_min_delta", 1e-6))
    )
    sinkhorn_perm_scale_search_steps = (
        parse_int_list(args.sinkhorn_perm_scale_search_steps)
        if args.sinkhorn_perm_scale_search_steps
        else normalize_int_list_from_config(sinkhorn_perm_scale_search_cfg.get("search_steps"))
    )
    sinkhorn_perm_scale_search_lrs = (
        parse_float_list(args.sinkhorn_perm_scale_search_lrs)
        if args.sinkhorn_perm_scale_search_lrs
        else normalize_float_list_from_config(sinkhorn_perm_scale_search_cfg.get("search_lrs"))
    )
    sinkhorn_perm_scale_search_taus = (
        parse_float_list(args.sinkhorn_perm_scale_search_taus)
        if args.sinkhorn_perm_scale_search_taus
        else normalize_float_list_from_config(sinkhorn_perm_scale_search_cfg.get("search_taus"))
    )
    sinkhorn_perm_scale_search_identity_strengths = (
        parse_float_list(args.sinkhorn_perm_scale_search_identity_strengths)
        if args.sinkhorn_perm_scale_search_identity_strengths
        else normalize_float_list_from_config(sinkhorn_perm_scale_search_cfg.get("search_identity_strengths"))
    )
    sinkhorn_perm_scale_search_regs = (
        parse_float_list(args.sinkhorn_perm_scale_search_regs)
        if args.sinkhorn_perm_scale_search_regs
        else normalize_float_list_from_config(sinkhorn_perm_scale_search_cfg.get("search_regs"))
    )
    sinkhorn_perm_scale_target_epsilon = (
        float(args.sinkhorn_perm_scale_target_epsilon)
        if args.sinkhorn_perm_scale_target_epsilon is not None
        else (
            None
            if sinkhorn_perm_scale_search_cfg.get("target_epsilon") is None
            else float(sinkhorn_perm_scale_search_cfg.get("target_epsilon"))
        )
    )
    sinkhorn_perm_scale_search_patience = (
        int(args.sinkhorn_perm_scale_search_patience)
        if int(args.sinkhorn_perm_scale_search_patience) > 0
        else int(sinkhorn_perm_scale_search_cfg.get("search_patience", 0))
    )
    sinkhorn_perm_scale_search_min_delta = (
        float(args.sinkhorn_perm_scale_search_min_delta)
        if abs(float(args.sinkhorn_perm_scale_search_min_delta) - 1e-6) > 1e-12
        else float(sinkhorn_perm_scale_search_cfg.get("search_min_delta", 1e-6))
    )

    print(f"sinkhorn_search_steps: {sinkhorn_search_steps if sinkhorn_search_steps else '<disabled>'}")
    print(f"sinkhorn_search_lrs: {sinkhorn_search_lrs if sinkhorn_search_lrs else '<disabled>'}")
    print(f"sinkhorn_search_taus: {sinkhorn_search_taus if sinkhorn_search_taus else '<disabled>'}")
    print(
        "sinkhorn_search_identity_strengths: "
        f"{sinkhorn_search_identity_strengths if sinkhorn_search_identity_strengths else '<disabled>'}"
    )
    print("sinkhorn_target_epsilon: <best exact permutation loss barrier per pair>")
    print(f"sinkhorn_search_patience: {sinkhorn_search_patience}")
    print(f"sinkhorn_search_min_delta: {sinkhorn_search_min_delta}")
    print(
        f"sinkhorn_perm_scale_target_epsilon: "
        f"{sinkhorn_perm_scale_target_epsilon if sinkhorn_perm_scale_target_epsilon is not None else '<disabled>'}"
    )
    print(
        f"sinkhorn_perm_scale_search_steps: "
        f"{sinkhorn_perm_scale_search_steps if sinkhorn_perm_scale_search_steps else '<disabled>'}"
    )
    print(
        f"sinkhorn_perm_scale_search_lrs: "
        f"{sinkhorn_perm_scale_search_lrs if sinkhorn_perm_scale_search_lrs else '<disabled>'}"
    )
    print(
        f"sinkhorn_perm_scale_search_taus: "
        f"{sinkhorn_perm_scale_search_taus if sinkhorn_perm_scale_search_taus else '<disabled>'}"
    )
    print(
        "sinkhorn_perm_scale_search_identity_strengths: "
        f"{sinkhorn_perm_scale_search_identity_strengths if sinkhorn_perm_scale_search_identity_strengths else '<disabled>'}"
    )
    print(
        f"sinkhorn_perm_scale_search_regs: "
        f"{sinkhorn_perm_scale_search_regs if sinkhorn_perm_scale_search_regs else '<disabled>'}"
    )
    print(f"sinkhorn_perm_scale_search_patience: {sinkhorn_perm_scale_search_patience}")
    print(f"sinkhorn_perm_scale_search_min_delta: {sinkhorn_perm_scale_search_min_delta}")
    print(f"perm_scale_target_epsilon: {perm_scale_target_epsilon if perm_scale_target_epsilon is not None else '<disabled>'}")
    print(f"perm_scale_search_steps: {perm_scale_search_steps if perm_scale_search_steps else '<disabled>'}")
    print(f"perm_scale_search_lrs: {perm_scale_search_lrs if perm_scale_search_lrs else '<disabled>'}")
    print(f"perm_scale_search_regs: {perm_scale_search_regs if perm_scale_search_regs else '<disabled>'}")
    print(f"perm_scale_search_patience: {perm_scale_search_patience}")
    print(f"perm_scale_search_min_delta: {perm_scale_search_min_delta}")

    pair_results: list[dict[str, Any]] = []
    for pair_index, (seed_a, seed_b) in enumerate(pairs, start=1):
        print(f"[pair {pair_index:02d}/{len(pairs):02d}] {seed_a}-{seed_b}")
        model_a = models_by_seed[seed_a]
        model_b = models_by_seed[seed_b]

        no_alignment_metrics = compute_linear_path(model_a, model_b, num_points=args.curve_eval_points)
        perm_model, best_perm, perm_results = align_models_exhaustive(
            model_a,
            model_b,
            num_points=args.curve_eval_points,
            verbose=args.verbose,
        )
        perm_metrics = compute_linear_path(model_a, perm_model, num_points=args.curve_eval_points)
        sinkhorn_perm_search = run_sinkhorn_search(
            model_a,
            model_b,
            eval_points=args.curve_eval_points,
            base_steps=args.sinkhorn_opt_steps,
            base_lr=args.sinkhorn_opt_lr,
            base_t_points=args.sinkhorn_opt_t_points,
            base_tau=args.sinkhorn_tau,
            base_sinkhorn_iters=args.sinkhorn_iters,
            base_identity_strength=args.sinkhorn_identity_strength,
            base_patience=args.sinkhorn_patience,
            base_min_delta=args.sinkhorn_min_delta,
            barrier_epsilon=float(perm_metrics["loss_barrier"]),
            search_steps=sinkhorn_search_steps,
            search_lrs=sinkhorn_search_lrs,
            search_taus=sinkhorn_search_taus,
            search_identity_strengths=sinkhorn_search_identity_strengths,
            search_patience=sinkhorn_search_patience,
            search_min_delta=sinkhorn_search_min_delta,
            verbose=args.verbose,
        )
        sinkhorn_perm_selected = sinkhorn_perm_search["selected"]
        sinkhorn_perm_result = sinkhorn_perm_selected["result"]
        sinkhorn_perm_model = sinkhorn_perm_result["model"]
        sinkhorn_perm_metrics = sinkhorn_perm_selected["metrics"]
        perm_scale_search = run_perm_scale_search(
            model_a,
            perm_model,
            eval_points=args.curve_eval_points,
            base_steps=args.scale_opt_steps,
            base_lr=args.scale_opt_lr,
            base_t_points=args.scale_opt_t_points,
            base_scale_reg=args.scale_reg,
            base_patience=args.scale_patience,
            base_min_delta=args.scale_min_delta,
            barrier_epsilon=perm_scale_target_epsilon,
            search_steps=perm_scale_search_steps,
            search_lrs=perm_scale_search_lrs,
            search_regs=perm_scale_search_regs,
            search_patience=perm_scale_search_patience,
            search_min_delta=perm_scale_search_min_delta,
            verbose=args.verbose,
        )
        perm_scale_selected = perm_scale_search["selected"]
        perm_scale_result = perm_scale_selected["scale_result"]
        perm_scale_model = perm_scale_result["model"]
        perm_scale_metrics = perm_scale_selected["metrics"]
        sinkhorn_perm_scale_search = run_sinkhorn_perm_scale_search(
            model_a,
            model_b,
            eval_points=args.curve_eval_points,
            base_steps=args.sinkhorn_opt_steps,
            base_lr=args.sinkhorn_opt_lr,
            base_t_points=args.sinkhorn_opt_t_points,
            base_tau=args.sinkhorn_tau,
            base_sinkhorn_iters=args.sinkhorn_iters,
            base_identity_strength=args.sinkhorn_identity_strength,
            base_scale_reg=args.scale_reg,
            base_patience=args.sinkhorn_patience,
            base_min_delta=args.sinkhorn_min_delta,
            barrier_epsilon=sinkhorn_perm_scale_target_epsilon,
            search_steps=sinkhorn_perm_scale_search_steps,
            search_lrs=sinkhorn_perm_scale_search_lrs,
            search_taus=sinkhorn_perm_scale_search_taus,
            search_identity_strengths=sinkhorn_perm_scale_search_identity_strengths,
            search_regs=sinkhorn_perm_scale_search_regs,
            search_patience=sinkhorn_perm_scale_search_patience,
            search_min_delta=sinkhorn_perm_scale_search_min_delta,
            verbose=args.verbose,
        )
        sinkhorn_perm_scale_selected = sinkhorn_perm_scale_search["selected"]
        sinkhorn_perm_scale_result = sinkhorn_perm_scale_selected["result"]
        sinkhorn_perm_scale_model = sinkhorn_perm_scale_result["model"]
        sinkhorn_perm_scale_metrics = sinkhorn_perm_scale_selected["metrics"]

        joint_perm_scale: dict[str, Any] | None = None
        if args.run_joint_perm_scale:
            joint_perm_scale = align_models_exhaustive_with_scale(
                model_a,
                model_b,
                opt_steps=args.scale_opt_steps,
                opt_lr=args.scale_opt_lr,
                opt_t_points=args.scale_opt_t_points,
                eval_points=args.curve_eval_points,
                scale_reg=args.scale_reg,
                patience=args.scale_patience,
                min_delta=args.scale_min_delta,
                verbose=args.verbose,
            )

        pair_dir = ensure_dir(evaluations_dir / f"pair_{seed_a}_{seed_b}")
        npz_paths = {
            "no_alignment": save_pair_curve(pair_dir, "no_alignment", model_a, model_b, no_alignment_metrics),
            "best_permutation": save_pair_curve(pair_dir, "best_permutation", model_a, perm_model, perm_metrics),
            "sinkhorn_permutation": save_pair_curve(pair_dir, "sinkhorn_permutation", model_a, sinkhorn_perm_model, sinkhorn_perm_metrics),
            "perm_plus_scale": save_pair_curve(pair_dir, "perm_plus_scale", model_a, perm_scale_model, perm_scale_metrics),
            "sinkhorn_perm_plus_scale": save_pair_curve(
                pair_dir,
                "sinkhorn_perm_plus_scale",
                model_a,
                sinkhorn_perm_scale_model,
                sinkhorn_perm_scale_metrics,
            ),
        }
        if joint_perm_scale is not None:
            npz_paths["joint_perm_scale_exact"] = save_pair_curve(
                pair_dir,
                "joint_perm_scale_exact",
                model_a,
                joint_perm_scale["model"],
                joint_perm_scale["metrics"],
            )

        pair_payload = {
            "seed_a": seed_a,
            "seed_b": seed_b,
            "endpoint_a": endpoint_results[seed_a],
            "endpoint_b": endpoint_results[seed_b],
            "no_alignment": {
                "metrics": no_alignment_metrics,
            },
            "best_permutation": {
                "best_perm": best_perm,
                "perm_results": perm_results,
                "metrics": perm_metrics,
            },
            "sinkhorn_permutation": {
                "best_step": sinkhorn_perm_result["best_step"],
                "best_objective": sinkhorn_perm_result["best_objective"],
                "soft_perm": sinkhorn_perm_result["soft_perm"].tolist(),
                "soft_perm_stats": sinkhorn_perm_result["soft_perm_stats"],
                "hard_perm": sinkhorn_perm_result["hard_perm"],
                "selected_hparams": {
                    "steps": int(sinkhorn_perm_selected["steps"]),
                    "lr": float(sinkhorn_perm_selected["lr"]),
                    "t_points": int(sinkhorn_perm_selected["t_points"]),
                    "tau": float(sinkhorn_perm_selected["tau"]),
                    "sinkhorn_iters": int(sinkhorn_perm_selected["sinkhorn_iters"]),
                    "identity_strength": float(sinkhorn_perm_selected["identity_strength"]),
                    "patience": int(sinkhorn_perm_selected["patience"]),
                    "min_delta": float(sinkhorn_perm_selected["min_delta"]),
                },
                "target_epsilon": float(perm_metrics["loss_barrier"]),
                "satisfied_target_epsilon": bool(sinkhorn_perm_search["satisfied_epsilon"]),
                "search_triggered": bool(sinkhorn_perm_search["search_triggered"]),
                "search_attempts": [
                    {
                        "label": attempt["label"],
                        "steps": int(attempt["steps"]),
                        "lr": float(attempt["lr"]),
                        "t_points": int(attempt["t_points"]),
                        "tau": float(attempt["tau"]),
                        "sinkhorn_iters": int(attempt["sinkhorn_iters"]),
                        "identity_strength": float(attempt["identity_strength"]),
                        "patience": int(attempt["patience"]),
                        "min_delta": float(attempt["min_delta"]),
                        "loss_barrier": float(attempt["metrics"]["loss_barrier"]),
                        "barrier": float(attempt["metrics"]["barrier"]),
                        "mean_loss": float(np.mean(attempt["metrics"]["loss"])),
                        "min_accuracy": float(attempt["metrics"]["min_accuracy"]),
                        "best_step": int(attempt["result"]["best_step"]),
                        "best_objective": float(attempt["result"]["best_objective"]),
                        "soft_perm_stats": attempt["result"]["soft_perm_stats"],
                        "hard_perm": attempt["result"]["hard_perm"],
                    }
                    for attempt in sinkhorn_perm_search["attempts"]
                ],
                "metrics": sinkhorn_perm_metrics,
            },
            "perm_plus_scale": {
                "best_perm": best_perm,
                "best_log_scales": perm_scale_result["best_log_scales"].tolist(),
                "best_scales": perm_scale_result["best_scales"].tolist(),
                "scale_stats": perm_scale_result["scale_stats"],
                "best_step": perm_scale_result["best_step"],
                "best_objective": perm_scale_result["best_objective"],
                "selected_hparams": {
                    "steps": int(perm_scale_selected["steps"]),
                    "lr": float(perm_scale_selected["lr"]),
                    "t_points": int(perm_scale_selected["t_points"]),
                    "scale_reg": float(perm_scale_selected["scale_reg"]),
                    "patience": int(perm_scale_selected["patience"]),
                    "min_delta": float(perm_scale_selected["min_delta"]),
                },
                "target_epsilon": perm_scale_target_epsilon,
                "satisfied_target_epsilon": bool(perm_scale_search["satisfied_epsilon"]),
                "search_triggered": bool(perm_scale_search["search_triggered"]),
                "search_attempts": [
                    {
                        "label": attempt["label"],
                        "steps": int(attempt["steps"]),
                        "lr": float(attempt["lr"]),
                        "t_points": int(attempt["t_points"]),
                        "scale_reg": float(attempt["scale_reg"]),
                        "patience": int(attempt["patience"]),
                        "min_delta": float(attempt["min_delta"]),
                        "loss_barrier": float(attempt["metrics"]["loss_barrier"]),
                        "barrier": float(attempt["metrics"]["barrier"]),
                        "mean_loss": float(np.mean(attempt["metrics"]["loss"])),
                        "min_accuracy": float(attempt["metrics"]["min_accuracy"]),
                        "best_step": int(attempt["scale_result"]["best_step"]),
                        "best_objective": float(attempt["scale_result"]["best_objective"]),
                        "scale_stats": attempt["scale_result"]["scale_stats"],
                    }
                    for attempt in perm_scale_search["attempts"]
                ],
                "metrics": perm_scale_metrics,
            },
            "sinkhorn_perm_plus_scale": {
                "best_step": sinkhorn_perm_scale_result["best_step"],
                "best_objective": sinkhorn_perm_scale_result["best_objective"],
                "soft_perm": sinkhorn_perm_scale_result["soft_perm"].tolist(),
                "soft_perm_stats": sinkhorn_perm_scale_result["soft_perm_stats"],
                "hard_perm": sinkhorn_perm_scale_result["hard_perm"],
                "best_log_scales": sinkhorn_perm_scale_result["best_log_scales"].tolist(),
                "best_scales": sinkhorn_perm_scale_result["best_scales"].tolist(),
                "scale_stats": sinkhorn_perm_scale_result["scale_stats"],
                "selected_hparams": {
                    "steps": int(sinkhorn_perm_scale_selected["steps"]),
                    "lr": float(sinkhorn_perm_scale_selected["lr"]),
                    "t_points": int(sinkhorn_perm_scale_selected["t_points"]),
                    "tau": float(sinkhorn_perm_scale_selected["tau"]),
                    "sinkhorn_iters": int(sinkhorn_perm_scale_selected["sinkhorn_iters"]),
                    "identity_strength": float(sinkhorn_perm_scale_selected["identity_strength"]),
                    "scale_reg": float(sinkhorn_perm_scale_selected["scale_reg"]),
                    "patience": int(sinkhorn_perm_scale_selected["patience"]),
                    "min_delta": float(sinkhorn_perm_scale_selected["min_delta"]),
                },
                "target_epsilon": sinkhorn_perm_scale_target_epsilon,
                "satisfied_target_epsilon": bool(sinkhorn_perm_scale_search["satisfied_epsilon"]),
                "search_triggered": bool(sinkhorn_perm_scale_search["search_triggered"]),
                "search_attempts": [
                    {
                        "label": attempt["label"],
                        "steps": int(attempt["steps"]),
                        "lr": float(attempt["lr"]),
                        "t_points": int(attempt["t_points"]),
                        "tau": float(attempt["tau"]),
                        "sinkhorn_iters": int(attempt["sinkhorn_iters"]),
                        "identity_strength": float(attempt["identity_strength"]),
                        "scale_reg": float(attempt["scale_reg"]),
                        "patience": int(attempt["patience"]),
                        "min_delta": float(attempt["min_delta"]),
                        "loss_barrier": float(attempt["metrics"]["loss_barrier"]),
                        "barrier": float(attempt["metrics"]["barrier"]),
                        "mean_loss": float(np.mean(attempt["metrics"]["loss"])),
                        "min_accuracy": float(attempt["metrics"]["min_accuracy"]),
                        "best_step": int(attempt["result"]["best_step"]),
                        "best_objective": float(attempt["result"]["best_objective"]),
                        "soft_perm_stats": attempt["result"]["soft_perm_stats"],
                        "scale_stats": attempt["result"]["scale_stats"],
                        "hard_perm": attempt["result"]["hard_perm"],
                    }
                    for attempt in sinkhorn_perm_scale_search["attempts"]
                ],
                "metrics": sinkhorn_perm_scale_metrics,
            },
            "improvement_over_best_permutation": {
                "barrier_delta": float(perm_scale_metrics["barrier"] - perm_metrics["barrier"]),
                "loss_barrier_delta": float(perm_scale_metrics["loss_barrier"] - perm_metrics["loss_barrier"]),
                "mean_loss_delta": float(np.mean(perm_scale_metrics["loss"]) - np.mean(perm_metrics["loss"])),
                "min_accuracy_delta": float(perm_scale_metrics["min_accuracy"] - perm_metrics["min_accuracy"]),
            },
            "npz_paths": npz_paths,
        }
        if joint_perm_scale is not None:
            pair_payload["joint_perm_scale_exact"] = {
                "best_perm": joint_perm_scale["best_perm"],
                "metrics": joint_perm_scale["metrics"],
                "all_results": joint_perm_scale["all_results"],
            }
        pair_results.append(pair_payload)

        print(
            "  loss barriers: no_align={:.6f} perm={:.6f} sinkhorn_perm={:.6f} perm+scale={:.6f} sinkhorn_perm+scale={:.6f}".format(
                float(no_alignment_metrics["loss_barrier"]),
                float(perm_metrics["loss_barrier"]),
                float(sinkhorn_perm_metrics["loss_barrier"]),
                float(perm_scale_metrics["loss_barrier"]),
                float(sinkhorn_perm_scale_metrics["loss_barrier"]),
            )
        )
        if sinkhorn_perm_search["search_triggered"]:
            print(
                "  sinkhorn search: attempts={} satisfied_epsilon={} selected=(steps={}, lr={}, tau={}, id={})".format(
                    len(sinkhorn_perm_search["attempts"]),
                    sinkhorn_perm_search["satisfied_epsilon"],
                    sinkhorn_perm_selected["steps"],
                    sinkhorn_perm_selected["lr"],
                    sinkhorn_perm_selected["tau"],
                    sinkhorn_perm_selected["identity_strength"],
                )
            )
        if perm_scale_search["search_triggered"]:
            print(
                "  perm+scale search: attempts={} satisfied_epsilon={} selected=(steps={}, lr={}, reg={})".format(
                    len(perm_scale_search["attempts"]),
                    perm_scale_search["satisfied_epsilon"],
                    perm_scale_selected["steps"],
                    perm_scale_selected["lr"],
                    perm_scale_selected["scale_reg"],
                )
            )
        if sinkhorn_perm_scale_search["search_triggered"]:
            print(
                "  sinkhorn perm+scale search: attempts={} satisfied_epsilon={} selected=(steps={}, lr={}, tau={}, id={}, reg={})".format(
                    len(sinkhorn_perm_scale_search["attempts"]),
                    sinkhorn_perm_scale_search["satisfied_epsilon"],
                    sinkhorn_perm_scale_selected["steps"],
                    sinkhorn_perm_scale_selected["lr"],
                    sinkhorn_perm_scale_selected["tau"],
                    sinkhorn_perm_scale_selected["identity_strength"],
                    sinkhorn_perm_scale_selected["scale_reg"],
                )
            )
        if joint_perm_scale is not None:
            print(f"  joint_perm_scale loss barrier={float(joint_perm_scale['metrics']['loss_barrier']):.6f}")

        plot_pair_curves(
            pair_payload,
            pair_dir / "pair_loss_curves.png",
            metric_key="loss",
            ylabel="Loss",
            title=f"XOR Pair {seed_a}-{seed_b}: Loss Along Interpolation",
        )
        plot_pair_curves(
            pair_payload,
            pair_dir / "pair_accuracy_curves.png",
            metric_key="acc",
            ylabel="Accuracy (%)",
            title=f"XOR Pair {seed_a}-{seed_b}: Accuracy Along Interpolation",
        )
        plot_pair_bar_metrics(
            pair_payload,
            pair_dir / "pair_bar_metrics.png",
            title=f"XOR Pair {seed_a}-{seed_b}: Method Comparison",
        )

        with open(pair_dir / "pair_results.json", "w") as handle:
            json.dump(pair_payload, handle, indent=2)

    print("Step 3/4: aggregating summaries")
    comparison_tolerance = 1.0e-9
    summary: dict[str, Any] = {
        "num_pairs": len(pair_results),
        "no_alignment": summarize_method([pair["no_alignment"]["metrics"] for pair in pair_results]),
        "best_permutation": summarize_method([pair["best_permutation"]["metrics"] for pair in pair_results]),
        "sinkhorn_permutation": summarize_method([pair["sinkhorn_permutation"]["metrics"] for pair in pair_results]),
        "perm_plus_scale": summarize_method([pair["perm_plus_scale"]["metrics"] for pair in pair_results]),
        "sinkhorn_perm_plus_scale": summarize_method([pair["sinkhorn_perm_plus_scale"]["metrics"] for pair in pair_results]),
        "improvement_over_best_permutation": {
            "barrier": summarize_barriers([pair["improvement_over_best_permutation"]["barrier_delta"] for pair in pair_results]),
            "loss_barrier": summarize_barriers([pair["improvement_over_best_permutation"]["loss_barrier_delta"] for pair in pair_results]),
            "mean_loss": summarize_barriers([pair["improvement_over_best_permutation"]["mean_loss_delta"] for pair in pair_results]),
            "min_accuracy": summarize_barriers([pair["improvement_over_best_permutation"]["min_accuracy_delta"] for pair in pair_results]),
        },
        "improvement_over_sinkhorn_permutation": {
            "barrier": summarize_barriers([
                float(pair["perm_plus_scale"]["metrics"]["barrier"] - pair["sinkhorn_permutation"]["metrics"]["barrier"])
                for pair in pair_results
            ]),
            "loss_barrier": summarize_barriers([
                float(pair["perm_plus_scale"]["metrics"]["loss_barrier"] - pair["sinkhorn_permutation"]["metrics"]["loss_barrier"])
                for pair in pair_results
            ]),
            "mean_loss": summarize_barriers([
                float(np.mean(pair["perm_plus_scale"]["metrics"]["loss"]) - np.mean(pair["sinkhorn_permutation"]["metrics"]["loss"]))
                for pair in pair_results
            ]),
            "min_accuracy": summarize_barriers([
                float(pair["perm_plus_scale"]["metrics"]["min_accuracy"] - pair["sinkhorn_permutation"]["metrics"]["min_accuracy"])
                for pair in pair_results
            ]),
        },
        "improvement_over_sinkhorn_perm_plus_scale": {
            "barrier": summarize_barriers([
                float(pair["sinkhorn_perm_plus_scale"]["metrics"]["barrier"] - pair["sinkhorn_permutation"]["metrics"]["barrier"])
                for pair in pair_results
            ]),
            "loss_barrier": summarize_barriers([
                float(pair["sinkhorn_perm_plus_scale"]["metrics"]["loss_barrier"] - pair["sinkhorn_permutation"]["metrics"]["loss_barrier"])
                for pair in pair_results
            ]),
            "mean_loss": summarize_barriers([
                float(np.mean(pair["sinkhorn_perm_plus_scale"]["metrics"]["loss"]) - np.mean(pair["sinkhorn_permutation"]["metrics"]["loss"]))
                for pair in pair_results
            ]),
            "min_accuracy": summarize_barriers([
                float(pair["sinkhorn_perm_plus_scale"]["metrics"]["min_accuracy"] - pair["sinkhorn_permutation"]["metrics"]["min_accuracy"])
                for pair in pair_results
            ]),
        },
        "perm_plus_scale_search": {
            "target_epsilon": perm_scale_target_epsilon,
            "num_pairs_triggered": int(sum(bool(pair["perm_plus_scale"]["search_triggered"]) for pair in pair_results)),
            "num_pairs_satisfied_epsilon": int(sum(bool(pair["perm_plus_scale"]["satisfied_target_epsilon"]) for pair in pair_results)),
        },
        "sinkhorn_permutation_search": {
            "target_epsilon": "best_permutation_loss_barrier_per_pair",
            "num_pairs_triggered": int(sum(bool(pair["sinkhorn_permutation"]["search_triggered"]) for pair in pair_results)),
            "num_pairs_satisfied_epsilon": int(sum(bool(pair["sinkhorn_permutation"]["satisfied_target_epsilon"]) for pair in pair_results)),
        },
        "sinkhorn_perm_plus_scale_search": {
            "target_epsilon": sinkhorn_perm_scale_target_epsilon,
            "num_pairs_triggered": int(sum(bool(pair["sinkhorn_perm_plus_scale"]["search_triggered"]) for pair in pair_results)),
            "num_pairs_satisfied_epsilon": int(sum(bool(pair["sinkhorn_perm_plus_scale"]["satisfied_target_epsilon"]) for pair in pair_results)),
        },
        "pairwise_comparison_counts": {
            "comparison_tolerance": comparison_tolerance,
            "perm_plus_scale_better_than_permutation": int(sum(
                float(pair["perm_plus_scale"]["metrics"]["loss_barrier"])
                <= float(pair["best_permutation"]["metrics"]["loss_barrier"]) + comparison_tolerance
                for pair in pair_results
            )),
            "sinkhorn_perm_plus_scale_better_than_permutation": int(sum(
                float(pair["sinkhorn_perm_plus_scale"]["metrics"]["loss_barrier"])
                <= float(pair["best_permutation"]["metrics"]["loss_barrier"]) + comparison_tolerance
                for pair in pair_results
            )),
            "sinkhorn_permutation_equal_to_permutation": int(sum(
                abs(
                    float(pair["sinkhorn_permutation"]["metrics"]["loss_barrier"])
                    - float(pair["best_permutation"]["metrics"]["loss_barrier"])
                ) <= comparison_tolerance
                for pair in pair_results
            )),
            "perm_plus_scale_better_than_sinkhorn_perm_plus_scale": int(sum(
                float(pair["perm_plus_scale"]["metrics"]["loss_barrier"])
                <= float(pair["sinkhorn_perm_plus_scale"]["metrics"]["loss_barrier"]) + comparison_tolerance
                for pair in pair_results
            )),
        },
    }
    if args.run_joint_perm_scale:
        summary["joint_perm_scale_exact"] = summarize_method([pair["joint_perm_scale_exact"]["metrics"] for pair in pair_results])

    aggregates = {
        "no_alignment": aggregate_curves(pair_results, "no_alignment"),
        "best_permutation": aggregate_curves(pair_results, "best_permutation"),
        "sinkhorn_permutation": aggregate_curves(pair_results, "sinkhorn_permutation"),
        "perm_plus_scale": aggregate_curves(pair_results, "perm_plus_scale"),
        "sinkhorn_perm_plus_scale": aggregate_curves(pair_results, "sinkhorn_perm_plus_scale"),
    }
    if args.run_joint_perm_scale:
        aggregates["joint_perm_scale_exact"] = aggregate_curves(pair_results, "joint_perm_scale_exact")

    plot_aggregate_curves(
        aggregates,
        plots_dir / "aggregate_loss_curves_with_std.png",
        metric_key="loss",
        ylabel="Loss",
        title="Mean XOR Loss Along Interpolation Across Pairs",
        include_std=True,
    )
    plot_aggregate_curves(
        aggregates,
        plots_dir / "aggregate_loss_curves_no_std.png",
        metric_key="loss",
        ylabel="Loss",
        title=None,
        include_std=False,
        show_legend=False,
    )
    plot_aggregate_curves(
        aggregates,
        plots_dir / "aggregate_accuracy_curves_with_std.png",
        metric_key="acc",
        ylabel="Accuracy (%)",
        title="Mean XOR Accuracy Along Interpolation Across Pairs",
        include_std=True,
    )
    plot_aggregate_curves(
        aggregates,
        plots_dir / "aggregate_accuracy_curves_no_std.png",
        metric_key="acc",
        ylabel="Accuracy (%)",
        title="Mean XOR Accuracy Along Interpolation Across Pairs",
        include_std=False,
    )
    write_aggregate_stats_txt(aggregates, plots_dir / "aggregate_loss_curves_stats.txt", metric_key="loss")
    write_aggregate_stats_txt(aggregates, plots_dir / "aggregate_accuracy_curves_stats.txt", metric_key="acc")
    plot_aggregate_bar_metrics(
        pair_results,
        plots_dir / "aggregate_bar_metrics.png",
        title="XOR Method Comparison Averaged Across Pairs",
    )
    write_method_summary_markdown(
        summary,
        output_dir / "method_summary_table.md",
        include_joint_perm_scale=bool(args.run_joint_perm_scale),
    )
    write_pairwise_comparison_table(
        summary,
        len(pair_results),
        output_dir / "pairwise_comparisons_table.md",
    )

    print("Step 4/4: writing results")
    results = {
        "config": {
            "checkpoints_dir": str(args.checkpoints_dir.resolve()),
            "hidden_size": int(hidden_size),
            "requested_seeds": requested_seeds,
            "available_seeds": available_seeds,
            "pairs": [list(pair) for pair in pairs],
            "curve_eval_points": int(args.curve_eval_points),
            "sinkhorn_opt_steps": int(args.sinkhorn_opt_steps),
            "sinkhorn_opt_lr": float(args.sinkhorn_opt_lr),
            "sinkhorn_opt_t_points": int(args.sinkhorn_opt_t_points),
            "sinkhorn_tau": float(args.sinkhorn_tau),
            "sinkhorn_iters": int(args.sinkhorn_iters),
            "sinkhorn_identity_strength": float(args.sinkhorn_identity_strength),
            "sinkhorn_patience": int(args.sinkhorn_patience),
            "sinkhorn_min_delta": float(args.sinkhorn_min_delta),
            "sinkhorn_search_steps": sinkhorn_search_steps,
            "sinkhorn_search_lrs": sinkhorn_search_lrs,
            "sinkhorn_search_taus": sinkhorn_search_taus,
            "sinkhorn_search_identity_strengths": sinkhorn_search_identity_strengths,
            "sinkhorn_search_patience": int(sinkhorn_search_patience),
            "sinkhorn_search_min_delta": float(sinkhorn_search_min_delta),
            "sinkhorn_perm_scale_target_epsilon": sinkhorn_perm_scale_target_epsilon,
            "sinkhorn_perm_scale_search_steps": sinkhorn_perm_scale_search_steps,
            "sinkhorn_perm_scale_search_lrs": sinkhorn_perm_scale_search_lrs,
            "sinkhorn_perm_scale_search_taus": sinkhorn_perm_scale_search_taus,
            "sinkhorn_perm_scale_search_identity_strengths": sinkhorn_perm_scale_search_identity_strengths,
            "sinkhorn_perm_scale_search_regs": sinkhorn_perm_scale_search_regs,
            "sinkhorn_perm_scale_search_patience": int(sinkhorn_perm_scale_search_patience),
            "sinkhorn_perm_scale_search_min_delta": float(sinkhorn_perm_scale_search_min_delta),
            "scale_opt_steps": int(args.scale_opt_steps),
            "scale_opt_lr": float(args.scale_opt_lr),
            "scale_opt_t_points": int(args.scale_opt_t_points),
            "scale_reg": float(args.scale_reg),
            "scale_patience": int(args.scale_patience),
            "scale_min_delta": float(args.scale_min_delta),
            "perm_scale_target_epsilon": perm_scale_target_epsilon,
            "perm_scale_search_steps": perm_scale_search_steps,
            "perm_scale_search_lrs": perm_scale_search_lrs,
            "perm_scale_search_regs": perm_scale_search_regs,
            "perm_scale_search_patience": int(perm_scale_search_patience),
            "perm_scale_search_min_delta": float(perm_scale_search_min_delta),
            "run_joint_perm_scale": bool(args.run_joint_perm_scale),
        },
        "endpoint_results": endpoint_results,
        "pair_results": pair_results,
        "summary": summary,
        "aggregate_curves": aggregates,
        "artifacts": {
            "checkpoints_dir": str(args.checkpoints_dir.resolve()),
            "evaluations_dir": str(evaluations_dir),
            "plots_dir": str(plots_dir),
            "aggregate_loss_plot_with_std": str(plots_dir / "aggregate_loss_curves_with_std.png"),
            "aggregate_loss_plot_no_std": str(plots_dir / "aggregate_loss_curves_no_std.png"),
            "aggregate_loss_stats_txt": str(plots_dir / "aggregate_loss_curves_stats.txt"),
            "aggregate_accuracy_plot_with_std": str(plots_dir / "aggregate_accuracy_curves_with_std.png"),
            "aggregate_accuracy_plot_no_std": str(plots_dir / "aggregate_accuracy_curves_no_std.png"),
            "aggregate_accuracy_stats_txt": str(plots_dir / "aggregate_accuracy_curves_stats.txt"),
            "aggregate_bar_metrics_plot": str(plots_dir / "aggregate_bar_metrics.png"),
            "method_summary_table_md": str(output_dir / "method_summary_table.md"),
            "pairwise_comparisons_table_md": str(output_dir / "pairwise_comparisons_table.md"),
        },
    }
    with open(output_dir / "xor_perm_scale_results.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print("")
    print("=" * 80)
    print("XOR PERMUTATION VS SCALE SUMMARY")
    print("=" * 80)
    print(f"Pairs evaluated: {len(pair_results)}")
    print(f"Mean loss barrier no alignment:      {summary['no_alignment']['loss_barrier_mean']:.6f}")
    print(f"Mean loss barrier best permutation: {summary['best_permutation']['loss_barrier_mean']:.6f}")
    print(f"Mean loss barrier sinkhorn perm:    {summary['sinkhorn_permutation']['loss_barrier_mean']:.6f}")
    print(f"Mean loss barrier perm+scale:       {summary['perm_plus_scale']['loss_barrier_mean']:.6f}")
    print(f"Mean loss barrier sinkhorn p+s:     {summary['sinkhorn_perm_plus_scale']['loss_barrier_mean']:.6f}")
    if args.run_joint_perm_scale:
        print(f"Mean loss barrier joint perm+scale: {summary['joint_perm_scale_exact']['loss_barrier_mean']:.6f}")
    print(
        "Mean improvement (perm+scale - perm) in loss barrier: "
        f"{summary['improvement_over_best_permutation']['loss_barrier']['mean']:.6f} "
        f"+/- {summary['improvement_over_best_permutation']['loss_barrier']['std']:.6f}"
    )
    print(
        "Mean improvement (perm+scale - sinkhorn_perm) in loss barrier: "
        f"{summary['improvement_over_sinkhorn_permutation']['loss_barrier']['mean']:.6f} "
        f"+/- {summary['improvement_over_sinkhorn_permutation']['loss_barrier']['std']:.6f}"
    )
    print(
        "Mean improvement (sinkhorn_perm+scale - sinkhorn_perm) in loss barrier: "
        f"{summary['improvement_over_sinkhorn_perm_plus_scale']['loss_barrier']['mean']:.6f} "
        f"+/- {summary['improvement_over_sinkhorn_perm_plus_scale']['loss_barrier']['std']:.6f}"
    )
    print(
        "sinkhorn perm+scale epsilon target summary: "
        f"{summary['sinkhorn_perm_plus_scale_search']['num_pairs_satisfied_epsilon']}/"
        f"{len(pair_results)} pairs <= {sinkhorn_perm_scale_target_epsilon}"
    )
    print(
        "sinkhorn epsilon target summary: "
        f"{summary['sinkhorn_permutation_search']['num_pairs_satisfied_epsilon']}/"
        f"{len(pair_results)} pairs <= best exact permutation barrier"
    )
    if perm_scale_target_epsilon is not None:
        print(
            "perm+scale epsilon target summary: "
            f"{summary['perm_plus_scale_search']['num_pairs_satisfied_epsilon']}/"
            f"{len(pair_results)} pairs <= {perm_scale_target_epsilon}"
        )
    print(
        "Pairwise comparisons: "
        f"perm+scale <= perm: {summary['pairwise_comparison_counts']['perm_plus_scale_better_than_permutation']}/"
        f"{len(pair_results)}, "
        f"sinkhorn_perm+scale <= perm: {summary['pairwise_comparison_counts']['sinkhorn_perm_plus_scale_better_than_permutation']}/"
        f"{len(pair_results)}, "
        f"sinkhorn_perm == perm: {summary['pairwise_comparison_counts']['sinkhorn_permutation_equal_to_permutation']}/"
        f"{len(pair_results)}, "
        f"perm+scale <= sinkhorn_perm+scale: {summary['pairwise_comparison_counts']['perm_plus_scale_better_than_sinkhorn_perm_plus_scale']}/"
        f"{len(pair_results)}"
    )
    print(f"Results written under: {output_dir}")


if __name__ == "__main__":
    main()
