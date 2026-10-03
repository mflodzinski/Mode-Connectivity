"""Sweep Sinkhorn-rebasin settings for retained VGG/CIFAR endpoint pairs.

This is the thesis-facing entrypoint for the scale-aware alignment chapter: it
loads fixed endpoints, runs the configured sweep, and exports rebased
checkpoints, comparison tables, and summary artifacts.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict

import matplotlib
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_connectivity.alignment.permutation_pipeline import resolve_device, write_summary_files
from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.common.utils import set_global_seed
from mode_connectivity.core import data as core_data
from mode_connectivity.core.output import ensure_dir, save_json
from mode_connectivity.sinkhorn.shared import (
    build_alignment_criterion,
    build_optimizer,
    build_variant_row,
    clone_module_state_dict,
    configure_trainable_alignment_params,
    enable_fixed_hard_permutation_scale_only_mode,
    extract_scale_artifacts,
    format_scale_stats,
    load_vgg_checkpoint_model,
    maybe_load_starting_alignment,
)
from mode_connectivity.sinkhorn.sweep_utils import (
    build_output_tag,
    build_sweep_comparison_row,
    enumerate_sweep_combinations,
    print_top_runs,
)
from mode_connectivity.external.sinkhorn_rebasin import import_vgg_rebasin_components


def evaluate_interp_grid_loss(
    model_left: torch.nn.Module,
    model_right: torch.nn.Module,
    loader,
    *,
    alpha_grid: list[float],
    lerp,
    eval_loss_acc,
    device: torch.device,
) -> float:
    """Average CE loss across multiple interpolation points on one loader."""

    losses: list[float] = []
    for alpha in alpha_grid:
        temporal_model = lerp(model_left, model_right, float(alpha))
        loss_value, _ = eval_loss_acc(temporal_model, loader, torch.nn.CrossEntropyLoss(), device)
        losses.append(float(loss_value))
    return float(sum(losses) / len(losses))


def run_one_alignment(
    cfg: DictConfig,
    *,
    VGGClass,
    RebasinNet,
    matching,
    MidLoss,
    RndLoss,
    DistL1Loss,
    DistL2Loss,
    eval_loss_acc,
    lerp,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
) -> dict[str, Any]:
    output_root = ensure_dir(Path(cfg.output_root))
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml")

    vgg_name = str(cfg.vgg_name)
    dataset_name = str(cfg.get("dataset_name", "CIFAR10")).upper()
    starting_alignment_artifact = cfg.get("starting_alignment_artifact", None)
    resolved_alignment_artifact = None
    if starting_alignment_artifact not in (None, "", "null"):
        resolved_alignment_artifact = str(Path(to_absolute_path(str(starting_alignment_artifact))))

    model_a = load_vgg_checkpoint_model(
        Path(cfg.model_a_checkpoint),
        VGGClass,
        vgg_name=vgg_name,
        image_size=int(cfg.image_size),
        device=device,
    )
    model_b = load_vgg_checkpoint_model(
        Path(cfg.model_b_checkpoint),
        VGGClass,
        vgg_name=vgg_name,
        image_size=int(cfg.image_size),
        device=device,
    )

    loss_a_val, acc_a_val = eval_loss_acc(model_a, val_loader, torch.nn.CrossEntropyLoss(), device)
    loss_b_val, acc_b_val = eval_loss_acc(model_b, val_loader, torch.nn.CrossEntropyLoss(), device)
    loss_a, acc_a = eval_loss_acc(model_a, test_loader, torch.nn.CrossEntropyLoss(), device)
    loss_b, acc_b = eval_loss_acc(model_b, test_loader, torch.nn.CrossEntropyLoss(), device)

    pi_model_a = RebasinNet(
        model_a,
        input_shape=(1, 3, int(cfg.image_size), int(cfg.image_size)),
        l=float(cfg.sinkhorn_l),
        tau=float(cfg.tau),
        n_iter=int(cfg.sinkhorn_iters),
        scale_invariant=bool(cfg.get("scale_invariant", False)),
        lambda_scale=float(cfg.get("lambda_scale", 1e-4)),
    )
    pi_model_a.to(device)
    if bool(cfg.identity_init):
        pi_model_a.identity_init()
    maybe_load_starting_alignment(
        pi_model_a,
        artifact_path=resolved_alignment_artifact,
        permutation_kind=str(cfg.get("starting_permutation_kind", "hard")),
        scale_invariant=bool(cfg.get("scale_invariant", False)),
    )

    finetune_mode = str(cfg.get("finetune_mode", "joint")).lower()
    if finetune_mode == "scale_only_fixed_hard":
        enable_fixed_hard_permutation_scale_only_mode(pi_model_a, matching=matching)
    configure_trainable_alignment_params(pi_model_a, finetune_mode=finetune_mode)

    loss_name = str(cfg.loss_name)
    criterion = build_alignment_criterion(loss_name, model_b, MidLoss, RndLoss, DistL1Loss, DistL2Loss)
    optimizer = build_optimizer(pi_model_a, learning_rate=float(cfg.alignment_lr))
    validation_alpha_grid = [float(alpha) for alpha in cfg.get("validation_alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]

    print("")
    print("=" * 80)
    print(f"SINKHORN {vgg_name} {dataset_name} ALIGNMENT")
    print("=" * 80)
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"model_a_checkpoint: {cfg.model_a_checkpoint}")
    print(f"model_b_checkpoint: {cfg.model_b_checkpoint}")
    print(f"output_root: {output_root}")
    print(f"loss_name: {cfg.loss_name}")
    print(f"tau: {cfg.tau}")
    print(f"alignment_lr: {cfg.alignment_lr}")
    print(f"sinkhorn_l: {cfg.sinkhorn_l}")
    print(f"sinkhorn_iters: {cfg.sinkhorn_iters}")
    print(f"scale_invariant: {bool(cfg.get('scale_invariant', False))}")
    print(f"lambda_scale: {float(cfg.get('lambda_scale', 1e-4))}")
    print(f"finetune_mode: {finetune_mode}")
    print(f"val_fraction: {float(cfg.val_fraction)}")
    print(f"split_seed: {int(cfg.split_seed)}")
    print(f"best_eval_interval: {int(cfg.get('best_eval_interval', 5))}")
    print(f"validation_alpha_grid: {validation_alpha_grid}")
    print(f"early_stopping_patience: {int(cfg.get('early_stopping_patience', 0))}")
    print(f"early_stopping_min_delta: {float(cfg.get('early_stopping_min_delta', 0.0))}")
    print(f"starting_alignment_artifact: {resolved_alignment_artifact}")
    print(f"starting_permutation_kind: {cfg.get('starting_permutation_kind', 'hard')}")
    print(f"device: {device}")
    print("")

    alignment_history: list[dict[str, float | int]] = []
    best_eval_interval = int(cfg.get("best_eval_interval", 5))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    best_alignment_iteration: int | None = None
    best_alignment_score: float | None = None
    best_alignment_state: dict[str, torch.Tensor] | None = None
    no_improve_evals = 0
    early_stopped = False
    stop_iteration: int | None = None

    t1 = time()
    for iteration in range(int(cfg.alignment_iterations)):
        pi_model_a.train()
        cumulative_train_loss = 0.0
        total_train = 0
        if loss_name in {"random", "midpoint"}:
            for x, y in train_loader:
                rebased_model = pi_model_a()
                loss_training = criterion(rebased_model, x.to(device), y.to(device))
                loss_training = loss_training + pi_model_a.scale_regularizer()
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()
                cumulative_train_loss += loss_training.item() * x.shape[0]
                total_train += x.shape[0]
            cumulative_train_loss /= total_train
        else:
            rebased_model = pi_model_a()
            loss_training = criterion(rebased_model)
            loss_training = loss_training + pi_model_a.scale_regularizer()
            optimizer.zero_grad()
            loss_training.backward()
            optimizer.step()
            cumulative_train_loss = float(loss_training.item())

        pi_model_a.eval()
        if loss_name in {"random", "midpoint"}:
            if (iteration + 1) % best_eval_interval == 0 or iteration + 1 == int(cfg.alignment_iterations):
                rebased_model = deepcopy(pi_model_a())
                rebased_model.eval()
                cumulative_val_loss = evaluate_interp_grid_loss(
                    rebased_model,
                    model_b,
                    val_loader,
                    alpha_grid=validation_alpha_grid,
                    lerp=lerp,
                    eval_loss_acc=eval_loss_acc,
                    device=device,
                )
            else:
                cumulative_val_loss = float("nan")
        else:
            rebased_model = pi_model_a()
            cumulative_val_loss = float((criterion(rebased_model) + pi_model_a.scale_regularizer()).item())

        alignment_history.append({"iteration": iteration, "train_loss": float(cumulative_train_loss), "val_loss": float(cumulative_val_loss)})

        should_track_best = (iteration + 1) % best_eval_interval == 0 or iteration + 1 == int(cfg.alignment_iterations)
        if should_track_best and not torch.isnan(torch.tensor(cumulative_val_loss)):
            improved = best_alignment_score is None or cumulative_val_loss < (best_alignment_score - early_stopping_min_delta)
            if improved:
                best_alignment_score = float(cumulative_val_loss)
                best_alignment_iteration = iteration
                best_alignment_state = clone_module_state_dict(pi_model_a)
                no_improve_evals = 0
                best_msg = "[sinkhorn_align] new_best iter={:03d} val_loss={:.4f}".format(iteration + 1, cumulative_val_loss)
                if bool(cfg.get("scale_invariant", False)) and hasattr(pi_model_a, "scale_stats"):
                    best_msg = f"{best_msg} {format_scale_stats(pi_model_a.scale_stats())}"
                print(best_msg)
            else:
                no_improve_evals += 1

        if iteration == 0 or (iteration + 1) % int(cfg.log_interval) == 0 or iteration + 1 == int(cfg.alignment_iterations):
            if cumulative_val_loss == cumulative_val_loss:
                iter_msg = "[sinkhorn_align] iter={:03d} train_loss={:.4f} val_loss={:.4f}".format(
                    iteration + 1,
                    cumulative_train_loss,
                    cumulative_val_loss,
                )
            else:
                iter_msg = "[sinkhorn_align] iter={:03d} train_loss={:.4f} val_loss=<skipped>".format(
                    iteration + 1,
                    cumulative_train_loss,
                )
            if bool(cfg.get("scale_invariant", False)) and hasattr(pi_model_a, "scale_stats"):
                iter_msg = f"{iter_msg} {format_scale_stats(pi_model_a.scale_stats())}"
            print(iter_msg)

        if early_stopping_patience > 0 and no_improve_evals >= early_stopping_patience:
            stop_iteration = iteration
            early_stopped = True
            print(
                "[sinkhorn_align] early_stop iter={:03d} best_iter={:03d} best_val_loss={:.4f} "
                "no_improve_evals={:d} patience={:d}".format(
                    iteration + 1,
                    int(best_alignment_iteration) + 1 if best_alignment_iteration is not None else iteration + 1,
                    float(best_alignment_score) if best_alignment_score is not None else float("nan"),
                    no_improve_evals,
                    early_stopping_patience,
                )
            )
            break

    if stop_iteration is None:
        stop_iteration = int(cfg.alignment_iterations) - 1

    print("Elapsed time {:1.3f} secs".format(time() - t1))
    save_json(alignment_history, output_root / "alignment_history.json", indent=2)

    if best_alignment_state is not None:
        pi_model_a.load_state_dict(best_alignment_state)
        print(
            "[sinkhorn_align] restored_best iter={:03d} val_loss={:.4f}".format(
                int(best_alignment_iteration) + 1,
                float(best_alignment_score),
            )
        )

    if hasattr(pi_model_a, "update_batchnorm"):
        pi_model_a.update_batchnorm(model_a)
    pi_model_a.eval()
    rebased_model = deepcopy(pi_model_a())
    rebased_model.eval()

    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in rebased_model.state_dict().items()},
            "metadata": {"architecture": vgg_name, "dataset": dataset_name, "method": "original_sinkhorn_rebasin"},
        },
        output_root / "rebased_model.pt",
    )

    raw_permutation_parameters = [parameter.detach().cpu().clone() for parameter in pi_model_a.p if parameter is not None]
    hard_permutation_matrices = [matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu() for parameter in pi_model_a.p if parameter is not None]
    scale_artifacts = extract_scale_artifacts(pi_model_a)
    torch.save(
        {
            "raw_parameters": raw_permutation_parameters,
            "hard_permutations": hard_permutation_matrices,
            **scale_artifacts,
            "best_alignment_iteration": best_alignment_iteration,
            "best_alignment_score": best_alignment_score,
            "best_eval_interval": best_eval_interval,
            "validation_alpha_grid": validation_alpha_grid,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopped": early_stopped,
            "stop_iteration": stop_iteration,
            "scale_invariant": bool(cfg.get("scale_invariant", False)),
            "lambda_scale": float(cfg.get("lambda_scale", 1e-4)),
            "finetune_mode": finetune_mode,
            "scale_stats": pi_model_a.scale_stats() if hasattr(pi_model_a, "scale_stats") else None,
            "starting_alignment_artifact": resolved_alignment_artifact,
            "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        output_root / "alignment_artifacts.pt",
    )

    lambdas = torch.linspace(0, 1, int(cfg.num_eval_points))
    val_costs_naive: list[float] = []
    val_costs_lmc: list[float] = []
    val_acc_naive: list[float] = []
    val_acc_lmc: list[float] = []
    costs_naive: list[float] = []
    costs_lmc: list[float] = []
    acc_naive: list[float] = []
    acc_lmc: list[float] = []

    print("\nComputing interpolation for validation/test summaries")
    for i in tqdm(range(lambdas.shape[0])):
        lam = lambdas[i]
        temporal_model = lerp(rebased_model, model_b, lam)
        val_loss_lmc, val_acc_l = eval_loss_acc(temporal_model, val_loader, torch.nn.CrossEntropyLoss(), device)
        loss_lmc, acc_l = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        val_costs_lmc.append(float(val_loss_lmc))
        val_acc_lmc.append(float(val_acc_l) * 100.0)
        costs_lmc.append(float(loss_lmc))
        acc_lmc.append(float(acc_l) * 100.0)

        temporal_model = lerp(model_a, model_b, lam)
        val_loss_n, val_acc_n = eval_loss_acc(temporal_model, val_loader, torch.nn.CrossEntropyLoss(), device)
        loss_n, acc_n = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        val_costs_naive.append(float(val_loss_n))
        val_acc_naive.append(float(val_acc_n) * 100.0)
        costs_naive.append(float(loss_n))
        acc_naive.append(float(acc_n) * 100.0)

    plt.figure()
    plt.plot(lambdas.tolist(), costs_naive, label="Naive")
    plt.plot(lambdas.tolist(), costs_lmc, label="Sinkhorn Re-basin")
    plt.title("Loss")
    plt.xticks([0, 1], ["ModelA", "ModelB"])
    plt.legend()
    plt.savefig(output_root / "lmc_cnn_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(lambdas.tolist(), acc_naive, label="Naive")
    plt.plot(lambdas.tolist(), acc_lmc, label="Sinkhorn Re-basin")
    plt.title("Accuracy")
    plt.xticks([0, 1], ["ModelA", "ModelB"])
    plt.legend()
    plt.savefig(output_root / "lmc_cnn_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    comparison_rows = [
        build_variant_row(
            variant_key="no_alignment",
            display_name="Naive",
            val_costs=val_costs_naive,
            val_accs=val_acc_naive,
            test_costs=costs_naive,
            test_accs=acc_naive,
            endpoint_a_val_loss=float(loss_a_val),
            endpoint_b_val_loss=float(loss_b_val),
            endpoint_a_test_loss=float(loss_a),
            endpoint_b_test_loss=float(loss_b),
            endpoint_a_val_acc=float(acc_a_val) * 100.0,
            endpoint_b_val_acc=float(acc_b_val) * 100.0,
            endpoint_a_test_acc=float(acc_a) * 100.0,
            endpoint_b_test_acc=float(acc_b) * 100.0,
        ),
        build_variant_row(
            variant_key="original_sinkhorn_lmc",
            display_name="Sinkhorn Re-basin",
            val_costs=val_costs_lmc,
            val_accs=val_acc_lmc,
            test_costs=costs_lmc,
            test_accs=acc_lmc,
            endpoint_a_val_loss=float(loss_a_val),
            endpoint_b_val_loss=float(loss_b_val),
            endpoint_a_test_loss=float(loss_a),
            endpoint_b_test_loss=float(loss_b),
            endpoint_a_val_acc=float(acc_a_val) * 100.0,
            endpoint_b_val_acc=float(acc_b_val) * 100.0,
            endpoint_a_test_acc=float(acc_a) * 100.0,
            endpoint_b_test_acc=float(acc_b) * 100.0,
        ),
    ]
    save_json(comparison_rows, output_root / "comparison.json", indent=2)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "model_a_checkpoint": str(cfg.model_a_checkpoint),
        "model_b_checkpoint": str(cfg.model_b_checkpoint),
        "model_a_val_loss": float(loss_a_val),
        "model_a_val_acc": float(acc_a_val) * 100.0,
        "model_b_val_loss": float(loss_b_val),
        "model_b_val_acc": float(acc_b_val) * 100.0,
        "model_a_test_loss": float(loss_a),
        "model_a_test_acc": float(acc_a) * 100.0,
        "model_b_test_loss": float(loss_b),
        "model_b_test_acc": float(acc_b) * 100.0,
        "best_alignment_iteration": best_alignment_iteration,
        "best_alignment_score": best_alignment_score,
        "best_eval_interval": best_eval_interval,
        "validation_alpha_grid": validation_alpha_grid,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopped": early_stopped,
        "stop_iteration": stop_iteration,
        "scale_invariant": bool(cfg.get("scale_invariant", False)),
        "lambda_scale": float(cfg.get("lambda_scale", 1e-4)),
        "finetune_mode": finetune_mode,
        "scale_stats": pi_model_a.scale_stats() if hasattr(pi_model_a, "scale_stats") else None,
        "layer_scale_stats": scale_artifacts["layer_scale_stats"],
        "starting_alignment_artifact": resolved_alignment_artifact,
        "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, output_root / "metadata.json", indent=2)
    return metadata


def run_alignment_sweep_all(cfg: DictConfig) -> None:
    combos = enumerate_sweep_combinations(cfg.sweep)
    total_runs = len(combos)
    start_index = int(cfg.get("start_index", 0))
    end_index_cfg = cfg.get("end_index", None)
    end_index = total_runs - 1 if end_index_cfg is None else int(end_index_cfg)
    continue_on_error = bool(cfg.get("continue_on_error", False))

    if start_index < 0 or start_index >= total_runs:
        raise ValueError(f"start_index={start_index} is out of range for {total_runs} runs.")
    if end_index < start_index or end_index >= total_runs:
        raise ValueError(f"end_index={end_index} is invalid for start_index={start_index} and total_runs={total_runs}.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    base_output_root = ensure_dir(Path(to_absolute_path(str(cfg.base_output_root))))

    (
        VGGClass,
        RebasinNet,
        matching,
        _dnn_data,
        DistL1Loss,
        DistL2Loss,
        MidLoss,
        RndLoss,
        eval_loss_acc,
        lerp,
    ) = import_vgg_rebasin_components()

    train_loader, val_loader, test_loader = core_data.build_cifar10_vgg_noaug_train_val_test_loaders(
        data_path=to_absolute_path(str(cfg.data_path)),
        batch_size=int(cfg.batch_size),
        val_fraction=float(cfg.val_fraction),
        split_seed=int(cfg.split_seed),
        train_seed=int(cfg.seed),
    )

    sweep_summary: Dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "base_output_root": str(base_output_root),
        "model_a_checkpoint": str(to_absolute_path(str(cfg.model_a_checkpoint))),
        "model_b_checkpoint": str(to_absolute_path(str(cfg.model_b_checkpoint))),
        "total_configured_runs": total_runs,
        "start_index": start_index,
        "end_index": end_index,
        "continue_on_error": continue_on_error,
        "runs": [],
    }

    print("=" * 80)
    print(f"SINKHORN {cfg.vgg_name} CIFAR10 ALIGNMENT SWEEP")
    print("=" * 80)
    print(f"total configured runs: {total_runs}")
    print(f"selected range: {start_index}..{end_index}")
    print(f"base_output_root: {base_output_root}")
    print(f"model_a_checkpoint: {to_absolute_path(str(cfg.model_a_checkpoint))}")
    print(f"model_b_checkpoint: {to_absolute_path(str(cfg.model_b_checkpoint))}")
    print("")

    for task_id in range(start_index, end_index + 1):
        combo = combos[task_id]
        output_tag = build_output_tag(
            combo,
            finetune_mode=str(cfg.get("finetune_mode", "joint")).lower(),
            starting_alignment_artifact=cfg.get("starting_alignment_artifact", None),
        )
        output_root = str(base_output_root / output_tag)
        run_cfg = OmegaConf.create(
            {
                "experiment_name": f"{cfg.experiment_name}_{output_tag}",
                "vgg_name": str(cfg.vgg_name),
                "model_a_checkpoint": to_absolute_path(str(cfg.model_a_checkpoint)),
                "model_b_checkpoint": to_absolute_path(str(cfg.model_b_checkpoint)),
                "output_root": output_root,
                "data_path": to_absolute_path(str(cfg.data_path)),
                "image_size": int(cfg.image_size),
                "val_fraction": float(cfg.val_fraction),
                "split_seed": int(cfg.split_seed),
                "alignment_iterations": int(combo["alignment_iterations"]),
                "alignment_lr": float(combo["lr"]),
                "loss_name": str(combo["loss_name"]),
                "tau": float(combo["tau"]),
                "sinkhorn_iters": int(cfg.sinkhorn_iters),
                "sinkhorn_l": float(combo["sinkhorn_l"]),
                "identity_init": bool(cfg.identity_init),
                "scale_invariant": bool(cfg.get("scale_invariant", False)),
                "lambda_scale": float(combo["lambda_scale"]) if "lambda_scale" in combo else float(cfg.get("lambda_scale", 1e-4)),
                "best_eval_interval": int(cfg.best_eval_interval),
                "validation_alpha_grid": [float(alpha) for alpha in cfg.validation_alpha_grid],
                "early_stopping_patience": int(cfg.get("early_stopping_patience", 0)),
                "early_stopping_min_delta": float(cfg.get("early_stopping_min_delta", 0.0)),
                "starting_alignment_artifact": cfg.get("starting_alignment_artifact", None),
                "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
                "finetune_mode": str(cfg.get("finetune_mode", "joint")),
                "num_eval_points": int(cfg.num_eval_points),
                "log_interval": int(cfg.log_interval),
            }
        )

        print("-" * 80)
        print(f"run {task_id + 1}/{total_runs}")
        print(f"task_id: {task_id}")
        print(f"combo: {combo}")
        print(f"output_root: {output_root}")
        print("")

        run_record: Dict[str, Any] = {"task_id": task_id, "output_tag": output_tag, "output_root": output_root, "combo": combo}
        try:
            metadata = run_one_alignment(
                run_cfg,
                VGGClass=VGGClass,
                RebasinNet=RebasinNet,
                matching=matching,
                MidLoss=MidLoss,
                RndLoss=RndLoss,
                DistL1Loss=DistL1Loss,
                DistL2Loss=DistL2Loss,
                eval_loss_acc=eval_loss_acc,
                lerp=lerp,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            run_record["status"] = "completed"
            run_record["run_metadata"] = metadata
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = repr(exc)
            sweep_summary["runs"].append(run_record)
            save_json(sweep_summary, base_output_root / "sweep_summary.json", indent=2)
            if not continue_on_error:
                raise
            print(f"run failed and continue_on_error=true: {exc}")
            continue

        sweep_summary["runs"].append(run_record)
        save_json(sweep_summary, base_output_root / "sweep_summary.json", indent=2)

    completed_runs = [run for run in sweep_summary["runs"] if run.get("status") == "completed"]
    comparison_rows = [build_sweep_comparison_row(run) for run in completed_runs]
    save_json(comparison_rows, base_output_root / "sweep_comparison.json", indent=2)
    write_summary_files(str(base_output_root / "sweep_comparison"), comparison_rows)
    print_top_runs(comparison_rows)


def main() -> None:
    cfg = compose_experiment_config(
        default_config_name="sinkhorn/runs/vgg11_cifar_perm_only",
        caller_file=__file__,
    )
    run_alignment_sweep_all(cfg)


if __name__ == "__main__":
    main()
