"""Sweep original sinkhorn re-basin alignment params for saved VGG11 CIFAR10 endpoints."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict

import hydra
import matplotlib
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline_sweep import enumerate_sweep_combinations, sanitize_value
from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import build_model, clone_module_state_dict, import_original_mnist_components
from scripts.lib.alignment.permutation_pipeline import resolve_device, write_summary_files
from scripts.lib.core.output import ensure_dir, load_json, save_json
from src.utils import set_global_seed


def build_output_tag(combo: Dict[str, Any]) -> str:
    parts = [
        f"steps{combo['alignment_iterations']}",
        f"tau{sanitize_value(combo['tau'])}",
        f"lr{sanitize_value(combo['lr'])}",
        f"l{sanitize_value(combo['sinkhorn_l'])}",
        f"loss{combo['loss_name']}",
    ]
    if "lambda_scale" in combo:
        parts.append(f"lam{sanitize_value(combo['lambda_scale'])}")
    return "_".join(parts)


def build_cifar10_loaders(cfg: DictConfig, dnn_data):
    transform_train = dnn_data.Transforms.CIFAR10.VGG.train
    transform_test = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(to_absolute_path(str(cfg.data_path))) / "cifar10"

    dataset_train = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))
    val_loader = test_loader
    return train_loader, val_loader, test_loader


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


def format_scale_stats(scale_stats: Dict[str, Any] | None) -> str:
    if not scale_stats:
        return "scale_stats=<disabled>"
    return (
        "scale[min={scale_min:.4f}, mean={scale_mean:.4f}, max={scale_max:.4f}] "
        "inv_scale[min={inv_scale_min:.4f}, mean={inv_scale_mean:.4f}, max={inv_scale_max:.4f}]"
    ).format(**scale_stats)


def extract_scale_artifacts(pi_model_a: torch.nn.Module) -> Dict[str, Any]:
    if not hasattr(pi_model_a, "u"):
        return {
            "raw_log_scales": [],
            "scales": [],
            "inv_scales": [],
            "layer_scale_stats": [],
        }

    raw_log_scales: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    inv_scales: list[torch.Tensor] = []
    layer_scale_stats: list[dict[str, float | int]] = []
    permutation_to_parameter_names: dict[int, list[str]] = {}
    if hasattr(pi_model_a, "reparamnet"):
        for parameter_name, permutation_graph_index in pi_model_a.reparamnet.map_param_index.items():
            permutation_index = pi_model_a.reparamnet.perm_dict[permutation_graph_index]
            if permutation_index is None:
                continue
            permutation_to_parameter_names.setdefault(int(permutation_index), []).append(parameter_name)
    for layer_index, log_scale in enumerate(pi_model_a.u):
        if log_scale is None:
            continue
        log_scale_cpu = log_scale.detach().cpu().clone()
        scale_cpu = torch.exp(log_scale_cpu)
        inv_scale_cpu = torch.exp(-log_scale_cpu)
        raw_log_scales.append(log_scale_cpu)
        scales.append(scale_cpu)
        inv_scales.append(inv_scale_cpu)
        layer_scale_stats.append(
            {
                "layer_index": int(layer_index),
                "parameter_names": sorted(permutation_to_parameter_names.get(int(layer_index), [])),
                "num_channels": int(scale_cpu.numel()),
                "log_scale_min": float(log_scale_cpu.min().item()),
                "log_scale_max": float(log_scale_cpu.max().item()),
                "log_scale_mean": float(log_scale_cpu.mean().item()),
                "scale_min": float(scale_cpu.min().item()),
                "scale_max": float(scale_cpu.max().item()),
                "scale_mean": float(scale_cpu.mean().item()),
                "inv_scale_min": float(inv_scale_cpu.min().item()),
                "inv_scale_max": float(inv_scale_cpu.max().item()),
                "inv_scale_mean": float(inv_scale_cpu.mean().item()),
            }
        )

    return {
        "raw_log_scales": raw_log_scales,
        "scales": scales,
        "inv_scales": inv_scales,
        "layer_scale_stats": layer_scale_stats,
    }


def maybe_load_starting_alignment(pi_model_a: torch.nn.Module, cfg: DictConfig) -> None:
    """Warm-start permutation parameters from a previous alignment artifact."""

    artifact_path = cfg.get("starting_alignment_artifact", None)
    if artifact_path in (None, "", "null"):
        return

    resolved_path = Path(to_absolute_path(str(artifact_path)))
    payload = torch.load(resolved_path, map_location="cpu")
    permutation_kind = str(cfg.get("starting_permutation_kind", "hard"))
    if permutation_kind == "hard":
        source = payload.get("hard_permutations")
    elif permutation_kind == "raw":
        source = payload.get("raw_parameters")
    else:
        raise ValueError(f"Unsupported starting_permutation_kind={permutation_kind!r}. Expected 'hard' or 'raw'.")

    if source is None:
        raise ValueError(f"Starting alignment artifact {resolved_path} does not contain {permutation_kind!r} permutation data.")

    target_params = [parameter for parameter in pi_model_a.p if parameter is not None]
    if len(source) != len(target_params):
        raise ValueError(
            f"Starting alignment artifact {resolved_path} has {len(source)} permutations but the current model expects {len(target_params)}."
        )

    with torch.no_grad():
        for target, source_value in zip(target_params, source):
            source_tensor = source_value if isinstance(source_value, torch.Tensor) else torch.as_tensor(source_value)
            target.data.copy_(source_tensor.to(device=target.device, dtype=target.dtype))


def load_model_from_checkpoint(model_path: Path, VGGClass, *, vgg_name: str, image_size: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(VGGClass, vgg_name, num_classes=10, image_size=image_size)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint payload at {model_path}; expected a raw state_dict or dict with "
            "'model_state'/'state_dict'."
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_criterion(loss_name: str, model_b: torch.nn.Module, MidLoss, RndLoss, DistL1Loss, DistL2Loss):
    if loss_name == "random":
        return RndLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    if loss_name == "midpoint":
        return MidLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    if loss_name == "dist_l1":
        return DistL1Loss(model_b)
    if loss_name == "dist_l2":
        return DistL2Loss(model_b)
    raise ValueError(f"Unsupported loss_name={loss_name!r}. Expected 'random', 'midpoint', 'dist_l1', or 'dist_l2'.")


def compute_curve_metrics(costs: list[float], accs: list[float], endpoint_a_loss: float, endpoint_b_loss: float) -> dict[str, float]:
    max_endpoint_loss = max(endpoint_a_loss, endpoint_b_loss)
    return {
        "mean_test_interp_loss": float(sum(costs) / len(costs)),
        "test_loss_barrier_avg": float((sum(costs) / len(costs)) - ((endpoint_a_loss + endpoint_b_loss) / 2.0)),
        "test_loss_barrier_max_endpoint": float(max(costs) - max_endpoint_loss),
        "min_test_acc": float(min(accs)),
        "endpoint_a_test_loss": float(endpoint_a_loss),
        "endpoint_b_test_loss": float(endpoint_b_loss),
    }


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
    vgg_name = str(cfg.vgg_name)
    model_a = load_model_from_checkpoint(Path(cfg.model_a_checkpoint), VGGClass, vgg_name=vgg_name, image_size=int(cfg.image_size), device=device)
    model_b = load_model_from_checkpoint(Path(cfg.model_b_checkpoint), VGGClass, vgg_name=vgg_name, image_size=int(cfg.image_size), device=device)

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
    maybe_load_starting_alignment(pi_model_a, cfg)

    loss_name = str(cfg.loss_name)
    criterion = build_criterion(loss_name, model_b, MidLoss, RndLoss, DistL1Loss, DistL2Loss)
    optimizer = torch.optim.AdamW(pi_model_a.parameters(), lr=float(cfg.alignment_lr))
    validation_alpha_grid = [float(alpha) for alpha in cfg.get("validation_alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]

    print("")
    print("=" * 80)
    print(f"ORIGINAL SINKHORN {vgg_name} CIFAR10 ALIGNMENT")
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
    print(f"best_eval_interval: {int(cfg.get('best_eval_interval', 5))}")
    print(f"validation_alpha_grid: {validation_alpha_grid}")
    print(f"starting_alignment_artifact: {cfg.get('starting_alignment_artifact', None)}")
    print(f"starting_permutation_kind: {cfg.get('starting_permutation_kind', 'hard')}")
    print(f"device: {device}")
    print("")

    alignment_history: list[dict[str, float | int]] = []
    t1 = time()
    best_eval_interval = int(cfg.get("best_eval_interval", 5))
    best_alignment_iteration: int | None = None
    best_alignment_score: float | None = None
    best_alignment_state: dict[str, torch.Tensor] | None = None
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
        if should_track_best and not torch.isnan(torch.tensor(cumulative_val_loss)) and (
            best_alignment_score is None or cumulative_val_loss < best_alignment_score
        ):
            best_alignment_score = float(cumulative_val_loss)
            best_alignment_iteration = iteration
            best_alignment_state = clone_module_state_dict(pi_model_a)
            best_msg = "[original_sinkhorn_align] new_best iter={:03d} val_loss={:.4f}".format(
                iteration + 1, cumulative_val_loss
            )
            if bool(cfg.get("scale_invariant", False)) and hasattr(pi_model_a, "scale_stats"):
                best_msg = f"{best_msg} {format_scale_stats(pi_model_a.scale_stats())}"
            print(best_msg)

        if iteration == 0 or (iteration + 1) % int(cfg.log_interval) == 0 or iteration + 1 == int(cfg.alignment_iterations):
            if cumulative_val_loss == cumulative_val_loss:
                iter_msg = "[original_sinkhorn_align] iter={:03d} train_loss={:.4f} val_loss={:.4f}".format(
                    iteration + 1,
                    cumulative_train_loss,
                    cumulative_val_loss,
                )
            else:
                iter_msg = "[original_sinkhorn_align] iter={:03d} train_loss={:.4f} val_loss=<skipped>".format(
                    iteration + 1,
                    cumulative_train_loss,
                )
            if bool(cfg.get("scale_invariant", False)) and hasattr(pi_model_a, "scale_stats"):
                iter_msg = f"{iter_msg} {format_scale_stats(pi_model_a.scale_stats())}"
            print(iter_msg)

    print("Elapsed time {:1.3f} secs".format(time() - t1))
    save_json(alignment_history, output_root / "alignment_history.json", indent=2)

    if best_alignment_state is not None:
        pi_model_a.load_state_dict(best_alignment_state)
        print(
            "[original_sinkhorn_align] restored_best iter={:03d} val_loss={:.4f}".format(
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
            "metadata": {"architecture": vgg_name, "dataset": "CIFAR10", "method": "original_sinkhorn_rebasin"},
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
            "scale_invariant": bool(cfg.get("scale_invariant", False)),
            "lambda_scale": float(cfg.get("lambda_scale", 1e-4)),
            "scale_stats": pi_model_a.scale_stats() if hasattr(pi_model_a, "scale_stats") else None,
            "starting_alignment_artifact": None if cfg.get("starting_alignment_artifact", None) in (None, "", "null") else str(to_absolute_path(str(cfg.starting_alignment_artifact))),
            "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        output_root / "alignment_artifacts.pt",
    )

    lambdas = torch.linspace(0, 1, int(cfg.num_eval_points))
    costs_naive: list[float] = []
    costs_lmc: list[float] = []
    acc_naive: list[float] = []
    acc_lmc: list[float] = []

    print("\nComputing interpolation for LMC visualization")
    for i in tqdm(range(lambdas.shape[0])):
        lam = lambdas[i]
        temporal_model = lerp(rebased_model, model_b, lam)
        loss_lmc, acc_l = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        costs_lmc.append(float(loss_lmc))
        acc_lmc.append(float(acc_l) * 100.0)

        temporal_model = lerp(model_a, model_b, lam)
        loss_n, acc_n = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
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
        {
            "variant_key": "no_alignment",
            "display_name": "Naive",
            "endpoint_a_test_acc": float(acc_a) * 100.0,
            "endpoint_b_test_acc": float(acc_b) * 100.0,
            **compute_curve_metrics(costs_naive, acc_naive, float(loss_a), float(loss_b)),
        },
        {
            "variant_key": "original_sinkhorn_lmc",
            "display_name": "Sinkhorn Re-basin",
            "endpoint_a_test_acc": float(acc_a) * 100.0,
            "endpoint_b_test_acc": float(acc_b) * 100.0,
            **compute_curve_metrics(costs_lmc, acc_lmc, float(loss_a), float(loss_b)),
        },
    ]
    save_json(comparison_rows, output_root / "comparison.json", indent=2)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "model_a_checkpoint": str(cfg.model_a_checkpoint),
        "model_b_checkpoint": str(cfg.model_b_checkpoint),
        "model_a_test_loss": float(loss_a),
        "model_a_test_acc": float(acc_a) * 100.0,
        "model_b_test_loss": float(loss_b),
        "model_b_test_acc": float(acc_b) * 100.0,
        "best_alignment_iteration": best_alignment_iteration,
        "best_alignment_score": best_alignment_score,
        "best_eval_interval": best_eval_interval,
        "validation_alpha_grid": validation_alpha_grid,
        "scale_invariant": bool(cfg.get("scale_invariant", False)),
        "lambda_scale": float(cfg.get("lambda_scale", 1e-4)),
        "scale_stats": pi_model_a.scale_stats() if hasattr(pi_model_a, "scale_stats") else None,
        "layer_scale_stats": scale_artifacts["layer_scale_stats"],
        "starting_alignment_artifact": None if cfg.get("starting_alignment_artifact", None) in (None, "", "null") else str(to_absolute_path(str(cfg.starting_alignment_artifact))),
        "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, output_root / "metadata.json", indent=2)
    return metadata


def load_run_comparison(output_root: str) -> Dict[str, Dict[str, Any]]:
    rows = load_json(Path(output_root) / "comparison.json")
    return {row["variant_key"]: row for row in rows}


def build_sweep_comparison_row(run_record: Dict[str, Any]) -> Dict[str, Any]:
    comparison = load_run_comparison(run_record["output_root"])
    lmc = comparison["original_sinkhorn_lmc"]
    no_alignment = comparison["no_alignment"]
    combo = run_record["combo"]
    return {
        "task_id": run_record["task_id"],
        "output_tag": run_record["output_tag"],
        "output_root": run_record["output_root"],
        "alignment_iterations": combo["alignment_iterations"],
        "loss_name": combo["loss_name"],
        "tau": combo["tau"],
        "lr": combo["lr"],
        "sinkhorn_l": combo["sinkhorn_l"],
        "lmc_mean_test_interp_loss": lmc["mean_test_interp_loss"],
        "lmc_test_loss_barrier_max_endpoint": lmc["test_loss_barrier_max_endpoint"],
        "lmc_min_test_acc": lmc["min_test_acc"],
        "no_align_mean_test_interp_loss": no_alignment["mean_test_interp_loss"],
        "no_align_test_loss_barrier_max_endpoint": no_alignment["test_loss_barrier_max_endpoint"],
        "no_align_min_test_acc": no_alignment["min_test_acc"],
        "delta_mean_test_interp_loss_vs_no_align": lmc["mean_test_interp_loss"] - no_alignment["mean_test_interp_loss"],
        "delta_test_loss_barrier_max_endpoint_vs_no_align": lmc["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"],
        "delta_min_test_acc_vs_no_align": lmc["min_test_acc"] - no_alignment["min_test_acc"],
    }


def print_top_runs(rows: list[Dict[str, Any]], *, top_k: int = 5) -> None:
    if not rows:
        print("No completed runs available for sweep comparison.")
        return
    top_barrier = sorted(rows, key=lambda row: row["lmc_test_loss_barrier_max_endpoint"])[:top_k]
    top_mean = sorted(rows, key=lambda row: row["lmc_mean_test_interp_loss"])[:top_k]
    print("")
    print("=" * 80)
    print("TOP RUNS BY LMC TEST BARRIER")
    print("=" * 80)
    for row in top_barrier:
        print(
            f"task={row['task_id']:02d} "
            f"steps={row['alignment_iterations']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"lmc_barrier={row['lmc_test_loss_barrier_max_endpoint']:.4f} "
            f"lmc_mean={row['lmc_mean_test_interp_loss']:.4f} "
            f"lmc_min_acc={row['lmc_min_test_acc']:.2f}"
        )
    print("")
    print("=" * 80)
    print("TOP RUNS BY LMC MEAN TEST LOSS")
    print("=" * 80)
    for row in top_mean:
        print(
            f"task={row['task_id']:02d} "
            f"steps={row['alignment_iterations']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"lmc_mean={row['lmc_mean_test_interp_loss']:.4f} "
            f"lmc_barrier={row['lmc_test_loss_barrier_max_endpoint']:.4f} "
            f"lmc_min_acc={row['lmc_min_test_acc']:.2f}"
        )


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

    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    base_output_root = ensure_dir(Path(to_absolute_path(str(cfg.base_output_root))))

    VGGClass, RebasinNet, matching, dnn_data, RndLoss, eval_loss_acc, lerp = import_original_mnist_components()
    from rebasin.loss import DistL1Loss, DistL2Loss, MidLoss

    train_loader, val_loader, test_loader = build_cifar10_loaders(cfg, dnn_data)

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
    print("ORIGINAL SINKHORN VGG11 CIFAR10 ALIGNMENT SWEEP")
    print("=" * 80)
    print(f"total configured runs: {total_runs}")
    print(f"selected range: {start_index}..{end_index}")
    print(f"base_output_root: {base_output_root}")
    print(f"model_a_checkpoint: {to_absolute_path(str(cfg.model_a_checkpoint))}")
    print(f"model_b_checkpoint: {to_absolute_path(str(cfg.model_b_checkpoint))}")
    print("")

    for task_id in range(start_index, end_index + 1):
        combo = combos[task_id]
        output_tag = build_output_tag(combo)
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
                "starting_alignment_artifact": cfg.get("starting_alignment_artifact", None),
                "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
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


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_vgg11_cifar10_align_sweep",
)
def main(cfg: DictConfig) -> None:
    run_alignment_sweep_all(cfg)


if __name__ == "__main__":
    main()
