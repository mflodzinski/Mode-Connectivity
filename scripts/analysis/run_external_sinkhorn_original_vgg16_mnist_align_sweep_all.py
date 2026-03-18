"""Sweep original sinkhorn re-basin alignment params for saved VGG16 MNIST endpoints."""

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
from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import (
    build_model,
    import_original_mnist_components,
)
from scripts.lib.core.output import ensure_dir, load_json, save_json
from scripts.lib.alignment.permutation_pipeline import resolve_device, write_summary_files
from src.utils import set_global_seed


def build_output_tag(combo: Dict[str, Any]) -> str:
    return "_".join(
        [
            f"steps{combo['alignment_iterations']}",
            f"tau{sanitize_value(combo['tau'])}",
            f"lr{sanitize_value(combo['lr'])}",
            f"l{sanitize_value(combo['sinkhorn_l'])}",
            f"loss{combo['loss_name']}",
        ]
    )


def build_mnist_loaders(cfg: DictConfig, dnn_data) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform_train = dnn_data.Transforms.MNIST.VGG.train
    transform_test = dnn_data.Transforms.MNIST.VGG.test
    mnist_root = Path(to_absolute_path(str(cfg.data_path))) / "mnist"

    dataset_train_source = torchvision.datasets.MNIST(
        root=mnist_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_val_source = torchvision.datasets.MNIST(
        root=mnist_root,
        train=True,
        download=True,
        transform=transform_test,
    )
    dataset_test_source = torchvision.datasets.MNIST(
        root=mnist_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_total_size = len(dataset_train_source)
    val_fraction = float(cfg.val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    val_size = int(train_total_size * val_fraction)
    train_size = train_total_size - val_size
    indices = torch.randperm(train_total_size, generator=torch.Generator().manual_seed(int(cfg.split_seed)))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    dataset_train = torch.utils.data.Subset(dataset_train_source, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val_source, val_indices)
    dataset_test = dataset_test_source

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, test_loader


def load_model_from_checkpoint(
    model_path: Path,
    VGGClass,
    *,
    vgg_name: str,
    image_size: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(VGGClass, vgg_name, num_classes=10, image_size=image_size)
    model.load_state_dict(checkpoint["model_state"])
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
    model_a = load_model_from_checkpoint(
        Path(cfg.model_a_checkpoint),
        VGGClass,
        vgg_name="VGG16",
        image_size=int(cfg.image_size),
        device=device,
    )
    model_b = load_model_from_checkpoint(
        Path(cfg.model_b_checkpoint),
        VGGClass,
        vgg_name="VGG16",
        image_size=int(cfg.image_size),
        device=device,
    )

    loss_a, acc_a = eval_loss_acc(model_a, test_loader, torch.nn.CrossEntropyLoss(), device)
    loss_b, acc_b = eval_loss_acc(model_b, test_loader, torch.nn.CrossEntropyLoss(), device)

    pi_model_a = RebasinNet(
        model_a,
        input_shape=(1, 3, int(cfg.image_size), int(cfg.image_size)),
        l=float(cfg.sinkhorn_l),
        tau=float(cfg.tau),
        n_iter=int(cfg.sinkhorn_iters),
    )
    pi_model_a.to(device)
    if bool(cfg.identity_init):
        pi_model_a.identity_init()

    loss_name = str(cfg.loss_name)
    criterion = build_criterion(loss_name, model_b, MidLoss, RndLoss, DistL1Loss, DistL2Loss)
    optimizer = torch.optim.AdamW(pi_model_a.p.parameters(), lr=float(cfg.alignment_lr))

    print("")
    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST ALIGNMENT")
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
    print(f"device: {device}")
    print("")

    alignment_history: list[dict[str, float | int]] = []
    t1 = time()
    for iteration in range(int(cfg.alignment_iterations)):
        pi_model_a.train()
        cumulative_train_loss = 0.0
        total_train = 0
        if loss_name in {"random", "midpoint"}:
            for x, y in train_loader:
                rebased_model = pi_model_a()
                loss_training = criterion(rebased_model, x.to(device), y.to(device))
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()
                cumulative_train_loss += loss_training.item() * x.shape[0]
                total_train += x.shape[0]
            cumulative_train_loss /= total_train
        else:
            rebased_model = pi_model_a()
            loss_training = criterion(rebased_model)
            optimizer.zero_grad()
            loss_training.backward()
            optimizer.step()
            cumulative_train_loss = float(loss_training.item())

        cumulative_val_loss = 0.0
        total_val = 0
        pi_model_a.eval()
        if loss_name in {"random", "midpoint"}:
            for x, y in val_loader:
                rebased_model = pi_model_a()
                loss_validation = criterion(rebased_model, x.to(device), y.to(device))
                cumulative_val_loss += loss_validation.item() * x.shape[0]
                total_val += x.shape[0]
            cumulative_val_loss /= total_val
        else:
            rebased_model = pi_model_a()
            cumulative_val_loss = float(criterion(rebased_model).item())
        alignment_history.append(
            {
                "iteration": iteration,
                "train_loss": float(cumulative_train_loss),
                "val_loss": float(cumulative_val_loss),
            }
        )

        if iteration == 0 or (iteration + 1) % int(cfg.log_interval) == 0 or iteration + 1 == int(cfg.alignment_iterations):
            print(
                "[original_sinkhorn_align] iter={:03d} train_loss={:.4f} val_loss={:.4f}".format(
                    iteration + 1, cumulative_train_loss, cumulative_val_loss
                )
            )

    print("Elapsed time {:1.3f} secs".format(time() - t1))
    save_json(alignment_history, output_root / "alignment_history.json", indent=2)

    if hasattr(pi_model_a, "update_batchnorm"):
        pi_model_a.update_batchnorm(model_a)
    pi_model_a.eval()
    rebased_model = deepcopy(pi_model_a())
    rebased_model.eval()

    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in rebased_model.state_dict().items()},
            "metadata": {"architecture": "VGG16", "method": "original_sinkhorn_rebasin"},
        },
        output_root / "rebased_model.pt",
    )
    raw_permutation_parameters = [parameter.detach().cpu().clone() for parameter in pi_model_a.p if parameter is not None]
    hard_permutation_matrices = [
        matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu()
        for parameter in pi_model_a.p
        if parameter is not None
    ]
    torch.save(
        {
            "raw_parameters": raw_permutation_parameters,
            "hard_permutations": hard_permutation_matrices,
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
        "delta_test_loss_barrier_max_endpoint_vs_no_align": (
            lmc["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"]
        ),
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
        raise ValueError(
            f"end_index={end_index} is invalid for start_index={start_index} and total_runs={total_runs}."
        )

    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    base_output_root = ensure_dir(Path(to_absolute_path(str(cfg.base_output_root))))

    VGGClass, RebasinNet, matching, dnn_data, RndLoss, eval_loss_acc, lerp = import_original_mnist_components()
    from rebasin.loss import DistL1Loss, DistL2Loss, MidLoss

    train_loader, val_loader, test_loader = build_mnist_loaders(cfg, dnn_data)

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
    print("ORIGINAL SINKHORN VGG16 MNIST ALIGNMENT SWEEP")
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

        run_record: Dict[str, Any] = {
            "task_id": task_id,
            "output_tag": output_tag,
            "output_root": output_root,
            "combo": combo,
        }
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
    config_name="external_sinkhorn_original_vgg16_mnist_align_sweep",
)
def main(cfg: DictConfig) -> None:
    run_alignment_sweep_all(cfg)


if __name__ == "__main__":
    main()
