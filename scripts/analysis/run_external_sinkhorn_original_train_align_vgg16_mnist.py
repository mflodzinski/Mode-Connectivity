"""Train two original sinkhorn-rebasin VGG16 MNIST models and align them.

This stays close to the upstream example structure in
``external/sinkhorn-rebasin/examples/main_lmc_cnn.py`` but uses the upstream
``VGG("VGG16", ...)`` model instead of ``VGG("Small", ...)``. Because that
architecture expects five pooling stages, MNIST inputs are resized to 32x32.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import hydra
import matplotlib
import numpy as np
import torch
import torchvision.transforms as transforms
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_global_seed
from scripts.analysis.run_external_sinkhorn_baseline import import_external_sinkhorn
from scripts.lib.alignment.permutation_pipeline import compute_barrier_metrics, resolve_device
from scripts.lib.core.output import ensure_dir, save_json


VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "original_sinkhorn_soft": "Original Sinkhorn LMC (soft)",
    "original_sinkhorn_hard": "Original Sinkhorn LMC (hard)",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "original_sinkhorn_soft": {"color": "#dc2626", "linestyle": "-"},
    "original_sinkhorn_hard": {"color": "#2563eb", "linestyle": "--"},
}


def format_loss_value(value: float) -> str:
    return f"{value:.6f}" if abs(value) >= 1e-4 else f"{value:.6e}"


def import_original_components():
    """Import upstream examples modules with the examples path first."""

    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    for path in (str(examples_root), str(sinkhorn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    VGG, RebasinNet, matching = import_external_sinkhorn()
    from datasets.classification import MNistDataset, SmallMNistDataset
    from rebasin.loss import MidLoss, RndLoss
    from utils import eval_loss_acc, lerp, train

    return VGG, RebasinNet, matching, MNistDataset, SmallMNistDataset, MidLoss, RndLoss, train, eval_loss_acc, lerp


def build_mnist_loaders(
    cfg: DictConfig,
    *,
    MNistDataset,
    SmallMNistDataset,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val/test loaders for resized MNIST."""

    image_size = int(cfg.image_size)
    transform = transforms.Resize((image_size, image_size))
    dataset_cls = SmallMNistDataset if bool(cfg.use_small_dataset) else MNistDataset

    train_dataset = dataset_cls(
        root=to_absolute_path(str(cfg.data_path)),
        download=True,
        train=True,
        transform=transform,
    )
    val_size = min(int(cfg.validation_size), len(train_dataset) - 1)
    train_size = len(train_dataset) - val_size
    split_generator = torch.Generator().manual_seed(int(cfg.split_seed))
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    test_dataset = MNistDataset(
        root=to_absolute_path(str(cfg.data_path)),
        download=True,
        train=False,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": int(cfg.train_batch_size),
        "num_workers": int(cfg.num_workers),
    }
    train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, test_loader


def train_one_model(
    *,
    seed: int,
    VGG,
    train_fn,
    eval_loss_acc,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    cfg: DictConfig,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train one upstream VGG16 model and evaluate it."""

    set_global_seed(seed)
    model = VGG(
        "VGG16",
        in_channels=1,
        out_features=10,
        h_in=int(cfg.image_size),
        w_in=int(cfg.image_size),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train_lr))
    criterion = torch.nn.CrossEntropyLoss()
    model = train_fn(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        int(cfg.train_epochs),
    )
    test_loss, test_acc = eval_loss_acc(model, test_loader, criterion, device)
    model.eval()
    return model, {
        "seed": seed,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def evaluate_alignment_objective(
    rebasin_net,
    criterion,
    loader,
    *,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            rebased_model = rebasin_net()
            loss = criterion(rebased_model, inputs.to(device), targets.to(device))
            total_loss += loss.item() * inputs.shape[0]
            total_examples += inputs.shape[0]

    if total_examples == 0:
        raise ValueError("Alignment loader produced zero examples.")
    return total_loss / total_examples


def save_model_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def evaluate_variant_curve(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    eval_loss_acc,
    lerp,
    test_loader,
    device: torch.device,
    num_eval_points: int,
) -> dict[str, Any]:
    ts = torch.linspace(0, 1, int(num_eval_points))
    losses = []
    accuracies = []
    criterion = torch.nn.CrossEntropyLoss()
    for lam in ts:
        temporal_model = lerp(model_a, model_b, lam)
        loss, acc = eval_loss_acc(temporal_model, test_loader, criterion, device)
        losses.append(float(loss))
        accuracies.append(float(acc) * 100.0)

    losses_np = np.asarray(losses, dtype=np.float64)
    acc_np = np.asarray(accuracies, dtype=np.float64)
    barriers = {
        key: float(value)
        for key, value in compute_barrier_metrics(
            {"te_loss": losses_np, "te_acc": acc_np, "tr_loss": losses_np, "tr_acc": acc_np}
        ).items()
    }
    return {
        "ts": ts.tolist(),
        "te_loss": losses,
        "te_acc": accuracies,
        "barriers": barriers,
    }


def plot_variant_curves(output_path: Path, variant_results: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for variant_key, results in variant_results.items():
        style = VARIANT_STYLES[variant_key]
        ts = results["ts"]
        axes[0].plot(ts, results["te_loss"], label=VARIANT_DISPLAY_NAMES[variant_key], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
        axes[1].plot(ts, results["te_acc"], label=VARIANT_DISPLAY_NAMES[variant_key], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)

    axes[0].set_title("Test Loss")
    axes[1].set_title("Test Accuracy")
    for axis in axes:
        axis.set_xlabel("Interpolation t")
        axis.grid(True, alpha=0.25)
    axes[1].legend(loc="lower center", bbox_to_anchor=(-0.1, 1.02), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_endpoints(cfg: DictConfig):
    """Train the two upstream VGG16 MNIST endpoints once."""

    (
        VGG,
        _RebasinNet,
        _matching,
        MNistDataset,
        SmallMNistDataset,
        _MidLoss,
        _RndLoss,
        train_fn,
        eval_loss_acc,
        _lerp,
    ) = import_original_components()

    runtime_device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))
    endpoints_dir = ensure_dir(output_root / "endpoints")
    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg,
        MNistDataset=MNistDataset,
        SmallMNistDataset=SmallMNistDataset,
    )

    model_a, model_a_metrics = train_one_model(
        seed=int(cfg.model_a_seed),
        VGG=VGG,
        train_fn=train_fn,
        eval_loss_acc=eval_loss_acc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=runtime_device,
        cfg=cfg,
    )
    print(f"Model A test loss {model_a_metrics['test_loss']:.4f}, test acc {model_a_metrics['test_acc'] * 100.0:.2f}")

    model_b, model_b_metrics = train_one_model(
        seed=int(cfg.model_b_seed),
        VGG=VGG,
        train_fn=train_fn,
        eval_loss_acc=eval_loss_acc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=runtime_device,
        cfg=cfg,
    )
    print(f"Model B test loss {model_b_metrics['test_loss']:.4f}, test acc {model_b_metrics['test_acc'] * 100.0:.2f}")

    model_a_path = endpoints_dir / "model_a.pt"
    model_b_path = endpoints_dir / "model_b.pt"
    save_model_checkpoint(model_a_path, model_a, model_a_metrics)
    save_model_checkpoint(model_b_path, model_b, model_b_metrics)

    return {
        "device": runtime_device,
        "model_a": model_a,
        "model_b": model_b,
        "model_a_metrics": model_a_metrics,
        "model_b_metrics": model_b_metrics,
        "model_a_path": model_a_path,
        "model_b_path": model_b_path,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "eval_loss_acc": eval_loss_acc,
        "lerp": _lerp,
    }


def run_alignment_from_models(
    cfg: DictConfig,
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    model_a_path: str | Path | None = None,
    model_b_path: str | Path | None = None,
    model_a_metrics: dict[str, Any] | None = None,
    model_b_metrics: dict[str, Any] | None = None,
    train_loader,
    val_loader,
    test_loader,
    eval_loss_acc,
    lerp,
    device: torch.device,
    include_no_alignment: bool = True,
) -> dict[str, Any]:
    """Run one original-LMC alignment sweep point using already-trained models."""

    (
        _VGG,
        RebasinNet,
        matching,
        _MNistDataset,
        _SmallMNistDataset,
        MidLoss,
        RndLoss,
        _train_fn,
        _eval_loss_acc,
        _lerp,
    ) = import_original_components()

    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))
    evaluation_dir = ensure_dir(output_root / "evaluation")

    set_global_seed(int(cfg.alignment_seed))
    rebasin_net = RebasinNet(
        model_a,
        input_shape=(1, 1, int(cfg.image_size), int(cfg.image_size)),
        l=float(cfg.sinkhorn_l),
        tau=float(cfg.tau),
        n_iter=int(cfg.sinkhorn_iters),
    )
    rebasin_net.to(runtime_device)
    if bool(cfg.identity_init):
        rebasin_net.identity_init()

    loss_name = str(cfg.loss_name).lower()
    if loss_name == "random":
        criterion = RndLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    elif loss_name == "midpoint":
        criterion = MidLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
    else:
        raise ValueError(f"Unsupported loss_name={cfg.loss_name!r}. Expected 'random' or 'midpoint'.")

    optimizer = torch.optim.Adam(rebasin_net.p.parameters(), lr=float(cfg.alignment_lr))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(cfg.alignment_epochs) + 1):
        rebasin_net.train()
        cumulative_train_loss = 0.0
        total_train = 0
        for inputs, targets in train_loader:
            rebased_model = rebasin_net()
            loss = criterion(rebased_model, inputs.to(runtime_device), targets.to(runtime_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item() * inputs.shape[0]
            total_train += inputs.shape[0]

        train_loss = cumulative_train_loss / total_train
        rebasin_net.eval()
        hard_loss = evaluate_alignment_objective(
            rebasin_net,
            criterion,
            val_loader,
            device=runtime_device,
        )
        history.append({"epoch": epoch, "train_loss": float(train_loss), "hard_loss": float(hard_loss)})

        should_log = epoch == 1 or epoch == int(cfg.alignment_epochs) or (
            int(cfg.log_interval) > 0 and epoch % int(cfg.log_interval) == 0
        )
        if should_log:
            print(
                f"[original_sinkhorn_lmc] epoch={epoch:03d} "
                f"train_loss={format_loss_value(train_loss)} "
                f"hard_loss={format_loss_value(hard_loss)}"
            )

        if hard_loss == 0.0:
            break

    if hasattr(rebasin_net, "update_batchnorm"):
        rebasin_net.update_batchnorm(model_a)

    rebasin_net.train()
    soft_model = deepcopy(rebasin_net())
    soft_model.eval()

    rebasin_net.eval()
    hard_model = deepcopy(rebasin_net())
    hard_model.eval()

    soft_model_path = output_root / "soft_aligned.pt"
    hard_model_path = output_root / "hard_aligned.pt"
    save_model_checkpoint(soft_model_path, soft_model, {"method": "original_sinkhorn_lmc_soft"})
    save_model_checkpoint(hard_model_path, hard_model, {"method": "original_sinkhorn_lmc_hard"})

    raw_permutation_parameters = [parameter.detach().cpu().clone() for parameter in rebasin_net.p if parameter is not None]
    hard_permutation_matrices = [
        matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu()
        for parameter in rebasin_net.p
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
    save_json(history, output_root / "training_history.json", indent=2)

    variant_models = {}
    if include_no_alignment:
        variant_models["no_alignment"] = (model_a, model_b)
    variant_models["original_sinkhorn_soft"] = (soft_model, model_b)
    variant_models["original_sinkhorn_hard"] = (hard_model, model_b)
    variant_rows = []
    variant_results = {}
    criterion = torch.nn.CrossEntropyLoss()
    for variant_key, (variant_a, variant_b) in variant_models.items():
        endpoint_a_loss, endpoint_a_acc = eval_loss_acc(variant_a, test_loader, criterion, device)
        endpoint_b_loss, endpoint_b_acc = eval_loss_acc(variant_b, test_loader, criterion, device)
        curve = evaluate_variant_curve(
            variant_a,
            variant_b,
            eval_loss_acc=eval_loss_acc,
            lerp=lerp,
            test_loader=test_loader,
            device=device,
            num_eval_points=int(cfg.num_eval_points),
        )
        variant_results[variant_key] = curve
        barriers = curve["barriers"]
        variant_rows.append(
            {
                "variant_key": variant_key,
                "variant_name": VARIANT_DISPLAY_NAMES[variant_key],
                "endpoint_a_test_loss": float(endpoint_a_loss),
                "endpoint_a_test_acc": float(endpoint_a_acc) * 100.0,
                "endpoint_b_test_loss": float(endpoint_b_loss),
                "endpoint_b_test_acc": float(endpoint_b_acc) * 100.0,
                "mean_test_interp_loss": float(np.mean(curve["te_loss"])),
                "raw_max_test_interp_loss": float(np.max(curve["te_loss"])),
                "test_loss_barrier_avg": barriers["test_loss_barrier_avg"],
                "test_loss_barrier_max_endpoint": barriers["test_loss_barrier_max_endpoint"],
                "min_test_acc": barriers["min_test_acc"],
                "test_acc_drop_from_endpoint_min": barriers["test_acc_drop_from_endpoint_min"],
            }
        )

    plot_variant_curves(evaluation_dir / str(cfg.plot_filename), variant_results)
    save_json(variant_rows, evaluation_dir / "comparison.json", indent=2)
    for variant_key, curve in variant_results.items():
        save_json(curve, evaluation_dir / f"{variant_key}.json", indent=2)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "model_a_path": None if model_a_path is None else str(model_a_path),
        "model_b_path": None if model_b_path is None else str(model_b_path),
        "soft_model_path": str(soft_model_path),
        "hard_model_path": str(hard_model_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_a_metrics": model_a_metrics,
        "model_b_metrics": model_b_metrics,
    }
    save_json(metadata, output_root / "metadata.json", indent=2)

    print("")
    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAIN + ALIGN COMPLETE")
    print("=" * 80)
    print(f"Model A checkpoint: {model_a_path}")
    print(f"Model B checkpoint: {model_b_path}")
    print(f"Alignment artifacts: {output_root / 'alignment_artifacts.pt'}")
    print(f"Evaluation summary: {evaluation_dir / 'comparison.json'}")
    print(f"Comparison plot: {evaluation_dir / str(cfg.plot_filename)}")

    return metadata


def run_train_and_align(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAIN + ALIGN")
    print("=" * 80)
    print(f"output_root: {to_absolute_path(str(cfg.output_root))}")
    print(f"device: {resolve_device(str(cfg.device))}")
    print(f"use_small_dataset: {bool(cfg.use_small_dataset)}")
    print(f"image_size: {int(cfg.image_size)}")
    print("")

    trained = train_endpoints(cfg)
    print(f"Model A test loss {trained['model_a_metrics']['test_loss']:.4f}, test acc {trained['model_a_metrics']['test_acc'] * 100.0:.2f}")
    print(f"Model B test loss {trained['model_b_metrics']['test_loss']:.4f}, test acc {trained['model_b_metrics']['test_acc'] * 100.0:.2f}")

    return run_alignment_from_models(
        cfg,
        model_a=deepcopy(trained["model_a"]),
        model_b=deepcopy(trained["model_b"]),
        model_a_path=trained["model_a_path"],
        model_b_path=trained["model_b_path"],
        model_a_metrics=trained["model_a_metrics"],
        model_b_metrics=trained["model_b_metrics"],
        train_loader=trained["train_loader"],
        val_loader=trained["val_loader"],
        test_loader=trained["test_loader"],
        eval_loss_acc=trained["eval_loss_acc"],
        lerp=trained["lerp"],
        device=trained["device"],
        include_no_alignment=True,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_train_align_vgg16_mnist",
)
def main(cfg: DictConfig) -> None:
    run_train_and_align(cfg)


if __name__ == "__main__":
    main()
