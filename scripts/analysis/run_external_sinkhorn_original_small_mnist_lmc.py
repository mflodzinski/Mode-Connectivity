"""Run the upstream sinkhorn-rebasin VGG16-MNIST LMC workflow with saved artifacts."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import hydra
import matplotlib
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_global_seed
from scripts.analysis.run_external_sinkhorn_baseline import import_external_sinkhorn
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json

def import_original_mnist_components():
    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    dnn_root = project_root / "external" / "dnn-mode-connectivity"
    for path in (str(examples_root), str(sinkhorn_root), str(dnn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    _, RebasinNet, matching = import_external_sinkhorn()
    from models.vgg import VGG
    import data as dnn_data
    from rebasin.loss import RndLoss
    from utils import eval_loss_acc, lerp

    return VGG, RebasinNet, matching, dnn_data, RndLoss, eval_loss_acc, lerp


def build_model(VGGClass, num_classes: int, image_size: int) -> torch.nn.Module:
    """Instantiate the sinkhorn-rebasin VGG16 model."""

    return VGGClass("VGG16", in_channels=3, out_features=num_classes, h_in=image_size, w_in=image_size)


def save_model_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def learning_rate_schedule(base_lr: float, epoch: int, total_epochs: int) -> float:
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def evaluate_model(
    model: torch.nn.Module,
    dataset,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    cumulative_loss = 0.0
    cumulative_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataset:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            cumulative_loss += loss.item() * x.shape[0]
            cumulative_correct += logits.argmax(dim=1).eq(y).sum().item()
            total += x.shape[0]
    return cumulative_loss / total, cumulative_correct / total


def train_model(
    model: torch.nn.Module,
    dataset_train,
    dataset_val,
    device: torch.device,
    epochs: int,
    *,
    base_lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.nn.Module:
    """Local endpoint training loop aligned with dnn-mode-connectivity training."""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.to(device)
    for epoch in range(epochs):
        lr = learning_rate_schedule(base_lr, epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        cumulative_train_loss = 0.0
        cumulative_train_correct = 0
        total_train = 0
        model.train()
        for x, y in dataset_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_training = criterion(logits, y)
            optimizer.zero_grad()
            loss_training.backward()
            optimizer.step()

            cumulative_train_loss += loss_training.item() * x.shape[0]
            cumulative_train_correct += logits.argmax(dim=1).eq(y).sum().item()
            total_train += x.shape[0]

        cumulative_train_loss /= total_train
        cumulative_train_acc = cumulative_train_correct / total_train
        cumulative_val_loss, cumulative_val_acc = evaluate_model(model, dataset_val, criterion, device)

        print(
            "Epoch {:02d}: lr {:.4f}, train loss {:1.4f}, train acc {:1.2f}, val loss {:1.4f}, val acc {:1.2f}".format(
                epoch + 1,
                lr,
                cumulative_train_loss,
                100.0 * cumulative_train_acc,
                cumulative_val_loss,
                100.0 * cumulative_val_acc,
            )
        )
        if cumulative_val_loss == 0:
            break

    return model


def run_one_batch_debug(
    *,
    VGGClass,
    dataset_train,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Run one manual optimization step to verify data/model wiring."""

    model = build_model(VGGClass, num_classes=10, image_size=int(cfg.image_size)).to(device)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=float(cfg.train_lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()

    x, y = next(iter(dataset_train))
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device, dtype=torch.long)

    model.train()
    before = next(model.parameters()).detach().clone()
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = next(model.parameters()).detach().clone()

    debug_metrics = {
        "loss": float(loss.item()),
        "param_change_norm": float((after - before).norm().item()),
        "out_shape": list(out.shape),
        "y_dtype": str(y.dtype),
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "x_min": float(x.min().item()),
        "x_max": float(x.max().item()),
        "y_min": int(y.min().item()),
        "y_max": int(y.max().item()),
    }

    print("")
    print("=" * 80)
    print("ONE-BATCH DEBUG")
    print("=" * 80)
    print(f"loss: {debug_metrics['loss']}")
    print(f"param_change_norm: {debug_metrics['param_change_norm']}")
    print(f"out_shape: {tuple(debug_metrics['out_shape'])}")
    print(f"y_dtype: {debug_metrics['y_dtype']}")
    print(f"x_shape: {tuple(debug_metrics['x_shape'])}")
    print(f"x_dtype: {debug_metrics['x_dtype']}")
    print(f"x_range: [{debug_metrics['x_min']}, {debug_metrics['x_max']}]")
    print(f"y_range: [{debug_metrics['y_min']}, {debug_metrics['y_max']}]")
    print("")

    return debug_metrics


def run_original_small_mnist_lmc(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))

    (
        VGGClass,
        RebasinNet,
        matching,
        dnn_data,
        RndLoss,
        eval_loss_acc,
        lerp,
    ) = import_original_mnist_components()

    if int(cfg.image_size) != 32:
        raise ValueError("This VGG16 pipeline uses the dnn-mode-connectivity MNIST VGG transform and requires image_size=32.")

    transform_train = dnn_data.Transforms.MNIST.VGG.train
    transform_test = dnn_data.Transforms.MNIST.VGG.test
    mnist_root = os.path.join(to_absolute_path(str(cfg.data_path)), "mnist")

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
    test_size = len(dataset_test)

    dataset_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    dataset_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    dataset_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST LMC")
    print("=" * 80)
    print(f"output_root: {output_root}")
    print(f"device: {device}")
    print(f"image_size: {int(cfg.image_size)}")
    print(f"train_transform: {transform_train}")
    print(f"test_transform: {transform_test}")
    print(f"dataset_split_sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"batch_size: {int(cfg.batch_size)}")
    print(f"train_epochs: {int(cfg.train_epochs)}")
    print(f"train_lr: {float(cfg.train_lr)}")
    print(f"momentum: {float(cfg.momentum)}")
    print(f"weight_decay: {float(cfg.weight_decay)}")
    print(f"alignment_iterations: {int(cfg.alignment_iterations)}")
    print(f"debug_one_batch: {bool(cfg.debug_one_batch)}")
    print("")

    if bool(cfg.debug_one_batch):
        debug_metrics = run_one_batch_debug(
            VGGClass=VGGClass,
            dataset_train=dataset_train,
            device=device,
            cfg=cfg,
        )
        save_json(debug_metrics, output_root / "one_batch_debug.json", indent=2)
        print(f"One-batch debug summary: {output_root / 'one_batch_debug.json'}")
        return {
            "experiment_name": str(cfg.experiment_name),
            "output_root": str(output_root),
            "debug_metrics": debug_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

    modelA = build_model(VGGClass, num_classes=10, image_size=int(cfg.image_size))
    print("Training network A")
    modelA = train_model(
        modelA,
        dataset_train,
        dataset_val,
        device,
        int(cfg.train_epochs),
        base_lr=float(cfg.train_lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
    )
    loss_a, acc_a = eval_loss_acc(modelA, dataset_test, torch.nn.CrossEntropyLoss(), device)
    print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss_a, acc_a))
    modelA.eval()

    modelB = build_model(VGGClass, num_classes=10, image_size=int(cfg.image_size))
    print("\nTraining network B")
    modelB = train_model(
        modelB,
        dataset_train,
        dataset_val,
        device,
        int(cfg.train_epochs),
        base_lr=float(cfg.train_lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
    )
    loss_b, acc_b = eval_loss_acc(modelB, dataset_test, torch.nn.CrossEntropyLoss(), device)
    print("Model B: test loss {:1.3f}, test accuracy {:1.3f}".format(loss_b, acc_b))
    modelB.eval()

    save_model_checkpoint(
        output_root / "model_a.pt",
        modelA,
        {"test_loss": float(loss_a), "test_acc": float(acc_a), "architecture": "VGG16"},
    )
    save_model_checkpoint(
        output_root / "model_b.pt",
        modelB,
        {"test_loss": float(loss_b), "test_acc": float(acc_b), "architecture": "VGG16"},
    )

    pi_modelA = RebasinNet(modelA, input_shape=(1, 3, int(cfg.image_size), int(cfg.image_size)))
    pi_modelA.to(device)

    criterion = RndLoss(modelB, criterion=torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.AdamW(pi_modelA.p.parameters(), lr=float(cfg.alignment_lr))

    print("\nTraining Re-Basing network")
    t1 = time()
    alignment_history: list[dict[str, float | int]] = []
    for iteration in range(int(cfg.alignment_iterations)):
        pi_modelA.train()
        cumulative_train_loss = 0.0
        total_train = 0
        for x, y in dataset_train:
            rebased_model = pi_modelA()
            loss_training = criterion(rebased_model, x.to(device), y.to(device))

            optimizer.zero_grad()
            loss_training.backward()
            optimizer.step()

            cumulative_train_loss += loss_training.item() * x.shape[0]
            total_train += x.shape[0]

        cumulative_train_loss /= total_train

        cumulative_val_loss = 0.0
        total_val = 0
        pi_modelA.eval()
        for x, y in dataset_val:
            rebased_model = pi_modelA()
            loss_validation = criterion(rebased_model, x.to(device), y.to(device))

            cumulative_val_loss += loss_validation.item() * x.shape[0]
            total_val += x.shape[0]

        cumulative_val_loss /= total_val
        alignment_history.append(
            {
                "iteration": iteration,
                "train_loss": float(cumulative_train_loss),
                "val_loss": float(cumulative_val_loss),
            }
        )

        print(
            "Iteration {:02d}: loss training {:1.3f}, loss validation {:1.3f}".format(
                iteration, cumulative_train_loss, cumulative_val_loss
            )
        )
        if cumulative_val_loss == 0:
            break

    print("Elapsed time {:1.3f} secs".format(time() - t1))
    save_json(alignment_history, output_root / "alignment_history.json", indent=2)

    if hasattr(pi_modelA, "update_batchnorm"):
        pi_modelA.update_batchnorm(modelA)
    pi_modelA.eval()
    rebased_model = deepcopy(pi_modelA())
    rebased_model.eval()

    save_model_checkpoint(
        output_root / "rebased_model.pt",
        rebased_model,
        {"method": "original_vgg16_mnist_lmc", "architecture": "VGG16"},
    )
    raw_permutation_parameters = [parameter.detach().cpu().clone() for parameter in pi_modelA.p if parameter is not None]
    hard_permutation_matrices = [
        matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu()
        for parameter in pi_modelA.p
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
    costs_naive = torch.zeros_like(lambdas)
    costs_lmc = torch.zeros_like(lambdas)
    acc_naive = torch.zeros_like(lambdas)
    acc_lmc = torch.zeros_like(lambdas)

    print("\nComputing interpolation for LMC visualization")
    for i in tqdm(range(lambdas.shape[0])):
        lam = lambdas[i]

        temporal_model = lerp(rebased_model, modelB, lam)
        costs_lmc[i], acc_lmc[i] = eval_loss_acc(
            temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
        )

        temporal_model = lerp(modelA, modelB, lam)
        costs_naive[i], acc_naive[i] = eval_loss_acc(
            temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
        )

    plt.figure()
    plt.plot(lambdas, costs_naive, label="Naive")
    plt.plot(lambdas, costs_lmc, label="Sinkhorn Re-basin")
    plt.title("Loss")
    plt.xticks([0, 1], ["ModelA", "ModelB"])
    plt.legend()
    plt.savefig(output_root / "lmc_cnn_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(lambdas, acc_naive, label="Naive")
    plt.plot(lambdas, acc_lmc, label="Sinkhorn Re-basin")
    plt.title("Accuracy")
    plt.xticks([0, 1], ["ModelA", "ModelB"])
    plt.legend()
    plt.savefig(output_root / "lmc_cnn_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    comparison = {
        "lambdas": lambdas.tolist(),
        "costs_naive": costs_naive.tolist(),
        "costs_lmc": costs_lmc.tolist(),
        "acc_naive": (acc_naive * 100.0).tolist(),
        "acc_lmc": (acc_lmc * 100.0).tolist(),
        "model_a_test_loss": float(loss_a),
        "model_a_test_acc": float(acc_a) * 100.0,
        "model_b_test_loss": float(loss_b),
        "model_b_test_acc": float(acc_b) * 100.0,
    }
    save_json(comparison, output_root / "comparison.json", indent=2)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, output_root / "metadata.json", indent=2)

    print("LMC!")
    print(f"Artifacts written under: {output_root}")
    return metadata


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_small_mnist_lmc",
)
def main(cfg: DictConfig) -> None:
    run_original_small_mnist_lmc(cfg)


if __name__ == "__main__":
    main()
