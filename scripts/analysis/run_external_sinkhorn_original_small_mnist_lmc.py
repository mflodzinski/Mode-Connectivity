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
import torchvision.transforms as transforms
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_global_seed
from scripts.analysis.run_external_sinkhorn_baseline import import_external_sinkhorn
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def import_original_mnist_components():
    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    for path in (str(examples_root), str(sinkhorn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    VGG, RebasinNet, matching = import_external_sinkhorn()
    from datasets.classification import MNistDataset
    from rebasin.loss import RndLoss
    from utils import eval_loss_acc, lerp, train

    return VGG, RebasinNet, matching, MNistDataset, RndLoss, train, eval_loss_acc, lerp


class NormalizedDataset(torch.utils.data.Dataset):
    """Apply MNIST normalization after the vendored dataset converts to float arrays."""

    def __init__(self, dataset: torch.utils.data.Dataset, *, mean: float, std: float) -> None:
        self.dataset = dataset
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = (x - self.mean) / self.std
        return x.astype("float32"), y


def save_model_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def run_one_batch_debug(
    *,
    VGG,
    dataset_train,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Run one manual optimization step to verify data/model wiring."""

    model = VGG("VGG16", in_channels=1, out_features=10, h_in=int(cfg.image_size), w_in=int(cfg.image_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train_lr))
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
        VGG,
        RebasinNet,
        matching,
        MNistDataset,
        RndLoss,
        train,
        eval_loss_acc,
        lerp,
    ) = import_original_mnist_components()

    transform = transforms.Resize((int(cfg.image_size), int(cfg.image_size)))
    dataset_train_source = MNistDataset(
        root=to_absolute_path(str(cfg.data_path)),
        download=True,
        train=True,
        transform=transform,
    )
    dataset_test_source = MNistDataset(
        root=to_absolute_path(str(cfg.data_path)),
        download=True,
        train=False,
        transform=transform,
    )
    full_dataset = torch.utils.data.ConcatDataset(
        [
            NormalizedDataset(dataset_train_source, mean=MNIST_MEAN, std=MNIST_STD),
            NormalizedDataset(dataset_test_source, mean=MNIST_MEAN, std=MNIST_STD),
        ]
    )
    total_size = len(full_dataset)
    train_fraction = float(cfg.train_fraction)
    val_fraction = float(cfg.val_fraction)
    test_fraction = float(cfg.test_fraction)
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            f"Split fractions must sum to 1.0, received "
            f"train={train_fraction}, val={val_fraction}, test={test_fraction}."
        )
    train_size = int(total_size * train_fraction)
    val_size = int(total_size * val_fraction)
    test_size = total_size - train_size - val_size
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(int(cfg.split_seed)),
    )

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
    print(f"transform: {transform}")
    print(f"dataset_normalization: mean={MNIST_MEAN}, std={MNIST_STD}")
    print(f"dataset_split_sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"batch_size: {int(cfg.batch_size)}")
    print(f"train_epochs: {int(cfg.train_epochs)}")
    print(f"alignment_iterations: {int(cfg.alignment_iterations)}")
    print(f"debug_one_batch: {bool(cfg.debug_one_batch)}")
    print("")

    if bool(cfg.debug_one_batch):
        debug_metrics = run_one_batch_debug(
            VGG=VGG,
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

    modelA = VGG("VGG16", in_channels=1, out_features=10, h_in=int(cfg.image_size), w_in=int(cfg.image_size))
    print("Training network A")
    modelA = train(
        modelA,
        dataset_train,
        dataset_val,
        torch.optim.AdamW(modelA.parameters(), lr=float(cfg.train_lr)),
        torch.nn.CrossEntropyLoss(),
        device,
        int(cfg.train_epochs),
    )
    loss_a, acc_a = eval_loss_acc(modelA, dataset_test, torch.nn.CrossEntropyLoss(), device)
    print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss_a, acc_a))
    modelA.eval()

    modelB = VGG("VGG16", in_channels=1, out_features=10, h_in=int(cfg.image_size), w_in=int(cfg.image_size))
    print("\nTraining network B")
    modelB = train(
        modelB,
        dataset_train,
        dataset_val,
        torch.optim.AdamW(modelB.parameters(), lr=float(cfg.train_lr)),
        torch.nn.CrossEntropyLoss(),
        device,
        int(cfg.train_epochs),
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

    pi_modelA = RebasinNet(modelA, input_shape=(1, 1, int(cfg.image_size), int(cfg.image_size)))
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
