"""Train epoch-split shared-training pairs using the external/pytorch-vgg-cifar10 recipe.

This mirrors the old shared-split experiment structure while using the VGG16/CIFAR10
model, data pipeline, and learning-rate schedule from `external/pytorch-vgg-cifar10`.

Key property versus the upstream script:
- checkpoints include optimizer state, so split-resume is correct at epoch boundaries
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PYTORCH_VGG_ROOT = PROJECT_ROOT / "external" / "pytorch-vgg-cifar10"

sys.path.insert(0, str(PYTORCH_VGG_ROOT))
import vgg  # type: ignore  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(arch: str, device: torch.device) -> nn.Module:
    model = vgg.__dict__[arch]()
    model.features = torch.nn.DataParallel(model.features)
    model.to(device)
    return model


def build_loaders(data_root: str, batch_size: int, workers: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def adjust_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int) -> float:
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return 100.0 * correct / target.size(0)


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * total_correct / total


def validate(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, 100.0 * total_correct / total


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_prec1: float,
) -> dict:
    return {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_prec1": best_prec1,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]), float(checkpoint.get("best_prec1", 0.0))


def save_epoch_artifacts(
    *,
    run_dir: Path,
    epoch: int,
    final_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_prec1: float,
    save_every: int,
) -> None:
    payload = checkpoint_payload(epoch=epoch, model=model, optimizer=optimizer, best_prec1=best_prec1)
    if save_every > 0 and epoch % save_every == 0:
        save_checkpoint(run_dir / f"checkpoint-{epoch}.pt", payload)
    if epoch == final_epochs:
        save_checkpoint(run_dir / f"checkpoint-{epoch}.pt", payload)
        torch.save(model.state_dict(), run_dir / "model_final_state_dict.pth")
        torch.save(payload, run_dir / "model_final.pth.tar")


def train_range(
    *,
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    start_epoch: int,
    final_epochs: int,
    base_lr: float,
    save_every: int,
    best_prec1: float,
    epoch_print_freq: int,
) -> float:
    for epoch in range(start_epoch, final_epochs):
        current_lr = adjust_learning_rate(optimizer, base_lr, epoch)
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = validate(test_loader, model, criterion, device)
        best_prec1 = max(best_prec1, val_acc)

        epoch_completed = epoch + 1
        save_epoch_artifacts(
            run_dir=run_dir,
            epoch=epoch_completed,
            final_epochs=final_epochs,
            model=model,
            optimizer=optimizer,
            best_prec1=best_prec1,
            save_every=save_every,
        )

        if (epoch_completed % epoch_print_freq == 0) or (epoch_completed == final_epochs):
            print(
                f"Epoch {epoch_completed}/{final_epochs}\t"
                f"lr {current_lr:.5f}\t"
                f"train_loss {train_loss:.4f}\ttrain_acc {train_acc:.3f}\t"
                f"val_loss {val_loss:.4f}\tval_acc {val_acc:.3f}\tbest_val_acc {best_prec1:.3f}"
            )
    return best_prec1


@hydra.main(
    version_base=None,
    config_path="../../configs/pytorch_vgg/endpoints",
    config_name="vgg16_lmc_connected_pair_30split",
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = False
    cudnn.deterministic = True

    output_root = Path(to_absolute_path(cfg.output_root))
    data_root = to_absolute_path(cfg.data_root)
    arch = cfg.arch

    shared_dir = output_root / "shared"

    print("=" * 72)
    print("PYTORCH-VGG16 SHARED-SPLIT TRAINING")
    print("=" * 72)
    print(f"Arch: {arch}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Shared epochs: {cfg.shared_epochs}")
    print(f"Final epochs: {cfg.final_epochs}")
    print(f"Shared seed: {cfg.shared_seed}")
    print(f"Split seeds: {list(cfg.split_seeds)}")

    if output_root.exists():
        print(f"Removing existing output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Stage 1: shared phase.
    if cfg.shared_epochs == 0:
        seed_all(int(cfg.shared_seed))
        model = build_model(arch, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.wd),
        )
        save_checkpoint(
            shared_dir / "checkpoint-0.pt",
            checkpoint_payload(epoch=0, model=model, optimizer=optimizer, best_prec1=0.0),
        )
        print(f"Saved random shared initialization to {shared_dir / 'checkpoint-0.pt'}")
    else:
        seed_all(int(cfg.shared_seed))
        model = build_model(arch, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.wd),
        )
        criterion = nn.CrossEntropyLoss().to(device)
        train_loader, test_loader = build_loaders(data_root, int(cfg.batch_size), int(cfg.workers))
        best_prec1 = train_range(
            run_dir=shared_dir,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            start_epoch=0,
            final_epochs=int(cfg.shared_epochs),
            base_lr=float(cfg.lr),
            save_every=int(cfg.save_every),
            best_prec1=0.0,
            epoch_print_freq=int(cfg.epoch_print_freq),
        )
        print(f"Shared phase complete. Best val acc: {best_prec1:.3f}")

    shared_checkpoint = shared_dir / f"checkpoint-{cfg.shared_epochs}.pt"
    if not shared_checkpoint.exists():
        raise FileNotFoundError(f"Missing shared checkpoint: {shared_checkpoint}")

    # Stage 2: split branches.
    for branch_idx, seed in enumerate(cfg.split_seeds):
        branch_dir = output_root / f"seed{branch_idx}"
        seed = int(seed)
        print("\n" + "-" * 72)
        print(f"Training branch {branch_idx} with seed {seed}")
        print("-" * 72)

        seed_all(seed)
        model = build_model(arch, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.wd),
        )
        criterion = nn.CrossEntropyLoss().to(device)
        start_epoch, best_prec1 = load_checkpoint(shared_checkpoint, model, optimizer)
        train_loader, test_loader = build_loaders(data_root, int(cfg.batch_size), int(cfg.workers))
        best_prec1 = train_range(
            run_dir=branch_dir,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            start_epoch=start_epoch,
            final_epochs=int(cfg.final_epochs),
            base_lr=float(cfg.lr),
            save_every=int(cfg.save_every),
            best_prec1=best_prec1,
            epoch_print_freq=int(cfg.epoch_print_freq),
        )
        print(f"Branch {branch_idx} final checkpoint: {branch_dir / f'checkpoint-{cfg.final_epochs}.pt'}")
        print(f"Branch {branch_idx} best val acc: {best_prec1:.3f}")


if __name__ == "__main__":
    main()
