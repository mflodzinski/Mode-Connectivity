"""Train VGG16 shared-split pairs with iteration-level split points.

This script is designed for the controlled benchmark setting where:
1. a shared training trunk is run from scratch,
2. checkpoints are saved at exact SGD iteration milestones, and
3. two branch models resume from one such milestone with different batch-order
   seeds and continue until the same final training horizon.

Unlike the existing epoch-based shared-split scripts, this runner supports
splitting inside an epoch. The split unit is one optimizer step on a batch of
size 128.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "dnn-mode-connectivity"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXTERNAL_ROOT))

import data  # type: ignore  # noqa: E402
import models  # type: ignore  # noqa: E402
import utils  # type: ignore  # noqa: E402

from src.utils import set_global_seed, worker_init_fn  # noqa: E402


class FixedOrderSampler(Sampler[int]):
    """Yield indices in the exact order provided."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train iteration-level shared-split VGG16 pairs for the controlled benchmark."
    )
    parser.add_argument("--mode", choices=["shared", "pair"], required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split-iter", type=int, default=None, help="Split iteration for pair mode")
    parser.add_argument(
        "--shared-iters",
        type=int,
        nargs="+",
        default=None,
        help="Shared trunk checkpoints to save in shared mode, e.g. 0 25 100 1000 5000",
    )
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--transform", type=str, default="VGG")
    parser.add_argument("--model", type=str, default="VGG16")
    parser.add_argument("--shared-seed", type=int, default=42)
    parser.add_argument("--split-seeds", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--no-train-aug", action="store_true")
    parser.add_argument("--save-freq-epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datasets(args: argparse.Namespace):
    ds_cls = getattr(torchvision.datasets, args.dataset)
    transform_bundle = getattr(getattr(data.Transforms, args.dataset), args.transform)
    train_transform = transform_bundle.test if args.no_train_aug else transform_bundle.train
    dataset_path = os.path.join(args.data_path, args.dataset.lower())
    train_set = ds_cls(dataset_path, train=True, download=True, transform=train_transform)
    test_set = ds_cls(dataset_path, train=False, download=True, transform=transform_bundle.test)
    num_classes = int(max(train_set.targets)) + 1
    return train_set, test_set, num_classes


def build_test_loader(test_set, batch_size: int, num_workers: int) -> DataLoader:
    pin_mem = not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        num_workers = 0
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def epoch_permutation(num_examples: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed) * 1_000_003 + int(epoch))
    return torch.randperm(num_examples, generator=generator).tolist()


def build_epoch_loader(
    train_set,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
    start_batch: int,
) -> DataLoader:
    indices = epoch_permutation(len(train_set), seed, epoch)
    start_index = min(start_batch * batch_size, len(indices))
    remaining_indices = indices[start_index:]
    sampler = FixedOrderSampler(remaining_indices)
    pin_mem = not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        num_workers = 0
    return DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def build_model_and_optimizer(
    model_name: str,
    num_classes: int,
    lr: float,
    momentum: float,
    wd: float,
    device: torch.device,
):
    model_class = getattr(models, model_name)
    model = model_class.base(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
    )
    return model, optimizer


def learning_rate_schedule(base_lr: float, epoch: int, total_epochs: int) -> float:
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def checkpoint_training_position(global_iter: int, iters_per_epoch: int) -> tuple[int, int, int]:
    completed_epochs = global_iter // iters_per_epoch
    next_epoch = completed_epochs + 1
    iter_in_epoch = global_iter % iters_per_epoch
    return completed_epochs, next_epoch, iter_in_epoch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_iter: int,
    iters_per_epoch: int,
    train_seed: int,
    batch_size: int,
) -> None:
    completed_epochs, next_epoch, iter_in_epoch = checkpoint_training_position(global_iter, iters_per_epoch)
    torch.save(
        {
            "epoch": completed_epochs,
            "next_epoch": next_epoch,
            "iter_in_epoch": iter_in_epoch,
            "global_iter": global_iter,
            "iters_per_epoch": iters_per_epoch,
            "train_seed": train_seed,
            "batch_size": batch_size,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iters_per_epoch: int,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    global_iter = int(checkpoint.get("global_iter", int(checkpoint["epoch"]) * iters_per_epoch))
    next_epoch = int(checkpoint.get("next_epoch", global_iter // iters_per_epoch + 1))
    iter_in_epoch = int(checkpoint.get("iter_in_epoch", global_iter % iters_per_epoch))
    return global_iter, next_epoch, iter_in_epoch


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += F.cross_entropy(outputs, targets, reduction="sum").item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def train_until_iter(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set,
    args: argparse.Namespace,
    order_seed: int,
    start_global_iter: int,
    start_epoch: int,
    start_iter_in_epoch: int,
    target_global_iter: int,
    device: torch.device,
    checkpoint_dir: Path,
    milestone_iters: set[int] | None,
    final_checkpoint_name: str | None,
) -> None:
    criterion = F.cross_entropy
    iters_per_epoch = math.ceil(len(train_set) / args.batch_size)
    global_iter = start_global_iter
    epoch = start_epoch
    iter_in_epoch = start_iter_in_epoch

    while global_iter < target_global_iter:
        lr = learning_rate_schedule(args.lr, epoch, args.final_epochs)
        utils.adjust_learning_rate(optimizer, lr)
        loader = build_epoch_loader(
            train_set,
            args.batch_size,
            args.num_workers,
            order_seed,
            epoch,
            iter_in_epoch,
        )

        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_batch_for_log = iter_in_epoch

        for batch_idx, (inputs, targets) in enumerate(loader, start=iter_in_epoch):
            if global_iter >= target_global_iter:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            epoch_correct += pred.eq(targets).sum().item()
            epoch_total += targets.size(0)

            global_iter += 1

            if milestone_iters and global_iter in milestone_iters:
                milestone_path = checkpoint_dir / f"checkpoint-iter{global_iter}.pt"
                save_checkpoint(
                    milestone_path,
                    model,
                    optimizer,
                    global_iter,
                    iters_per_epoch,
                    order_seed,
                    args.batch_size,
                )
                print(f"[milestone] saved {milestone_path}")

        epoch_acc = 100.0 * epoch_correct / max(epoch_total, 1)
        epoch_loss = epoch_loss_sum / max(epoch_total, 1)
        completed_epochs, next_epoch, next_iter_in_epoch = checkpoint_training_position(global_iter, iters_per_epoch)
        print(
            f"[train] seed={order_seed} epoch={epoch} "
            f"start_batch={start_batch_for_log} next_iter={global_iter}/{target_global_iter} "
            f"loss={epoch_loss:.4f} acc={epoch_acc:.2f}% lr={lr:.5f}"
        )

        if global_iter % iters_per_epoch == 0:
            if args.save_freq_epochs > 0 and completed_epochs > 0 and completed_epochs % args.save_freq_epochs == 0:
                epoch_ckpt = checkpoint_dir / f"checkpoint-{completed_epochs}.pt"
                save_checkpoint(
                    epoch_ckpt,
                    model,
                    optimizer,
                    global_iter,
                    iters_per_epoch,
                    order_seed,
                    args.batch_size,
                )
                print(f"[epoch] saved {epoch_ckpt}")

        epoch = next_epoch
        iter_in_epoch = next_iter_in_epoch

    if final_checkpoint_name:
        final_path = checkpoint_dir / final_checkpoint_name
        save_checkpoint(
            final_path,
            model,
            optimizer,
            global_iter,
            iters_per_epoch,
            order_seed,
            args.batch_size,
        )
        print(f"[final] saved {final_path}")


def run_shared_mode(args: argparse.Namespace, device: torch.device) -> None:
    if not args.shared_iters:
        raise ValueError("--shared-iters is required in shared mode")

    shared_iters = sorted(set(int(x) for x in args.shared_iters))
    if shared_iters[0] != 0:
        raise ValueError("--shared-iters must include 0 so the initial checkpoint is saved explicitly")

    output_root = Path(args.output_root)
    shared_dir = output_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set, num_classes = build_datasets(args)
    _ = build_test_loader(test_set, args.batch_size, args.num_workers)
    iters_per_epoch = math.ceil(len(train_set) / args.batch_size)

    set_global_seed(args.shared_seed)
    model, optimizer = build_model_and_optimizer(
        args.model, num_classes, args.lr, args.momentum, args.wd, device
    )

    init_path = shared_dir / "checkpoint-iter0.pt"
    save_checkpoint(
        init_path,
        model,
        optimizer,
        global_iter=0,
        iters_per_epoch=iters_per_epoch,
        train_seed=args.shared_seed,
        batch_size=args.batch_size,
    )
    print(f"[init] saved {init_path}")

    max_shared_iter = max(shared_iters)
    if max_shared_iter == 0:
        return

    train_until_iter(
        model=model,
        optimizer=optimizer,
        train_set=train_set,
        args=args,
        order_seed=args.shared_seed,
        start_global_iter=0,
        start_epoch=1,
        start_iter_in_epoch=0,
        target_global_iter=max_shared_iter,
        device=device,
        checkpoint_dir=shared_dir,
        milestone_iters=set(shared_iters[1:]),
        final_checkpoint_name=None,
    )


def run_pair_mode(args: argparse.Namespace, device: torch.device) -> None:
    if args.split_iter is None:
        raise ValueError("--split-iter is required in pair mode")

    split_iter = int(args.split_iter)
    output_root = Path(args.output_root)
    shared_ckpt = output_root / "shared" / f"checkpoint-iter{split_iter}.pt"
    if not shared_ckpt.exists():
        raise FileNotFoundError(f"Shared checkpoint not found: {shared_ckpt}")

    train_set, test_set, num_classes = build_datasets(args)
    test_loader = build_test_loader(test_set, args.batch_size, args.num_workers)
    iters_per_epoch = math.ceil(len(train_set) / args.batch_size)
    total_target_iters = args.final_epochs * iters_per_epoch

    for idx, split_seed in enumerate(args.split_seeds):
        branch_dir = output_root / f"iter{split_iter}" / f"seed{idx}"
        branch_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(split_seed)
        model, optimizer = build_model_and_optimizer(
            args.model, num_classes, args.lr, args.momentum, args.wd, device
        )
        start_global_iter, start_epoch, start_iter_in_epoch = load_checkpoint(
            shared_ckpt, model, optimizer, device, iters_per_epoch
        )

        print(
            f"[resume] split_iter={split_iter} branch_seed={split_seed} "
            f"start_global_iter={start_global_iter} start_epoch={start_epoch} "
            f"start_iter_in_epoch={start_iter_in_epoch}"
        )

        train_until_iter(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            args=args,
            order_seed=split_seed,
            start_global_iter=start_global_iter,
            start_epoch=start_epoch,
            start_iter_in_epoch=start_iter_in_epoch,
            target_global_iter=total_target_iters,
            device=device,
            checkpoint_dir=branch_dir,
            milestone_iters=None,
            final_checkpoint_name=f"checkpoint-{args.final_epochs}.pt",
        )

        final_ckpt = branch_dir / f"checkpoint-{args.final_epochs}.pt"
        final_state = torch.load(final_ckpt, map_location=device)
        model.load_state_dict(final_state["model_state"])
        metrics = evaluate_model(model, test_loader, device)
        print(
            f"[eval] split_iter={split_iter} branch_seed={split_seed} "
            f"test_loss={metrics['loss']:.4f} test_acc={metrics['accuracy']:.2f}%"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Train augmentation disabled: {args.no_train_aug}")

    if args.mode == "shared":
        run_shared_mode(args, device)
    elif args.mode == "pair":
        run_pair_mode(args, device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
