"""Dataset-loading helpers for the retained experiment families.

Most functions here wrap the vendored CIFAR pipelines and expose a consistent
interface for endpoint evaluation, training, and benchmark reproduction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mode_connectivity.common.utils import worker_init_fn
from mode_connectivity.external import load_data_module

dnn_data = load_data_module()


class CIFAR10Utils:
    """CIFAR-10 specific constants and utilities."""

    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    MEAN = np.array([0.4914, 0.4822, 0.4465])
    STD = np.array([0.2470, 0.2435, 0.2616])

    @classmethod
    def get_class_name(cls, idx: int) -> str:
        return cls.CLASS_NAMES[idx]

    @classmethod
    def get_class_names(cls) -> List[str]:
        return cls.CLASS_NAMES.copy()

    @classmethod
    def denormalize(cls, img: np.ndarray) -> np.ndarray:
        img = img.transpose(1, 2, 0)
        img = img * cls.STD + cls.MEAN
        return np.clip(img * 255, 0, 255).astype(np.uint8)


def get_loaders(
    dataset: str,
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    transform_name: str = "VGG",
    use_test: bool = True,
    shuffle_train: bool = True,
) -> Tuple[Dict, int]:
    """Thin wrapper around retained dnn-mode-connectivity loaders."""

    loaders, num_classes = dnn_data.loaders(
        dataset,
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name=transform_name,
        use_test=use_test,
        shuffle_train=shuffle_train,
    )
    return loaders, num_classes


def get_class_names(dataset: str) -> List[str]:
    if dataset.upper() == "CIFAR10":
        return CIFAR10Utils.get_class_names()
    raise NotImplementedError(f"Only CIFAR10 class names are retained; got {dataset}")


def _pin_memory() -> bool:
    return torch.cuda.is_available()


def build_cifar10_pytorch_vgg_eval_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Eval-only CIFAR10 loaders using the pytorch-vgg-cifar10 normalization."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_eval = transforms.Compose([transforms.ToTensor(), normalize])
    root = Path(data_root)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transform_eval, download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transform_eval, download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return train_loader, test_loader


def build_cifar10_pytorch_vgg_train_test_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Shared-train/eval CIFAR10 loaders using the pytorch-vgg-cifar10 recipe."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    root = Path(data_root)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=test_transform, download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return train_loader, test_loader


def build_cifar10_vgg_eval_loaders(
    *,
    data_path: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str]:
    """Eval-only CIFAR10 loaders using the retained VGG transform from the external repo."""

    transform_eval = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(data_path) / "cifar10"
    dataset_train = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_eval)
    dataset_test = datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_eval)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return train_loader, test_loader, str(transform_eval)


def build_cifar10_vgg_train_val_test_loaders(
    *,
    data_path: str | Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    split_seed: int,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    int,
    int,
    int,
    str,
    str,
]:
    """Train/val/test CIFAR10 loaders using retained VGG transforms."""

    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")

    transform_train = dnn_data.Transforms.CIFAR10.VGG.train
    transform_test = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(data_path) / "cifar10"

    dataset_train_source = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_train)
    dataset_val_source = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_test)
    dataset_test_source = datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_test)

    train_total_size = len(dataset_train_source)
    val_size = int(train_total_size * float(val_fraction))
    train_size = train_total_size - val_size
    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(train_total_size, generator=generator)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    dataset_train = torch.utils.data.Subset(dataset_train_source, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val_source, val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_size,
        val_size,
        len(dataset_test_source),
        str(transform_train),
        str(transform_test),
    )


def build_cifar10_vgg_noaug_train_val_test_loaders(
    *,
    data_path: str | Path,
    batch_size: int,
    val_fraction: float,
    split_seed: int,
    train_seed: int,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Deterministic train/val/test loaders using the eval VGG transform everywhere.

    This preserves the retained Sinkhorn sweep behavior where alignment is trained
    without stochastic data augmentation and with single-process data loading.
    """

    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")

    transform_eval = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(data_path) / "cifar10"

    dataset_train_source = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_eval)
    dataset_val_source = datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_eval)
    dataset_test_source = datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_eval)

    full_train_size = len(dataset_train_source)
    val_size = int(full_train_size * float(val_fraction))
    train_size = full_train_size - val_size
    split_generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(full_train_size, generator=split_generator)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    dataset_train = torch.utils.data.Subset(dataset_train_source, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val_source, val_indices)

    effective_num_workers = 0
    train_generator = torch.Generator().manual_seed(int(train_seed))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        generator=train_generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader
