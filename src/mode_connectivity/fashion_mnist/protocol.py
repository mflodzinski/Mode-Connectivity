"""Frozen Fashion-MNIST splits, artifacts, and deterministic data access."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mode_connectivity.training_stage.protocol import (
    StopFlag,
    StopRequested,
    artifact_provenance,
    digest,
    file_hash,
    load,
    make_subsets,
    mixed_seed,
    restore_rng,
    rng_state,
    save,
    seed_all,
    write_json,
)

__all__ = [
    "StopFlag",
    "StopRequested",
    "artifact_provenance",
    "checkpoint",
    "code_hash",
    "Data",
    "digest",
    "file_hash",
    "load",
    "mixed_seed",
    "pair_dir",
    "prepare",
    "protocol_hash",
    "restore_rng",
    "rng_state",
    "root",
    "save",
    "seed_all",
    "verify_protocol",
    "write_json",
]


def root(cfg) -> Path:
    return Path(cfg["output_root"]).resolve()


def checkpoint(cfg, seed: int, epoch: int) -> Path:
    return root(cfg) / "endpoints" / str(seed) / f"epoch_{epoch:03d}.pt"


def pair_dir(cfg, replicate: int, epoch: int) -> Path:
    return root(cfg) / "pairs" / f"r{replicate}" / f"epoch_{epoch:03d}"


def protocol_hash(cfg) -> str:
    return digest(
        {k: v for k, v in cfg.items() if k not in ("output_root", "data_root")}
    )


@lru_cache(maxsize=1)
def code_hash() -> str:
    project = Path(__file__).resolve().parents[3]
    files = sorted((project / "src/mode_connectivity/fashion_mnist").glob("*.py"))
    files += sorted((project / "external/sinkhorn-rebasin/rebasin").rglob("*.py"))
    files += [
        project / "src/mode_connectivity/alignment/permutation_spec.py",
        project / "src/mode_connectivity/alignment/weight_matching.py",
        project / "src/mode_connectivity/training_stage/geometry.py",
    ]
    return digest({str(path.relative_to(project)): file_hash(path) for path in files})


def prepare(cfg) -> None:
    from torchvision.datasets import FashionMNIST

    destination = root(cfg)
    if (destination / "protocol.json").exists():
        verify_protocol(cfg)
        return
    train = FashionMNIST(cfg["data_root"], train=True, download=True)
    test = FashionMNIST(cfg["data_root"], train=False, download=True)
    subsets = make_subsets(train.targets.numpy(), test.targets.numpy(), cfg)
    train_eval = set(subsets["indices"]["train_eval"])
    subsets["indices"]["curve_fit"] = [
        index for index in subsets["indices"]["train"] if index not in train_eval
    ]
    subsets["hashes"]["curve_fit"] = digest(subsets["indices"]["curve_fit"])
    selection = set(subsets["indices"]["selection"])
    subsets["indices"]["validation_audit"] = [
        index for index in subsets["indices"]["validation"] if index not in selection
    ]
    subsets["hashes"]["validation_audit"] = digest(
        subsets["indices"]["validation_audit"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "subsets.json", subsets)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    write_json(
        destination / "protocol.json",
        dict(
            hash=protocol_hash(cfg),
            code_hash=code_hash(),
            revision=revision,
            torch_version=torch.__version__,
            config=cfg,
            subset_hashes=subsets["hashes"],
        ),
    )


def verify_protocol(cfg):
    record = json.loads((root(cfg) / "protocol.json").read_text())
    if record["hash"] != protocol_hash(cfg):
        raise ValueError("Fashion-MNIST protocol changed; use a new output_root.")
    if record["code_hash"] != code_hash():
        raise ValueError(
            "Experiment code changed after preparation; use a new output_root."
        )
    subsets = json.loads((root(cfg) / "subsets.json").read_text())
    if record["subset_hashes"] != subsets["hashes"]:
        raise ValueError("Frozen Fashion-MNIST subset hashes changed.")
    if any(digest(v) != subsets["hashes"][k] for k, v in subsets["indices"].items()):
        raise ValueError("Frozen Fashion-MNIST subset contents changed.")
    return subsets


class IndexedFashionMNIST(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        from PIL import Image

        index = self.indices[position]
        image = Image.fromarray(self.base.data[index].numpy(), mode="L")
        return self.transform(image), int(self.base.targets[index])


class Data:
    def __init__(self, cfg, allow_test: bool = False):
        self.cfg = cfg
        self.allow_test = allow_test
        self.subsets = verify_protocol(cfg)
        self.bases = {}

    def loader(
        self,
        name: str,
        *,
        shuffle: bool = False,
        batch_size: int | None = None,
        order_seed: int | None = None,
    ):
        from torchvision import datasets, transforms

        is_test = name.startswith("test_")
        if is_test and not self.allow_test:
            raise ValueError("Test access is prohibited during training and selection.")
        if is_test not in self.bases:
            self.bases[is_test] = datasets.FashionMNIST(
                self.cfg["data_root"], train=not is_test, download=False
            )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [float(self.cfg["data_mean"])],
                    [float(self.cfg["data_std"])],
                ),
            ]
        )
        dataset = IndexedFashionMNIST(
            self.bases[is_test], self.subsets["indices"][name], transform
        )
        generator = torch.Generator().manual_seed(int(order_seed or 0))
        sampler = None
        if order_seed is not None:
            sampler = torch.randperm(len(dataset), generator=generator).tolist()
            shuffle = False
        workers = int(self.cfg["workers"])
        return DataLoader(
            dataset,
            batch_size=batch_size or int(self.cfg["eval_batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            num_workers=workers,
            pin_memory=str(self.cfg["device"]).startswith("cuda"),
            persistent_workers=False,
            **({"prefetch_factor": 1} if workers else {}),
        )
