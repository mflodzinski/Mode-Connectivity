"""Protocol, durable artifacts, and bounded-memory CIFAR data access."""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load(path):
    # Only experiment-owned checkpoints; recovery includes Python/NumPy RNG state.
    return torch.load(path, map_location="cpu", weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixed_seed(seed, label):
    """Stable integer seed for independently reproducible RNG streams."""
    payload = f"{int(seed)}:{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class StopRequested(RuntimeError):
    pass


class StopFlag:
    def __init__(self):
        self.requested = False
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._set)

    def _set(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise StopRequested("Interrupted; resume from the last durable boundary.")


def stratified_take(labels, indices, count, rng):
    labels, indices = np.asarray(labels), np.asarray(indices, dtype=np.int64)
    classes = np.unique(labels[indices])
    if count % len(classes):
        raise ValueError("Subset sizes must be divisible by the number of classes.")
    selected = []
    for cls in classes:
        candidates = indices[labels[indices] == cls].copy()
        rng.shuffle(candidates)
        if len(candidates) < count // len(classes):
            raise ValueError("Insufficient examples for disjoint stratified subsets.")
        selected.extend(candidates[: count // len(classes)].tolist())
    rng.shuffle(selected)
    return selected


def make_subsets(train_labels, test_labels, cfg):
    rng = np.random.default_rng(cfg["split_seed"])
    all_train = np.arange(len(train_labels))
    val = stratified_take(train_labels, all_train, cfg["validation_size"], rng)
    heldout_train = np.setdiff1d(all_train, val).tolist()
    train = (
        all_train.tolist() if cfg.get("train_full_data", False) else heldout_train
    )
    opt = stratified_take(
        train_labels, heldout_train, cfg["alignment_size"], rng
    )
    train_eval = stratified_take(
        train_labels,
        np.setdiff1d(heldout_train, opt),
        cfg["train_eval_size"],
        rng,
    )
    subsets = dict(
        train=train,
        train_full=all_train.tolist(),
        validation=val,
        alignment=opt,
        train_eval=train_eval,
        selection=stratified_take(train_labels, val, cfg["selection_size"], rng),
        test_eval=stratified_take(
            test_labels, np.arange(len(test_labels)), cfg["test_eval_size"], rng
        ),
        test_full=list(range(len(test_labels))),
    )
    return dict(indices=subsets, hashes={k: digest(v) for k, v in subsets.items()})


def root(cfg):
    return Path(cfg["output_root"]).resolve()


def checkpoint(cfg, seed, epoch):
    return root(cfg) / "endpoints" / str(seed) / f"epoch_{epoch:03d}.pt"


def pair_name(a, b):
    return f"{a:03d}_{b:03d}"


def pair_dir(cfg, replicate, a, b):
    return root(cfg) / "pairs" / str(replicate) / pair_name(a, b)


def protocol_hash(cfg):
    return digest(
        {k: v for k, v in cfg.items() if k not in ("output_root", "data_root")}
    )


def verify_protocol(cfg):
    record = json.loads((root(cfg) / "protocol.json").read_text())
    if record["hash"] != protocol_hash(cfg):
        raise ValueError("Protocol changed: use a new output_root.")
    if record.get("code_hash") != code_hash():
        raise ValueError(
            "Experiment code changed after protocol preparation: use a new output_root."
        )
    subsets = json.loads((root(cfg) / "subsets.json").read_text())
    if any(digest(v) != subsets["hashes"][k] for k, v in subsets["indices"].items()):
        raise ValueError("Subset indices failed integrity verification.")
    if record["subset_hashes"] != subsets["hashes"]:
        raise ValueError("Subsets differ from the frozen protocol.")
    return subsets


def prepare(cfg):
    from torchvision.datasets import CIFAR10

    destination = root(cfg)
    if (destination / "protocol.json").exists():
        verify_protocol(cfg)
        return
    train = CIFAR10(cfg["data_root"], train=True, download=True)
    test = CIFAR10(cfg["data_root"], train=False, download=True)
    subsets = make_subsets(train.targets, test.targets, cfg)
    write_json(destination / "subsets.json", subsets)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    write_json(
        destination / "protocol.json",
        dict(
            hash=protocol_hash(cfg),
            config=cfg,
            subset_hashes=subsets["hashes"],
            revision=revision,
            code_hash=code_hash(),
            torch_version=torch.__version__,
        ),
    )


class CIFARView(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from PIL import Image

        i = self.indices[index]
        return self.transform(Image.fromarray(self.base.data[i])), self.base.targets[i]


class Data:
    def __init__(self, cfg, allow_test=False):
        self.cfg, self.allow_test = cfg, allow_test
        self.subsets = verify_protocol(cfg)
        self.bases = {}

    def loader(
        self,
        name,
        augment=False,
        shuffle=False,
        batch_size=None,
        raw=False,
        order_seed=None,
        worker_seed=None,
    ):
        from torchvision import datasets, transforms

        is_test = name.startswith("test_")
        if is_test and not self.allow_test:
            raise ValueError("Test access prohibited during training/alignment/audit.")
        if is_test not in self.bases:
            self.bases[is_test] = datasets.CIFAR10(
                self.cfg["data_root"], train=not is_test, download=False
            )
        if raw:
            transform = transforms.PILToTensor()
        else:
            if self.cfg.get("data_recipe") == "git_rebasin_cifar10" and augment:
                raise ValueError(
                    "Git Re-Basin augmentation is applied to raw batches in training.py."
                )
            transform = transforms.Compose(
                (
                    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
                    if augment
                    else []
                )
                + [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        dataset = CIFARView(
            self.bases[is_test], self.subsets["indices"][name], transform
        )
        workers = int(self.cfg["workers"])
        generator = None
        sampler = None
        if order_seed is not None:
            generator = torch.Generator().manual_seed(int(order_seed))
            sampler = torch.randperm(len(dataset), generator=generator).tolist()
            shuffle = False
        worker_generator = torch.Generator().manual_seed(
            int(worker_seed if worker_seed is not None else order_seed or 0)
        )
        return DataLoader(
            dataset,
            batch_size=batch_size or self.cfg["eval_batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            generator=worker_generator,
            num_workers=workers,
            pin_memory=self.cfg["device"].startswith("cuda"),
            persistent_workers=False,
            **({"prefetch_factor": 1} if workers else {}),
        )


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def artifact_provenance(cfg, endpoint_paths):
    return dict(
        protocol_hash=protocol_hash(cfg),
        code_hash=code_hash(),
        endpoints=[
            dict(path=str(path), sha256=file_hash(path)) for path in endpoint_paths
        ],
    )


@lru_cache(maxsize=1)
def code_hash():
    project = Path(__file__).resolve().parents[3]
    files = sorted((project / "src/mode_connectivity/training_stage").glob("*.py"))
    files += sorted((project / "external/sinkhorn-rebasin/rebasin").rglob("*.py"))
    files += [
        project / "external/sinkhorn-rebasin/examples/models/vgg.py",
        project / "src/mode_connectivity/alignment/weight_matching.py",
        project / "src/mode_connectivity/alignment/permutation_spec.py",
        project / "src/mode_connectivity/sinkhorn/shared.py",
    ]
    return digest({str(p.relative_to(project)): file_hash(p) for p in files})
