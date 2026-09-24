"""Frozen sources, data splits, persistence, and provenance for dense stage runs."""

from __future__ import annotations

import hashlib
import json
import random
import signal
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False, default=str) + "\n")
    temporary.replace(path)


def save(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load(path: str | Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def root(cfg) -> Path:
    return Path(cfg["output_root"]).resolve()


def source_root(cfg, seed: int) -> Path:
    roots = cfg["source_roots"]
    value = roots.get(str(seed), roots.get(seed))
    if value is None:
        raise KeyError(f"No source root configured for seed {seed}.")
    return Path(value).resolve()


def checkpoint(cfg, seed: int, epoch: int) -> Path:
    return source_root(cfg, seed) / "endpoints" / str(seed) / f"epoch_{epoch:03d}.pt"


def pair_dir(cfg, replicate: int, left_epoch: int, right_epoch: int) -> Path:
    base = Path(cfg.get("_artifact_root", root(cfg)))
    return base / "pairs" / f"r{replicate}" / f"{left_epoch:03d}_{right_epoch:03d}"


def endpoint_manifest(cfg) -> list[dict]:
    rows = []
    for seed in sum(cfg["seed_pairs"], []):
        for epoch in cfg["stages"]:
            path = checkpoint(cfg, seed, epoch)
            if not path.exists():
                raise FileNotFoundError(f"Missing source endpoint: {path}")
            rows.append(
                dict(seed=seed, epoch=epoch, path=str(path), sha256=file_hash(path))
            )
    return rows


_VERIFIED_ENDPOINTS: dict[str, str] = {}


def verify_endpoint(cfg, path: str | Path) -> str:
    path = Path(path).resolve()
    rows = json.loads((root(cfg) / "endpoints.json").read_text())
    expected = {str(Path(row["path"]).resolve()): row["sha256"] for row in rows}
    if str(path) not in expected:
        raise ValueError(f"Endpoint is outside the frozen manifest: {path}")
    if _VERIFIED_ENDPOINTS.get(str(path)) == expected[str(path)]:
        return expected[str(path)]
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_hash(path)
    if actual != expected[str(path)]:
        raise ValueError(f"Source endpoint hash changed: {path}")
    _VERIFIED_ENDPOINTS[str(path)] = actual
    return actual


def protocol_hash(cfg) -> str:
    if cfg.get("_protocol_hash_override"):
        return str(cfg["_protocol_hash_override"])
    ignored = {
        "output_root", "device", "workers", "cpu_threads", "slurm_resources",
        "alignment_pairs_per_task", "evaluation_pairs_per_task", "endpoint_pairs_per_task",
        "calibration_pairs_per_task", "replication_pairs_per_task",
        "calibration_evaluation_pairs_per_task",
    }
    return digest({k: v for k, v in cfg.items() if k not in ignored and not k.startswith("_")})


@lru_cache(maxsize=1)
def code_hash() -> str:
    project = Path(__file__).resolve().parents[3]
    files = sorted((project / "src/mode_connectivity/dense_linear_stage").glob("*.py"))
    files += sorted((project / "external/sinkhorn-rebasin/rebasin").rglob("*.py"))
    files += [
        project / "src/mode_connectivity/alignment/weight_matching.py",
        project / "src/mode_connectivity/alignment/permutation_spec.py",
        project / "src/mode_connectivity/sinkhorn/shared.py",
    ]
    return digest({str(p.relative_to(project)): file_hash(p) for p in files})


def _source_records(cfg) -> tuple[dict, list[dict]]:
    records, common = [], None
    for source in sorted({source_root(cfg, s) for s in sum(cfg["seed_pairs"], [])}):
        protocol_path, subset_path = source / "protocol.json", source / "subsets.json"
        if not protocol_path.exists() or not subset_path.exists():
            raise FileNotFoundError(f"Source protocol is incomplete: {source}")
        protocol = json.loads(protocol_path.read_text())
        subsets = json.loads(subset_path.read_text())
        cfg_source = protocol.get("config", protocol)
        if cfg["dataset"] == "cifar10" and cfg_source.get("model") != cfg["model"]:
            raise ValueError(
                f"Source {source} contains {cfg_source.get('model')}, expected {cfg['model']}."
            )
        if cfg["dataset"] == "fashion_mnist" and (
            int(cfg_source.get("hidden_layers", -1)) != int(cfg["hidden_layers"])
            or int(cfg_source.get("hidden_width", -1)) != int(cfg["hidden_width"])
        ):
            raise ValueError(f"Source {source} does not contain the requested Fashion MLP.")
        current = dict(
            source_root=str(source),
            protocol_sha256=file_hash(protocol_path),
            subsets_sha256=file_hash(subset_path),
            model=cfg_source.get("model"),
            hidden_layers=cfg_source.get("hidden_layers"),
            hidden_width=cfg_source.get("hidden_width"),
            split_seed=cfg_source.get("split_seed"),
            validation_size=cfg_source.get("validation_size"),
            train_hash=subsets["hashes"]["train"],
            validation_hash=subsets["hashes"]["validation"],
            train_indices=subsets["indices"]["train"],
            validation_indices=subsets["indices"]["validation"],
        )
        comparable = {
            k: current[k]
            for k in ("split_seed", "validation_size", "train_hash", "validation_hash")
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError("Source roots do not use identical frozen train/validation splits.")
        records.append({k: v for k, v in current.items() if not k.endswith("_indices")})
    first_root = source_root(cfg, sum(cfg["seed_pairs"], [])[0])
    subsets = json.loads((first_root / "subsets.json").read_text())
    return subsets, records


def _split_validation(indices, labels, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    tune, select = [], []
    values = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    for cls in sorted(np.unique(labels[values]).tolist()):
        members = values[labels[values] == cls].copy()
        rng.shuffle(members)
        midpoint = len(members) // 2
        tune.extend(members[:midpoint].tolist())
        select.extend(members[midpoint:].tolist())
    rng.shuffle(tune)
    rng.shuffle(select)
    return tune, select


def _stratified_subset(indices, labels, size: int, seed: int) -> list[int]:
    """Select an exact-size deterministic class-balanced reporting subset."""
    values = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    if size < 1 or size > len(values):
        raise ValueError(f"Requested subset size {size} outside [1, {len(values)}].")
    classes = sorted(np.unique(labels[values]).tolist())
    per_class, remainder = divmod(int(size), len(classes))
    rng, selected = np.random.default_rng(seed), []
    for position, cls in enumerate(classes):
        members = values[labels[values] == cls].copy()
        rng.shuffle(members)
        count = per_class + int(position < remainder)
        if count > len(members):
            raise ValueError(f"Class {cls} has only {len(members)} candidates for {count} slots.")
        selected.extend(members[:count].tolist())
    rng.shuffle(selected)
    return selected


def prepare(cfg) -> None:
    destination = root(cfg)
    if (destination / "protocol.json").exists():
        verify_protocol(cfg)
        return
    destination.mkdir(parents=True, exist_ok=True)
    source_subsets, sources = _source_records(cfg)
    if cfg["dataset"] == "cifar10":
        from torchvision.datasets import CIFAR10

        train = CIFAR10(cfg["data_root"], train=True, download=True)
        test = CIFAR10(cfg["data_root"], train=False, download=True)
        labels = np.asarray(train.targets)
    elif cfg["dataset"] == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        train = FashionMNIST(cfg["data_root"], train=True, download=True)
        test = FashionMNIST(cfg["data_root"], train=False, download=True)
        labels = train.targets.numpy()
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    train_indices = list(source_subsets["indices"]["train"])
    validation = list(source_subsets["indices"]["validation"])
    tune, select = _split_validation(validation, labels, int(cfg["validation_split_seed"]))
    expected_train = len(train) - len(validation)
    if len(train_indices) != expected_train:
        raise ValueError(
            f"Source endpoint train split has {len(train_indices)} examples; expected {expected_train}."
        )
    train_report = _stratified_subset(
        train_indices,
        labels,
        int(cfg["train_report_size"]),
        int(cfg["report_subset_seed"]),
    )
    subsets = dict(
        indices=dict(
            train_fit=train_indices,
            train_report=train_report,
            val_tune=tune,
            val_select=select,
            validation=validation,
            test_full=list(range(len(test))),
        )
    )
    subsets["hashes"] = {k: digest(v) for k, v in subsets["indices"].items()}
    endpoints = endpoint_manifest(cfg)
    write_json(destination / "subsets.json", subsets)
    write_json(destination / "endpoints.json", endpoints)
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
            sources=sources,
            subset_hashes=subsets["hashes"],
            endpoint_manifest_hash=digest(endpoints),
            test_data_used=False,
        ),
    )


def verify_protocol(cfg) -> dict:
    directory = root(cfg)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["hash"] != protocol_hash(cfg):
        raise ValueError("Dense-stage protocol changed; use a new output_root.")
    if protocol["code_hash"] != code_hash():
        raise ValueError("Dense-stage code changed after preparation; use a new output_root.")
    subsets = json.loads((directory / "subsets.json").read_text())
    if subsets["hashes"] != protocol["subset_hashes"]:
        raise ValueError("Frozen subset hashes changed.")
    if any(digest(v) != subsets["hashes"][k] for k, v in subsets["indices"].items()):
        raise ValueError("Frozen subset contents changed.")
    endpoints = json.loads((directory / "endpoints.json").read_text())
    if digest(endpoints) != protocol["endpoint_manifest_hash"]:
        raise ValueError("Frozen endpoint manifest changed.")
    for row in endpoints:
        if not Path(row["path"]).exists():
            raise ValueError(f"Source endpoint disappeared: {row['path']}")
    return subsets


class IndexedDataset(Dataset):
    def __init__(self, base, indices, transform, fashion: bool):
        self.base, self.indices, self.transform, self.fashion = base, list(indices), transform, fashion

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        from PIL import Image

        index = self.indices[position]
        if self.fashion:
            image = Image.fromarray(self.base.data[index].numpy(), mode="L")
            label = int(self.base.targets[index])
        else:
            image = Image.fromarray(self.base.data[index])
            label = int(self.base.targets[index])
        return self.transform(image), label


class Data:
    def __init__(self, cfg, allow_test: bool = False):
        self.cfg, self.allow_test = cfg, allow_test
        self.subsets = verify_protocol(cfg)
        self.bases = {}

    def loader(self, name: str, *, fit=False, order_seed=None, batch_size=None):
        from torchvision import datasets, transforms

        is_test = name.startswith("test_")
        if is_test and not self.allow_test:
            raise ValueError("Test access is prohibited before artifacts and selections are frozen.")
        if is_test not in self.bases:
            cls = datasets.CIFAR10 if self.cfg["dataset"] == "cifar10" else datasets.FashionMNIST
            self.bases[is_test] = cls(self.cfg["data_root"], train=not is_test, download=False)
        fashion = self.cfg["dataset"] == "fashion_mnist"
        if fashion:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([float(self.cfg["data_mean"])], [float(self.cfg["data_std"])])
                ]
            )
        else:
            augmentation = (
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
                if fit and self.cfg.get("fit_augmentation", False)
                else []
            )
            transform = transforms.Compose(
                augmentation
                + [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        dataset = IndexedDataset(
            self.bases[is_test], self.subsets["indices"][name], transform, fashion
        )
        sampler = None
        generator = torch.Generator().manual_seed(int(order_seed or 0))
        if order_seed is not None:
            sampler = torch.randperm(len(dataset), generator=generator).tolist()
        workers = int(self.cfg["workers"])
        return DataLoader(
            dataset,
            batch_size=int(batch_size or self.cfg["eval_batch_size"]),
            sampler=sampler,
            shuffle=False,
            generator=generator,
            num_workers=workers,
            pin_memory=str(self.cfg["device"]).startswith("cuda"),
            persistent_workers=False,
            **({"prefetch_factor": 1} if workers else {}),
        )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state) -> None:
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
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, self._set)

    def _set(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise StopRequested("Pre-timeout signal received; resume from recovery state.")
