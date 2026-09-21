"""Frozen sources, pair enumeration, data access, and durable artifacts."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mode_connectivity.training_stage.protocol import (
    CIFARView,
    StopFlag,
    StopRequested,
    digest,
    file_hash,
    load,
    mixed_seed,
    restore_rng,
    rng_state,
    save,
    seed_all,
    write_json,
)


def root(cfg):
    return Path(cfg["output_root"]).resolve()


def protocol_hash(cfg):
    return digest({k: v for k, v in cfg.items() if k not in ("output_root", "data_root")})


@lru_cache(maxsize=1)
def code_hash():
    project = Path(__file__).resolve().parents[3]
    files = sorted((project / "src/mode_connectivity/nonlinear_stage").glob("*.py"))
    files += [
        project / "src/mode_connectivity/training_stage/geometry.py",
        project / "external/sinkhorn-rebasin/examples/models/vgg.py",
    ]
    return digest({str(p.relative_to(project)): file_hash(p) for p in files})


def source_pairs(cfg):
    result = []
    for rep, item in enumerate(cfg["source_pairs"]):
        result.append(
            dict(
                replicate=rep,
                seeds=[int(s) for s in item["seeds"]],
                root=str(Path(item["root"]).resolve()),
            )
        )
    return result


def endpoint_path(cfg, replicate, seed, epoch):
    source = source_pairs(cfg)[replicate]
    if seed not in source["seeds"]:
        raise ValueError(f"Seed {seed} is not in source pair {replicate}.")
    return Path(source["root"]) / "endpoints" / str(seed) / f"epoch_{epoch:03d}.pt"


def primary_pairs(cfg):
    """Return the 56 unique path definitions per replicate."""
    stages, final, pairs = cfg["stage_epochs"], int(cfg["final_epoch"]), []
    for source in source_pairs(cfg):
        rep, (a, b) = source["replicate"], source["seeds"]

        def add(kind, n, left_seed, left_epoch, right_seed, right_epoch):
            index = len([p for p in pairs if p["replicate"] == rep])
            pairs.append(
                dict(
                    id=f"r{rep}_{kind}_{n:03d}",
                    replicate=rep,
                    index=index,
                    kind=kind,
                    n=int(n),
                    left_seed=int(left_seed),
                    left_epoch=int(left_epoch),
                    right_seed=int(right_seed),
                    right_epoch=int(right_epoch),
                )
            )

        for n in stages:
            add("same", n, a, n, b, n)
        for n in stages[:-1]:
            add("final_left", n, a, final, b, n)
        for n in stages[:-1]:
            add("final_right", n, a, n, b, final)
        for n in stages[:-1]:
            add("within_a", n, a, n, a, final)
        for n in stages[:-1]:
            add("within_b", n, b, n, b, final)
    return pairs


def pair_by_id(cfg, pair_id):
    return next(p for p in primary_pairs(cfg) if p["id"] == pair_id)


def pilot_pair_ids(cfg):
    final = int(cfg["final_epoch"])
    wanted = {
        ("same", 0),
        ("same", 1),
        ("same", 60),
        ("same", final),
        ("final_left", 1),
        ("final_right", 1),
        ("within_a", 1),
    }
    return [
        p["id"]
        for p in primary_pairs(cfg)
        if p["replicate"] == 0 and (p["kind"], p["n"]) in wanted
    ]


def path_dir(cfg, pair_id, family="bezier", restart=0):
    return root(cfg) / "paths" / pair_id / f"{family}_r{int(restart)}"


def selected_dir(cfg, pair_id):
    return root(cfg) / "selected" / pair_id


TRAINING_PROTOCOL_KEYS = (
    "model",
    "training_recipe",
    "data_recipe",
    "train_full_data",
    "augmentation_seed_mode",
    "split_seed",
    "epochs",
    "train_batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "lr_step",
    "validation_size",
)

TRAINING_PROTOCOL_DEFAULTS = {
    # Older training-stage manifests predate these explicit fields. Their
    # training code used the same defaults through cfg.get(..., default).
    "training_recipe": "vgg_cifar10",
    "data_recipe": "vgg_cifar10",
    "train_full_data": False,
    "augmentation_seed_mode": "run",
}


def _training_protocol_value(source_cfg, key):
    value = source_cfg.get(key, TRAINING_PROTOCOL_DEFAULTS.get(key))
    if value is None and key in TRAINING_PROTOCOL_DEFAULTS:
        return TRAINING_PROTOCOL_DEFAULTS[key]
    return value


def _source_record(
    cfg, source, reference_subsets=None, reference_source_cfg=None
):
    source_root = Path(source["root"])
    protocol_path, subsets_path = source_root / "protocol.json", source_root / "subsets.json"
    if not protocol_path.exists() or not subsets_path.exists():
        raise FileNotFoundError(f"Missing training-stage protocol under {source_root}")
    source_protocol = json.loads(protocol_path.read_text())
    source_cfg = source_protocol["config"]
    if source_cfg.get("model") != "VGG11" or int(source_cfg.get("epochs", -1)) != 200:
        raise ValueError(f"Source {source_root} is not the required 200-epoch VGG11 run.")
    if not set(cfg["stage_epochs"]).issubset(source_cfg.get("checkpoints", [])):
        raise ValueError(f"Source {source_root} lacks required checkpoints.")
    subsets = json.loads(subsets_path.read_text())
    if any(digest(v) != subsets["hashes"][k] for k, v in subsets["indices"].items()):
        raise ValueError(f"Corrupt subset indices in {source_root}")
    if reference_source_cfg is not None:
        mismatched = {
            key: (
                _training_protocol_value(reference_source_cfg, key),
                _training_protocol_value(source_cfg, key),
            )
            for key in TRAINING_PROTOCOL_KEYS
            if _training_protocol_value(reference_source_cfg, key)
            != _training_protocol_value(source_cfg, key)
        }
        if mismatched:
            raise ValueError(
                "Source endpoint training protocols differ: "
                + ", ".join(
                    f"{key}={left!r}/{right!r}"
                    for key, (left, right) in mismatched.items()
                )
            )
    if reference_subsets is not None:
        # The first source root owns the canonical fitting and evaluation
        # subsets for this experiment. Other endpoint-only roots may contain
        # different analysis-subset metadata, but their models must have seen
        # exactly the same training/validation partition.
        for name in ("train", "validation"):
            if subsets["indices"][name] != reference_subsets["indices"][name]:
                raise ValueError(
                    f"Source pairs use different frozen {name} indices; "
                    "the endpoints are not training-protocol matched."
                )
    endpoints = []
    for seed in source["seeds"]:
        for epoch in cfg["stage_epochs"]:
            path = endpoint_path(cfg, source["replicate"], seed, epoch)
            if not path.exists():
                raise FileNotFoundError(path)
            endpoints.append(
                dict(seed=seed, epoch=epoch, path=str(path), sha256=file_hash(path))
            )
    return subsets, dict(
        **source,
        source_protocol_hash=source_protocol.get("hash"),
        source_code_hash=source_protocol.get("code_hash"),
        source_config=source_cfg,
        source_subset_hashes=subsets["hashes"],
        endpoints=endpoints,
    )


def prepare(cfg):
    destination = root(cfg)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists():
        verify_protocol(cfg)
        return
    records, subsets, reference_source_cfg = [], None, None
    for source in source_pairs(cfg):
        current, record = _source_record(
            cfg, source, subsets, reference_source_cfg
        )
        subsets = current if subsets is None else subsets
        reference_source_cfg = (
            record["source_config"]
            if reference_source_cfg is None
            else reference_source_cfg
        )
        records.append(record)
    selection = set(subsets["indices"]["selection"])
    validation_audit = [i for i in subsets["indices"]["validation"] if i not in selection]
    if len(validation_audit) != 4000:
        raise ValueError("Expected 4,000 validation examples outside selection.")
    subsets["indices"]["validation_audit"] = validation_audit
    subsets["hashes"]["validation_audit"] = digest(validation_audit)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "subsets.json", subsets)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    write_json(
        protocol_path,
        dict(
            hash=protocol_hash(cfg),
            code_hash=code_hash(),
            revision=revision,
            torch_version=torch.__version__,
            config=cfg,
            subset_hashes=subsets["hashes"],
            sources=records,
        ),
    )


def verify_protocol(cfg):
    record = json.loads((root(cfg) / "protocol.json").read_text())
    if record["hash"] != protocol_hash(cfg):
        raise ValueError("Nonlinear protocol changed; use a new output_root.")
    if record["code_hash"] != code_hash():
        raise ValueError("Experiment code changed after preparation; use a new output_root.")
    subsets = json.loads((root(cfg) / "subsets.json").read_text())
    if record["subset_hashes"] != subsets["hashes"]:
        raise ValueError("Frozen subset hashes changed.")
    if any(digest(v) != subsets["hashes"][k] for k, v in subsets["indices"].items()):
        raise ValueError("Frozen subset contents changed.")
    return record, subsets


class Data:
    def __init__(self, cfg, allow_test=False):
        self.cfg, self.allow_test = cfg, allow_test
        _, self.subsets = verify_protocol(cfg)
        self.bases = {}

    def loader(self, name, *, augment=False, shuffle=False, batch_size=None, order_seed=None):
        from torchvision import datasets, transforms

        is_test = name.startswith("test_")
        if is_test and not self.allow_test:
            raise ValueError("Test access is frozen until final path selection.")
        if is_test not in self.bases:
            self.bases[is_test] = datasets.CIFAR10(
                self.cfg["data_root"], train=not is_test, download=False
            )
        transform = transforms.Compose(
            (
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
                if augment
                else []
            )
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        dataset = CIFARView(self.bases[is_test], self.subsets["indices"][name], transform)
        sampler = None
        generator = torch.Generator().manual_seed(int(order_seed or 0))
        if order_seed is not None:
            sampler = torch.randperm(len(dataset), generator=generator).tolist()
            shuffle = False
        workers = int(self.cfg["workers"])
        return DataLoader(
            dataset,
            batch_size=batch_size or self.cfg["eval_batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            num_workers=workers,
            pin_memory=str(self.cfg["device"]).startswith("cuda"),
            persistent_workers=False,
            **({"prefetch_factor": 1} if workers else {}),
        )


def endpoint_metadata(cfg, pair):
    record, _ = verify_protocol(cfg)
    lookup = {
        (s["replicate"], e["seed"], e["epoch"]): e
        for s in record["sources"]
        for e in s["endpoints"]
    }
    return [
        lookup[(pair["replicate"], pair["left_seed"], pair["left_epoch"])],
        lookup[(pair["replicate"], pair["right_seed"], pair["right_epoch"])],
    ]


def seed_for(cfg, pair_id, family, restart):
    # NumPy's legacy global RNG accepts only uint32 seeds; keep one seed valid
    # for Python, NumPy, torch CPU, and torch CUDA recovery.
    return mixed_seed(cfg["path_seed"], f"{pair_id}:{family}:{restart}") % (2**32)


__all__ = [
    "Data", "StopFlag", "StopRequested", "code_hash", "digest", "endpoint_metadata",
    "endpoint_path", "file_hash", "load", "mixed_seed", "pair_by_id", "path_dir",
    "pilot_pair_ids", "prepare", "primary_pairs", "protocol_hash", "restore_rng",
    "rng_state", "root", "save", "seed_all", "seed_for", "selected_dir",
    "source_pairs", "verify_protocol", "write_json",
]
