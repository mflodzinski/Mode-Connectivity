"""Inventory independent final endpoints without loading large checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGETS = {
    "VGG11/CIFAR-10": ("VGG11", 200),
    "VGG13/CIFAR-10": ("VGG13", 200),
    "VGG16/CIFAR-10": ("VGG16", 200),
    "VGG19/CIFAR-10": ("VGG19", 200),
    "MLP-10x512/Fashion-MNIST": ("FashionMLP10x512", 100),
}


def _config(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("config", value) if isinstance(value, dict) else None


def _model_name(cfg):
    model = cfg.get("model")
    if model in {"VGG11", "VGG13", "VGG16", "VGG19"}:
        return model
    if int(cfg.get("hidden_layers", -1)) == 10 and int(cfg.get("hidden_width", -1)) == 512:
        return "FashionMLP10x512"
    return None


def _seeds(cfg):
    pairs = cfg.get("seed_pairs", [])
    return sorted({int(seed) for pair in pairs for seed in pair})


def _candidate_paths(root: Path, seed: int, epoch: int):
    return (
        root / "endpoints" / str(seed) / f"epoch_{epoch:03d}.pt",
        root / "endpoints" / f"seed{seed}" / f"epoch_{epoch:03d}.pt",
        root / "endpoints" / f"seed{seed}" / "checkpoints" / f"checkpoint-{epoch}.pt",
        root / f"seed{seed}" / "checkpoints" / f"checkpoint-{epoch}.pt",
    )


def _record(root: Path, protocol: Path, cfg, model: str, epoch: int):
    seeds = _seeds(cfg)
    if not seeds:
        seeds = list(range(6))
    found = {}
    for seed in seeds:
        for candidate in _candidate_paths(root, seed, epoch):
            if candidate.is_file() and candidate.stat().st_size > 0:
                resolved = candidate.resolve()
                # Ignore alignment-grid roots whose ``endpoints`` directory is
                # only a symlink to the actual training experiment. The owning
                # training root will be reported separately.
                if root.resolve() not in resolved.parents:
                    continue
                found[str(seed)] = str(resolved)
                break
    subsets_path = root / "subsets.json"
    validation_hash = None
    if subsets_path.exists():
        try:
            validation_hash = json.loads(subsets_path.read_text()).get("hashes", {}).get("validation")
        except (OSError, json.JSONDecodeError):
            pass
    return dict(
        model=model,
        root=str(root.resolve()),
        protocol=str(protocol.resolve()),
        epoch=epoch,
        training_recipe=cfg.get("training_recipe"),
        split_seed=cfg.get("split_seed"),
        validation_hash=validation_hash,
        training_signature={
            key: cfg.get(key)
            for key in (
                "lr", "momentum", "weight_decay", "lr_step",
                "train_batch_size", "validation_size", "epochs",
            )
        },
        configured_seeds=seeds,
        available_seeds=sorted(int(seed) for seed in found),
        checkpoints=found,
        complete_three_disjoint_pairs=all(str(seed) in found for seed in range(6)),
    )


def inventory(project: Path):
    records = []
    results = project / "results"
    if results.exists():
        for protocol in results.rglob("protocol.json"):
            cfg = _config(protocol)
            if not isinstance(cfg, dict):
                continue
            model = _model_name(cfg)
            if model is None:
                continue
            epoch = 100 if model == "FashionMLP10x512" else 200
            row = _record(protocol.parent, protocol, cfg, model, epoch)
            if row["available_seeds"]:
                records.append(row)

    external = project / "external" / "pytorch-vgg-cifar10"
    external_rows = {}
    if external.exists():
        pattern = re.compile(r"save_vgg(11|13|16|19)_seed(\d+)$")
        for directory in external.glob("save_vgg*_seed*"):
            match = pattern.match(directory.name)
            checkpoint = directory / "model_final_state_dict.pth"
            if not match or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                continue
            model, seed = f"VGG{match.group(1)}", int(match.group(2))
            external_rows.setdefault(model, {})[seed] = str(checkpoint.resolve())
    for model, checkpoints in external_rows.items():
        records.append(dict(
            model=model,
            root=str(external.resolve()),
            protocol=None,
            epoch="external final",
            training_recipe="external pytorch-vgg-cifar10",
            split_seed=None,
            validation_hash=None,
            training_signature={},
            configured_seeds=sorted(checkpoints),
            available_seeds=sorted(checkpoints),
            checkpoints={str(seed): path for seed, path in checkpoints.items()},
            complete_three_disjoint_pairs=all(seed in checkpoints for seed in range(6)),
        ))

    # Older Garipov-style endpoint jobs predate protocol.json. Report them so
    # they are not mistaken for missing training, but keep them in a separate
    # recipe group because their checkpoint/model format is not compatible
    # with the maintained training-stage benchmark.
    for depth in (11, 13, 16, 19):
        standard = results / f"vgg{depth}" / "cifar10" / "endpoints" / "standard"
        checkpoints = {}
        for directory in standard.glob("seed[0-9]*") if standard.exists() else ():
            match = re.fullmatch(r"seed(\d+)", directory.name)
            if not match:
                continue
            checkpoint = directory / "checkpoints" / "checkpoint-200.pt"
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                checkpoints[int(match.group(1))] = str(checkpoint.resolve())
        if checkpoints:
            records.append(dict(
                model=f"VGG{depth}", root=str(standard.resolve()), protocol=None,
                epoch=200,
                training_recipe="legacy Garipov standard (unmanifested)",
                split_seed=None, validation_hash=None, training_signature={},
                configured_seeds=sorted(checkpoints),
                available_seeds=sorted(checkpoints),
                checkpoints={str(seed): path for seed, path in checkpoints.items()},
                complete_three_disjoint_pairs=all(seed in checkpoints for seed in range(6)),
            ))
    return sorted(records, key=lambda row: (row["model"], row["root"]))


def summarize(rows):
    groups = {}
    for row in rows:
        signature = json.dumps(
            [
                row["model"], row["epoch"], row["training_recipe"],
                row["split_seed"], row["validation_hash"], row["training_signature"],
            ],
            sort_keys=True,
        )
        # Older VGG11 pair-(0,1) metadata predates the recipe label, but its
        # split and complete optimizer signature match the later seed runs.
        if row["model"] == "VGG11" and row["validation_hash"]:
            signature = json.dumps(
                [row["model"], row["epoch"], row["split_seed"], row["validation_hash"], row["training_signature"]],
                sort_keys=True,
            )
        group = groups.setdefault(signature, dict(
            model=row["model"], epoch=row["epoch"], recipe=row["training_recipe"],
            roots=[], checkpoints={},
        ))
        group["roots"].append(row["root"])
        group["checkpoints"].update(row["checkpoints"])
    result = []
    for group in groups.values():
        group["roots"] = sorted(set(group["roots"]))
        group["available_seeds"] = sorted(int(seed) for seed in group["checkpoints"])
        group["complete_three_disjoint_pairs"] = all(
            str(seed) in group["checkpoints"] for seed in range(6)
        )
        result.append(group)
    return sorted(result, key=lambda row: (row["model"], row["roots"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = inventory(args.project_root.resolve())
    if args.json:
        print(json.dumps({"summary": summarize(rows), "details": rows}, indent=2))
        return
    print("model\tseeds\tthree_pairs\trecipe\troot")
    for row in summarize(rows):
        seeds = ",".join(map(str, row["available_seeds"]))
        complete = "YES" if row["complete_three_disjoint_pairs"] else "no"
        print(f"{row['model']}\t{seeds}\t{complete}\t{row['recipe']}\t{','.join(row['roots'])}")


if __name__ == "__main__":
    main()
