"""Validation-only Sinkhorn grid search over frozen training-stage endpoints.

This runner deliberately lives outside ``mode_connectivity.training_stage`` so
adding the sweep does not change the source hash of an already frozen protocol.
Each grid point gets a fresh protocol root that reuses the source split and
endpoint checkpoints.  Sinkhorn is selected first, then Sinkhorn+scale starts
from that globally selected hard permutation.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from itertools import product
from pathlib import Path
import statistics

import torch

from mode_connectivity.training_stage.alignment import optimize
from mode_connectivity.training_stage.protocol import (
    StopFlag,
    checkpoint,
    code_hash,
    digest,
    load,
    pair_dir,
    protocol_hash,
    verify_protocol,
    write_json,
)
from mode_connectivity.training_stage.runner import validate_config


def _numbers(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive numbers"
        )
    return values


def _slug(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def base_combinations(args) -> list[dict]:
    return [
        dict(base_lr=lr, tau=tau, sinkhorn_l=args.sinkhorn_l)
        for lr, tau in product(args.base_lrs, args.taus)
    ]


def scale_combinations(args) -> list[dict]:
    return [
        dict(scale_lr=lr, lambda_scale=regularizer)
        for lr, regularizer in product(args.scale_lrs, args.lambda_scales)
    ]


def combo_tag(phase: str, combo: dict) -> str:
    if phase == "base":
        return "lr{}_tau{}_l{}".format(
            _slug(combo["base_lr"]),
            _slug(combo["tau"]),
            _slug(combo["sinkhorn_l"]),
        )
    return "lr{}_lambda{}".format(
        _slug(combo["scale_lr"]), _slug(combo["lambda_scale"])
    )


def source_config(source_root: Path) -> dict:
    source_root = source_root.resolve()
    protocol_path = source_root / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Missing frozen training-stage protocol: {protocol_path}"
        )
    cfg = json.loads(protocol_path.read_text())["config"]
    cfg["output_root"] = str(source_root)
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    validate_config(cfg)
    # The point of this job is to rerun alignment with the current code and new
    # parameters, so the source protocol's historical code hash need not equal
    # the current checkout.  Still verify its immutable config and subset data
    # before trusting any checkpoint paths.
    record = json.loads(protocol_path.read_text())
    if record["hash"] != protocol_hash(cfg):
        raise ValueError(
            f"Source protocol config failed integrity verification: {protocol_path}"
        )
    subsets_path = source_root / "subsets.json"
    subsets = json.loads(subsets_path.read_text())
    if any(
        digest(value) != subsets["hashes"][name]
        for name, value in subsets["indices"].items()
    ):
        raise ValueError(
            f"Source subset indices failed integrity verification: {subsets_path}"
        )
    if record["subset_hashes"] != subsets["hashes"]:
        raise ValueError(
            f"Source subset hashes do not match the protocol: {subsets_path}"
        )
    # Older VGG manifests predate the explicit recipe labels; absence means the
    # original VGG/CIFAR-10 path in this package.
    if (
        cfg["model"] != "VGG11"
        or cfg.get("data_recipe", "vgg_cifar10") != "vgg_cifar10"
    ):
        raise ValueError(
            "This sweep requires the frozen VGG11/CIFAR-10 training-stage run."
        )
    for seeds in cfg["seed_pairs"]:
        for seed in seeds:
            for epoch in cfg["stages"]:
                path = checkpoint(cfg, seed, epoch)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing trained endpoint checkpoint: {path}"
                    )
    return cfg


def pairs(cfg: dict) -> list[tuple[int, int, int]]:
    return [
        (replicate, left, right)
        for replicate in range(len(cfg["seed_pairs"]))
        for left in cfg["stages"]
        for right in cfg["stages"]
    ]


def run_root(sweep_root: Path, phase: str, combo: dict) -> Path:
    return sweep_root.resolve() / phase / combo_tag(phase, combo)


def configured(source: dict, destination: Path, overrides: dict) -> dict:
    cfg = copy.deepcopy(source)
    cfg.update(overrides)
    cfg["output_root"] = str(destination.resolve())
    return cfg


def prepare_run_root(source: dict, destination: Path, overrides: dict) -> dict:
    """Create a protocol root without copying or retraining endpoint checkpoints."""
    destination.mkdir(parents=True, exist_ok=True)
    cfg = configured(source, destination, overrides)
    validate_config(cfg)
    lock_path = destination / ".prepare.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        endpoints = destination / "endpoints"
        source_endpoints = Path(source["output_root"]).resolve() / "endpoints"
        if not endpoints.exists():
            endpoints.symlink_to(source_endpoints, target_is_directory=True)
        elif endpoints.resolve() != source_endpoints:
            raise ValueError(f"{endpoints} does not point to {source_endpoints}")

        source_subsets = Path(source["output_root"]).resolve() / "subsets.json"
        subset_data = json.loads(source_subsets.read_text())
        destination_subsets = destination / "subsets.json"
        if destination_subsets.exists():
            if json.loads(destination_subsets.read_text()) != subset_data:
                raise ValueError(f"Frozen subsets differ in {destination}")
        else:
            write_json(destination_subsets, subset_data)

        record = dict(
            hash=protocol_hash(cfg),
            config=cfg,
            subset_hashes=subset_data["hashes"],
            code_hash=code_hash(),
            source_protocol=str(
                Path(source["output_root"]).resolve() / "protocol.json"
            ),
            purpose="validation-only alignment hyperparameter sweep",
            torch_version=torch.__version__,
        )
        protocol_path = destination / "protocol.json"
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text())
            if (
                existing["hash"] != record["hash"]
                or existing["code_hash"] != record["code_hash"]
            ):
                raise ValueError(
                    f"Sweep protocol changed in {destination}; use a new sweep root."
                )
        else:
            write_json(protocol_path, record)
    verify_protocol(cfg)
    return cfg


def selected_base(sweep_root: Path) -> dict:
    path = sweep_root.resolve() / "selected_base.json"
    if not path.exists():
        raise FileNotFoundError(f"Base selection has not completed: {path}")
    return json.loads(path.read_text())["selected"]


def task_index(args) -> int:
    value = args.task_id
    if value is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("Set --task-id or run inside a Slurm array task.")
    return int(value)


def configure_torch(cfg: dict) -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def run_phase(args, phase: str) -> None:
    source = source_config(args.source_root)
    pair_grid = pairs(source)
    combos = base_combinations(args) if phase == "base" else scale_combinations(args)
    index = task_index(args)
    if index < 0 or index >= len(combos) * len(pair_grid):
        raise IndexError(
            f"Task {index} is outside the {len(combos) * len(pair_grid)}-task {phase} grid."
        )
    combo = combos[index // len(pair_grid)]
    replicate, left, right = pair_grid[index % len(pair_grid)]

    if phase == "base":
        overrides = combo
    else:
        base = selected_base(args.sweep_root)
        overrides = {**base["combo"], **combo}
    destination = run_root(args.sweep_root, phase, combo)
    cfg = prepare_run_root(source, destination, overrides)
    configure_torch(cfg)

    directory = pair_dir(cfg, replicate, left, right)
    directory.mkdir(parents=True, exist_ok=True)
    if phase == "scale":
        base = selected_base(args.sweep_root)
        source_artifact = (
            Path(base["root"])
            / "pairs"
            / str(replicate)
            / f"{left:03d}_{right:03d}"
            / "base.pt"
        )
        if not source_artifact.exists():
            raise FileNotFoundError(
                f"Selected base artifact is missing: {source_artifact}"
            )
        destination_artifact = directory / "base.pt"
        if not destination_artifact.exists():
            destination_artifact.symlink_to(source_artifact.resolve())
        elif destination_artifact.resolve() != source_artifact.resolve():
            raise ValueError(f"Unexpected base artifact at {destination_artifact}")

    artifact = directory / f"{phase}.pt"
    status_path = (
        destination / "grid_status" / f"{replicate}_{left:03d}_{right:03d}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            artifact.exists()
            and status_path.exists()
            and json.loads(status_path.read_text()).get("status") == "complete"
        ):
            print(f"Already complete: {phase} {combo} pair={(replicate, left, right)}")
            return
        write_json(
            status_path,
            dict(
                status="running",
                phase=phase,
                combo=combo,
                pair=[replicate, left, right],
            ),
        )
        try:
            result = optimize(cfg, replicate, left, right, phase, StopFlag())
            write_json(
                status_path,
                dict(
                    status="complete",
                    phase=phase,
                    combo=combo,
                    pair=[replicate, left, right],
                    artifact=str(artifact),
                    result=result,
                ),
            )
        except BaseException as exc:
            write_json(
                status_path,
                dict(
                    status="failed",
                    phase=phase,
                    combo=combo,
                    pair=[replicate, left, right],
                    error=repr(exc),
                ),
            )
            raise


def rank_combinations(args, phase: str) -> dict:
    source = source_config(args.source_root)
    pair_grid = pairs(source)
    combos = base_combinations(args) if phase == "base" else scale_combinations(args)
    rows = []
    for combo in combos:
        destination = run_root(args.sweep_root, phase, combo)
        scores = []
        for replicate, left, right in pair_grid:
            artifact_path = (
                destination
                / "pairs"
                / str(replicate)
                / f"{left:03d}_{right:03d}"
                / f"{phase}.pt"
            )
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Incomplete {phase} grid; missing {artifact_path}"
                )
            score = load(artifact_path).get("score")
            if score is None or len(score) < 2:
                raise ValueError(f"Missing validation score in {artifact_path}")
            scores.append([float(score[0]), float(score[1])])
        row = dict(
            combo=combo,
            tag=combo_tag(phase, combo),
            root=str(destination),
            pair_count=len(scores),
            mean_validation_chord=statistics.fmean(score[0] for score in scores),
            mean_validation_curve_loss=statistics.fmean(score[1] for score in scores),
            max_validation_chord=max(score[0] for score in scores),
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["mean_validation_chord"],
            row["mean_validation_curve_loss"],
            row["max_validation_chord"],
            row["tag"],
        )
    )
    report = dict(
        phase=phase,
        selection_data="selection subset only; no test data",
        ranking_order=[
            "mean_validation_chord",
            "mean_validation_curve_loss",
            "max_validation_chord",
        ],
        selected=rows[0],
        candidates=rows,
    )
    write_json(args.sweep_root.resolve() / f"selected_{phase}.json", report)
    return report


def materialize_selected(args, scale_report: dict) -> None:
    source = source_config(args.source_root)
    base = selected_base(args.sweep_root)
    scale = scale_report["selected"]
    destination = args.sweep_root.resolve() / "selected"
    cfg = prepare_run_root(source, destination, {**base["combo"], **scale["combo"]})
    for replicate, left, right in pairs(source):
        target_dir = pair_dir(cfg, replicate, left, right)
        target_dir.mkdir(parents=True, exist_ok=True)
        for phase, selection in [("base", base), ("scale", scale)]:
            source_path = (
                Path(selection["root"])
                / "pairs"
                / str(replicate)
                / f"{left:03d}_{right:03d}"
                / f"{phase}.pt"
            )
            target_path = target_dir / f"{phase}.pt"
            if not target_path.exists():
                target_path.symlink_to(source_path.resolve())
            elif target_path.resolve() != source_path.resolve():
                raise ValueError(f"Unexpected selected artifact at {target_path}")
    write_json(
        destination / "selection.json",
        dict(
            source_root=str(args.source_root.resolve()),
            selected_base=base,
            selected_scale=scale,
            note="Hyperparameters were selected globally over all stage pairs using validation-only chord and mean loss.",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["count", "base", "select-base", "scale", "select-scale"]
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["base", "scale"])
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--base-lrs", type=_numbers, default=_numbers("0.005,0.01,0.05")
    )
    parser.add_argument("--taus", type=_numbers, default=_numbers("1.0,1.5"))
    parser.add_argument("--sinkhorn-l", type=float, default=1.0)
    parser.add_argument("--scale-lrs", type=_numbers, default=_numbers("0.01,0.05"))
    parser.add_argument(
        "--lambda-scales", type=_numbers, default=_numbers("0.0001,0.001,0.01")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sinkhorn_l <= 0:
        raise ValueError("--sinkhorn-l must be positive.")
    if args.operation == "count":
        if args.phase is None:
            raise ValueError("count requires --phase.")
        source = source_config(args.source_root)
        combos = (
            base_combinations(args)
            if args.phase == "base"
            else scale_combinations(args)
        )
        print(len(pairs(source)) * len(combos))
    elif args.operation in {"base", "scale"}:
        run_phase(args, args.operation)
    elif args.operation == "select-base":
        report = rank_combinations(args, "base")
        print(json.dumps(report["selected"], indent=2))
    else:
        report = rank_combinations(args, "scale")
        materialize_selected(args, report)
        print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
