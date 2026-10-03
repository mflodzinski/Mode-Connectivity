"""Run preparation or one independent VGG endpoint-training task."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import torch

from .protocol import (
    StopFlag, StopRequested, checkpoint, code_hash, prepare, protocol_hash,
    root, verify_protocol, write_json,
)
from .scheduling import build_dag, completed


def validate_config(cfg):
    if cfg["model"] not in {"VGG11", "VGG13", "VGG16", "VGG19"}:
        raise ValueError("model must be VGG11, VGG13, VGG16, or VGG19")
    if int(cfg["epochs"]) <= 0 or int(cfg["train_batch_size"]) <= 0:
        raise ValueError("epochs and train_batch_size must be positive")
    checkpoints = [int(value) for value in cfg["checkpoints"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        raise ValueError("checkpoints must be ordered, unique, and include epoch zero")
    if checkpoints[-1] != int(cfg["epochs"]):
        raise ValueError("the final training epoch must be checkpointed")
    seeds = sum(cfg["seed_pairs"], [])
    if len(seeds) != len(set(seeds)) or any(len(pair) != 2 for pair in cfg["seed_pairs"]):
        raise ValueError("seed_pairs must contain six distinct endpoints in three pairs")
    if int(cfg["validation_size"]) >= 50_000:
        raise ValueError("validation_size must leave training examples")
    if int(cfg["workers"]) not in (0, 1):
        raise ValueError("workers must be zero or one for the frozen cluster protocol")


def expected_outputs(cfg, task):
    if task["operation"] == "prepare":
        return [root(cfg) / "protocol.json", root(cfg) / "subsets.json"]
    directory = root(cfg) / "endpoints" / str(task["seed"])
    return [directory / "history.json", directory / "recovery.pt"] + [
        checkpoint(cfg, task["seed"], epoch) for epoch in cfg["checkpoints"]
    ]


def dispatch(cfg, task, stop):
    if task["operation"] == "prepare":
        prepare(cfg)
        return {}
    if task["operation"] == "train":
        verify_protocol(cfg)
        from .training import train
        return train(cfg, task["seed"], stop)
    raise ValueError(task["operation"])


def run_task(cfg, task):
    validate_config(cfg)
    torch.set_num_threads(1)
    directory = root(cfg) / "status"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task['id']}.json"
    with (directory / f"{task['id']}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(cfg, task):
            print(f"Already complete: {task['id']}")
            return
        by_id = {value["id"]: value for value in build_dag(cfg)}
        for dependency in task["dependencies"]:
            if not completed(cfg, by_id[dependency]):
                raise RuntimeError(f"Unfinished prerequisite: {dependency}")
        stop, started = StopFlag(), time.monotonic()
        attempts = json.loads(path.read_text()).get("attempts", []) if path.exists() else []
        record = dict(status="running", task=task, protocol_hash=protocol_hash(cfg), attempts=attempts)
        write_json(path, record)
        gpu = task["operation"] == "train" and str(cfg["device"]).startswith("cuda")
        try:
            if gpu and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            result = dispatch(cfg, task, stop)
            stop.check()
            outputs = expected_outputs(cfg, task)
            if any(not value.exists() for value in outputs):
                raise RuntimeError("Operation finished without every declared output")
            record.update(status="complete", result=result, outputs=[str(value) for value in outputs])
        except BaseException as error:
            record.update(
                status="incomplete" if isinstance(error, StopRequested) else "failed",
                error=str(error), traceback=traceback.format_exc(),
            )
            raise
        finally:
            scale = 1 if sys.platform == "darwin" else 1024
            own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
            children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
            attempts.append(dict(
                seconds=time.monotonic() - started, status=record["status"],
                code_hash=code_hash(), torch_version=torch.__version__, hostname=os.uname().nodename,
                host_peak_bytes_upper_bound=own + children,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu and torch.cuda.is_available() else 0,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            ))
            write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", choices=("prepare", "train"))
    parser.add_argument("--manifest")
    parser.add_argument("--operation", dest="manifest_operation")
    parser.add_argument("--selector")
    parser.add_argument("--seed", type=int)
    args, overrides = parser.parse_known_args()
    if args.manifest:
        if overrides:
            parser.error("Manifest tasks cannot override the frozen protocol")
        manifest = json.loads(Path(args.manifest).read_text())
        cfg = manifest["config"]
        task = next(value for value in manifest["tasks"] if value["id"] == args.selector)
        if task["operation"] != args.manifest_operation:
            parser.error("Manifest operation and selected task disagree")
    else:
        from mode_connectivity.common.hydra_compat import compose_experiment_config
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(compose_experiment_config(
            default_config_name="training_stage/default", caller_file=__file__, argv=overrides,
        ), resolve=True)
        candidates = [
            value for value in build_dag(cfg)
            if value["operation"] == args.operation
            and (args.seed is None or value.get("seed") == args.seed)
        ]
        if len(candidates) != 1:
            parser.error("train requires one configured --seed")
        task = candidates[0]
    run_task(cfg, task)


if __name__ == "__main__":
    main()
