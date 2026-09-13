"""Operation dispatch, task locking, completion markers, and resource telemetry."""

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
    StopFlag,
    StopRequested,
    root,
    protocol_hash,
    verify_protocol,
    write_json,
    checkpoint,
    pair_dir,
    prepare,
    code_hash,
)
from .scheduling import build_dag, completed, prerequisite_tasks


def validate_config(cfg):
    if cfg["stages"] != sorted(set(cfg["stages"])) or len(cfg["stages"]) < 2:
        raise ValueError("stages must contain at least two distinct ordered epochs.")
    if cfg["stages"][-1] != cfg["epochs"] or not set(cfg["stages"]).issubset(
        cfg["checkpoints"]
    ):
        raise ValueError("Main stages must be saved, and the last stage must be final.")
    if 0 not in cfg["checkpoints"] or max(cfg["checkpoints"]) > cfg["epochs"]:
        raise ValueError(
            "Save initialization and do not request checkpoints beyond training."
        )
    if cfg["checkpoints"] != sorted(set(cfg["checkpoints"])):
        raise ValueError("checkpoints must be distinct and ordered.")
    seeds = sum(cfg["seed_pairs"], [])
    if len(seeds) != len(set(seeds)) or any(len(p) != 2 for p in cfg["seed_pairs"]):
        raise ValueError("Seed pairs must be independent and contain two seeds each.")
    for key in [
        "epochs",
        "base_passes",
        "branch_passes",
        "validation_interval",
        "patience",
        "alignment_batch_size",
        "train_batch_size",
        "eval_batch_size",
        "lr_step",
    ]:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if cfg["workers"] not in (0, 1):
        raise ValueError(
            "The 2 CPU / 4 GB protocol supports zero or one loader worker."
        )
    subset_sizes = [
        cfg[k]
        for k in [
            "validation_size",
            "alignment_size",
            "train_eval_size",
            "selection_size",
            "test_eval_size",
        ]
    ]
    if any(size <= 0 or size % 10 for size in subset_sizes):
        raise ValueError(
            "Every fixed subset must have a positive, class-balanced size."
        )
    if (
        cfg["validation_size"] >= 50000
        or cfg["selection_size"] > cfg["validation_size"]
    ):
        raise ValueError(
            "The validation and selection sizes do not fit CIFAR-10 training data."
        )
    training_size = 50000 - cfg["validation_size"]
    if cfg["alignment_size"] + cfg["train_eval_size"] > training_size:
        raise ValueError("Alignment and training-evaluation subsets must be disjoint.")
    if cfg["test_eval_size"] > 10000:
        raise ValueError("The sampled test subset exceeds CIFAR-10 test data.")
    if cfg["alignment_size"] % cfg["alignment_batch_size"]:
        raise ValueError(
            "alignment_size must be divisible by alignment_batch_size for fixed budgets."
        )
    if cfg["eval_points"] < 2 or cfg["audit_points"] < 2:
        raise ValueError("Evaluation needs both endpoints.")
    grid = cfg["validation_alphas"]
    if grid != sorted(set(grid)) or grid[0] != 0 or grid[-1] != 1:
        raise ValueError("Validation grid must be ordered and include endpoints.")


def expected_outputs(cfg, task):
    operation = task["operation"]
    if operation == "prepare":
        return [root(cfg) / "protocol.json", root(cfg) / "subsets.json"]
    if operation == "train":
        directory = root(cfg) / "endpoints" / str(task["seed"])
        return [directory / "history.json", directory / "recovery.pt"] + [
            checkpoint(cfg, task["seed"], e) for e in cfg["checkpoints"]
        ]
    if operation in ("base", "scale", "continue", "evaluate"):
        directory = pair_dir(cfg, task["replicate"], task["ea"], task["eb"])
        names = (
            ["base.pt", "wm.pt"]
            if operation == "base"
            else ["profiles.json" if operation == "evaluate" else operation + ".pt"]
        )
        return [directory / name for name in names]
    if operation == "controls":
        directory = root(cfg) / "controls" / str(task["replicate"])
        return (
            [directory / f"diagonal_{e:03d}.json" for e in cfg["checkpoints"]]
            + [
                directory / f"within_{s}_{e:03d}.json"
                for s in cfg["seed_pairs"][task["replicate"]]
                for e in cfg["stages"][:-1]
            ]
            + [
                directory / f"endpoint_{s}_full_test.json"
                for s in cfg["seed_pairs"][task["replicate"]]
            ]
        )
    if operation == "audit":
        return [root(cfg) / "audit/discrepancies.json"]
    if operation == "report":
        return [
            root(cfg) / "report/summary.json",
            root(cfg) / "report/aggregates.json",
            root(cfg) / "report/runtime.csv",
            root(cfg) / "report/absolute_profiles.pdf",
            root(cfg) / "report/final_endpoints_full_test.json",
        ]
    return []


def gate(cfg):
    pilot = prerequisite_tasks(build_dag(cfg), "gate")
    checked = []
    for task in pilot:
        if task["id"] == "gate":
            continue
        if not completed(cfg, task):
            raise RuntimeError(f'Pilot incomplete: {task["id"]}')
        status = json.loads((root(cfg) / "status" / f"{task['id']}.json").read_text())
        if any(
            a["host_peak_bytes_upper_bound"] >= 4 * 1024**3 for a in status["attempts"]
        ):
            raise RuntimeError(
                f'Pilot host memory exceeds the 4 GB target: {task["id"]}'
            )
        checked.append(task["id"])
    return dict(
        checked=checked,
        host_memory_limit_bytes=4 * 1024**3,
        note="Process/worker telemetry; consult Slurm accounting for cgroup peak memory.",
    )


def dispatch(cfg, task, stop):
    operation = task["operation"]
    if operation == "prepare":
        prepare(cfg)
        return {}
    verify_protocol(cfg)
    if operation == "smoke":
        from .checks import smoke

        return smoke(cfg)
    if operation == "train":
        from .training import train

        return train(cfg, task["seed"], stop)
    if operation in ("base", "scale", "continue"):
        from .alignment import optimize

        return optimize(cfg, task["replicate"], task["ea"], task["eb"], operation, stop)
    if operation == "evaluate":
        from .evaluation import evaluate_pair

        result = evaluate_pair(cfg, task["replicate"], task["ea"], task["eb"], stop)
        return dict(profiles=len(result))
    if operation == "controls":
        from .evaluation import controls

        return controls(cfg, task["replicate"], stop)
    if operation == "audit":
        from .evaluation import audit

        return audit(cfg, stop)
    if operation == "gate":
        return gate(cfg)
    if operation == "report":
        from .reporting import report

        return report(cfg)
    raise ValueError(operation)


def run_task(cfg, task):
    validate_config(cfg)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # Keep this explicitly FP32, including CUDA matrix multiplication.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = root(cfg) / "status"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task['id']}.json"
    with (directory / f"{task['id']}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(cfg, task):
            print(f"Already complete: {task['id']}")
            return
        by_id = {t["id"]: t for t in build_dag(cfg)}
        for dep in task["dependencies"]:
            if not completed(cfg, by_id[dep]):
                raise RuntimeError(f"Unfinished prerequisite: {dep}")
        stop = StopFlag()
        start = time.monotonic()
        attempts = (
            json.loads(path.read_text()).get("attempts", []) if path.exists() else []
        )
        record = dict(
            status="running",
            task=task,
            protocol_hash=protocol_hash(cfg),
            attempts=attempts,
        )
        write_json(path, record)
        gpu = cfg["device"].startswith("cuda") and task["operation"] not in (
            "prepare",
            "gate",
            "report",
        )
        try:
            if gpu:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA requested but unavailable; use device=cpu only for local checks."
                    )
                torch.cuda.reset_peak_memory_stats()
            result = dispatch(cfg, task, stop)
            stop.check()
            outputs = expected_outputs(cfg, task)
            if any(not p.exists() for p in outputs):
                raise RuntimeError("Operation returned without every expected output.")
            record.update(
                status="complete", result=result, outputs=[str(p) for p in outputs]
            )
        except BaseException as exc:
            record.update(
                status="incomplete" if isinstance(exc, StopRequested) else "failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            scale = 1 if sys.platform == "darwin" else 1024
            own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
            children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
            attempts.append(
                dict(
                    seconds=time.monotonic() - start,
                    status=record["status"],
                    code_hash=code_hash(),
                    torch_version=torch.__version__,
                    hostname=os.uname().nodename,
                    host_peak_bytes_upper_bound=own + children,
                    process_peak_bytes=own,
                    worker_peak_bytes=children,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()
                    if gpu and torch.cuda.is_available()
                    else 0,
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()
                    if gpu and torch.cuda.is_available()
                    else 0,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                )
            )
            write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?")
    parser.add_argument("--manifest")
    parser.add_argument("--operation", dest="manifest_operation")
    parser.add_argument("--selector")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--ea", type=int)
    parser.add_argument("--eb", type=int)
    args, overrides = parser.parse_known_args()
    if args.manifest:
        if overrides:
            parser.error("Manifest tasks cannot override the frozen protocol.")
        manifest = json.loads(Path(args.manifest).read_text())
        cfg = manifest["config"]
        operation = args.manifest_operation
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            task_id = f"{operation}_{args.selector}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        else:
            task_id = args.selector
        task = next(t for t in manifest["tasks"] if t["id"] == task_id)
    else:
        from mode_connectivity.common.hydra_compat import compose_experiment_config
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(
            compose_experiment_config(
                default_config_name="training_stage/default",
                caller_file=__file__,
                argv=overrides,
            ),
            resolve=True,
        )
        candidates = [
            t
            for t in build_dag(cfg)
            if t["operation"] == args.operation
            and t.get("seed", args.seed) == args.seed
            and t.get("replicate", args.replicate) == args.replicate
            and (args.ea is None or t.get("ea") == args.ea)
            and (args.eb is None or t.get("eb") == args.eb)
        ]
        if len(candidates) != 1:
            parser.error(
                "Specify an operation and, for pair tasks, --replicate, --ea and --eb."
            )
        task = candidates[0]
    run_task(cfg, task)


if __name__ == "__main__":
    main()
