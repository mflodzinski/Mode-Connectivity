"""Operation dispatch, locking, completion records, and resource telemetry."""

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

from .evaluation import (
    dense_audit,
    endpoint_test,
    evaluate_test,
    evaluate_validation,
    freeze_selection,
    pilot_gate,
    screen,
)
from .protocol import (
    StopFlag,
    StopRequested,
    code_hash,
    pair_by_id,
    path_dir,
    prepare,
    protocol_hash,
    root,
    selected_dir,
    verify_protocol,
    write_json,
)


def validate_config(cfg):
    expected = [0, 1, 2, 3, 5, 10, 20, 30, 60, 100, 150, 200]
    if cfg["model"] != "VGG11" or cfg["final_epoch"] != 200:
        raise ValueError("The frozen nonlinear-stage protocol requires VGG11 at epoch 200.")
    if cfg["stage_epochs"] != expected:
        raise ValueError(f"stage_epochs must be {expected}")
    if len(cfg["source_pairs"]) not in (1, 3):
        raise ValueError("Use one pilot source pair or all three final source pairs.")
    seeds = []
    for source in cfg["source_pairs"]:
        if len(source["seeds"]) != 2:
            raise ValueError("Every source must contain exactly two seeds.")
        seeds += list(source["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Source seeds must be unique.")
    for key in [
        "fit_batch_size", "fit_passes", "validation_interval", "selection_points",
        "eval_points", "dense_points", "eval_batch_size",
    ]:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if cfg["fit_passes"] % cfg["validation_interval"]:
        raise ValueError("fit_passes must be divisible by validation_interval.")
    if cfg["workers"] not in (0, 1):
        raise ValueError("The 2 CPU / 4 GB protocol permits zero or one loader worker.")
    if cfg["restart_noise"] != [0.0, 0.01, 0.05]:
        raise ValueError("Restart noise levels are frozen by the protocol.")


def expected_outputs(cfg, task):
    op = task["operation"]
    if op == "prepare":
        return [root(cfg) / "protocol.json", root(cfg) / "subsets.json"]
    if op == "fit":
        directory = path_dir(cfg, task["pair_id"], task["family"], task["restart"])
        return [directory / "path.pt", directory / "history.json"]
    if op == "validate":
        return [
            path_dir(cfg, task["pair_id"], task["family"], task["restart"])
            / "validation.json"
        ]
    if op == "audit":
        return [root(cfg) / "audit/dense_discrepancies.json"]
    if op == "screen":
        return [root(cfg) / "confirmation_targets.json"]
    if op == "freeze":
        return [selected_dir(cfg, task["pair_id"]) / "selection.json"]
    if op == "test":
        return [selected_dir(cfg, task["pair_id"]) / "test.json"]
    if op == "endpoint_test":
        return [root(cfg) / "endpoint_test" / f"replicate_{task['replicate']}.json"]
    if op == "report":
        return [
            root(cfg) / "report/summary.json",
            root(cfg) / "report/results.csv",
            root(cfg) / "report/same_stage.png",
            root(cfg) / "report/cross_stage.png",
            root(cfg) / "report/within_run.png",
            root(cfg) / "report/absolute_profiles.pdf",
            root(cfg) / "report/path_geometry.png",
            root(cfg) / "report/barrier_reduction.png",
            root(cfg) / "report/restart_results.png",
            root(cfg) / "report/restart_results.json",
            root(cfg) / "report/initialization_controls.json",
            root(cfg) / "report/runtime.csv",
            root(cfg) / "report/issues.csv",
        ]
    return []


def completed(cfg, task):
    path = root(cfg) / "status" / f"{task['id']}.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return (
        value.get("status") == "complete"
        and value.get("protocol_hash") == protocol_hash(cfg)
        and all(Path(p).exists() for p in value.get("outputs", []))
    )


def dispatch(cfg, task, stop):
    op = task["operation"]
    if op == "prepare":
        prepare(cfg)
        return {}
    verify_protocol(cfg)
    if op == "smoke":
        from .checks import smoke
        return smoke(cfg)
    if op == "fit":
        from .fitting import fit
        return fit(
            cfg, pair_by_id(cfg, task["pair_id"]), task["family"],
            task["restart"], task["noise"], stop,
        )
    if op == "validate":
        return evaluate_validation(
            cfg, pair_by_id(cfg, task["pair_id"]), task["family"], task["restart"], stop
        )
    if op == "audit":
        return dense_audit(cfg, stop)
    if op == "gate":
        return pilot_gate(cfg)
    if op == "screen":
        return screen(cfg)
    if op == "freeze":
        return freeze_selection(cfg, pair_by_id(cfg, task["pair_id"]))
    if op == "selection_gate":
        return {"frozen": len([p for p in (root(cfg) / "selected").glob("*/selection.json")])}
    if op == "test":
        return evaluate_test(cfg, pair_by_id(cfg, task["pair_id"]), stop)
    if op == "endpoint_test":
        return endpoint_test(cfg, task["replicate"])
    if op == "report":
        from .reporting import report
        return report(cfg)
    raise ValueError(op)


def run_task(cfg, task, all_tasks):
    validate_config(cfg)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    status_dir = root(cfg) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{task['id']}.json"
    with (status_dir / f"{task['id']}.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(cfg, task):
            print(f"Already complete: {task['id']}")
            return
        by_id = {item["id"]: item for item in all_tasks}
        for dependency in task["dependencies"]:
            if dependency not in by_id or not completed(cfg, by_id[dependency]):
                raise RuntimeError(f"Unfinished prerequisite: {dependency}")
        stop, started = StopFlag(), time.monotonic()
        attempts = (
            json.loads(status_path.read_text()).get("attempts", [])
            if status_path.exists() else []
        )
        record = dict(
            status="running", task=task, protocol_hash=protocol_hash(cfg), attempts=attempts
        )
        write_json(status_path, record)
        gpu = str(cfg["device"]).startswith("cuda") and task["operation"] not in {
            "prepare", "gate", "screen", "freeze", "selection_gate", "report"
        }
        try:
            if gpu:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable; use device=cpu only for local checks.")
                torch.cuda.reset_peak_memory_stats()
            result = dispatch(cfg, task, stop)
            stop.check()
            outputs = expected_outputs(cfg, task)
            if any(not path.exists() for path in outputs):
                raise RuntimeError("Operation returned without every expected output.")
            record.update(status="complete", result=result, outputs=[str(p) for p in outputs])
        except BaseException as exc:
            record.update(
                status="incomplete" if isinstance(exc, StopRequested) else "failed",
                error=str(exc), traceback=traceback.format_exc(),
            )
            raise
        finally:
            scale = 1 if sys.platform == "darwin" else 1024
            own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
            children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
            attempts.append(
                dict(
                    seconds=time.monotonic() - started,
                    status=record["status"],
                    code_hash=code_hash(),
                    torch_version=torch.__version__,
                    hostname=os.uname().nodename,
                    host_peak_bytes_upper_bound=own + children,
                    process_peak_bytes=own,
                    worker_peak_bytes=children,
                    gpu_peak_allocated_bytes=(
                        torch.cuda.max_memory_allocated() if gpu and torch.cuda.is_available() else 0
                    ),
                    gpu_peak_reserved_bytes=(
                        torch.cuda.max_memory_reserved() if gpu and torch.cuda.is_available() else 0
                    ),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                )
            )
            write_json(status_path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    tasks, cfg = manifest["tasks"], manifest["config"]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        matches = [
            task for task in tasks
            if task.get("array_group") == args.group
            and task.get("replicate") == args.replicate
            and task.get("index") == index
        ]
    else:
        matches = [task for task in tasks if task["id"] == args.group]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one task for {args.group}, found {len(matches)}")
    run_task(cfg, matches[0], tasks)


if __name__ == "__main__":
    main()
