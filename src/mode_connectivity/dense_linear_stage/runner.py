"""Manifest operation runner with recovery-aware status and resource telemetry."""

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
    code_hash,
    prepare,
    protocol_hash,
    root,
    verify_protocol,
    write_json,
)
from .tasks import build_dag


CPU_OPERATIONS = {
    "prepare", "freeze_choices", "endpoints_complete", "report",
}


def validate_config(cfg):
    if cfg.get("benchmark_mode") != "final_alignment":
        raise ValueError("Only the paper's final_alignment benchmark is supported.")
    if cfg["dataset"] not in ("cifar10", "fashion_mnist"):
        raise ValueError("dataset must be cifar10 or fashion_mnist")
    if len(cfg["stages"]) != 1:
        raise ValueError("Exactly one final endpoint epoch is required.")
    if len(cfg["seed_pairs"]) != 3 or any(len(pair) != 2 for pair in cfg["seed_pairs"]):
        raise ValueError("Exactly three independent seed pairs are required.")
    seeds = sum(cfg["seed_pairs"], [])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed pairs must not reuse endpoints.")
    if int(cfg["profile_points_all"]) < 3 or int(cfg["profile_points_selected"]) < int(cfg["profile_points_all"]):
        raise ValueError("Selected profiles must be at least as dense as all-method profiles.")
    for key in (
        "calibration_pairs_per_task", "calibration_evaluation_pairs_per_task",
        "endpoint_pairs_per_task",
    ):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    expected_train = 45000 if cfg["dataset"] == "cifar10" else 55000
    if int(cfg["expected_train_examples"]) != expected_train:
        raise ValueError(f"{cfg['dataset']} must use its full {expected_train}-example endpoint-training split.")
    report_size = int(cfg["train_report_size"])
    if report_size < 1 or report_size > expected_train:
        raise ValueError("train_report_size must be within the endpoint-training split.")
    for method in ("wm_scale", "sinkhorn", "sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
        if method not in cfg["method_hyperparameters"] or not cfg["calibration_candidates"].get(method):
            raise ValueError(f"Missing hyperparameters or calibration candidates for {method}.")
    for method, shards in cfg.get("calibration_candidate_shards", {}).items():
        if method not in cfg["calibration_candidates"]:
            raise ValueError(f"Candidate shards reference unknown method {method}.")
        flattened = [int(index) for shard in shards for index in shard]
        expected = list(range(len(cfg["calibration_candidates"][method])))
        if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
            raise ValueError(
                f"Candidate shards for {method} must cover every candidate exactly once."
            )
        if not 2 <= len(shards) <= 3 or any(not shard for shard in shards):
            raise ValueError(f"{method} must use two or three nonempty grid shards.")


def completed(cfg, task):
    path = root(cfg) / "status" / f"{task['id']}.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return (
        record.get("status") == "complete"
        and record.get("protocol_hash") == protocol_hash(cfg)
        and all(Path(value).exists() for value in record.get("outputs", []))
    )


def dispatch(cfg, task, stop):
    operation = task["operation"]
    if operation == "prepare":
        prepare(cfg)
        return {}, [root(cfg) / "protocol.json", root(cfg) / "subsets.json", root(cfg) / "endpoints.json"]
    verify_protocol(cfg)
    if operation == "calibrate_wm":
        from .alignment import artifact_path
        from .calibration import fit_wm_cell
        item = task["item"]
        rep, left, right = (
            int(item["replicate"]), int(item["left_epoch"]),
            int(item["right_epoch"]),
        )
        result = fit_wm_cell(cfg, rep, left, right, stop)
        return result, [artifact_path(cfg, rep, left, right, "wm")]
    if operation in ("calibrate_base_grid", "calibrate_branch_grid"):
        from .calibration import candidate_record_path, fit_candidate_shard
        item, method = task["item"], task["method"]
        rep, left, right = (
            int(item["replicate"]), int(item["left_epoch"]),
            int(item["right_epoch"]),
        )
        indices = [int(index) for index in task["candidate_indices"]]
        result = fit_candidate_shard(
            cfg, rep, left, right, method, indices, stop
        )
        outputs = [
            candidate_record_path(cfg, rep, left, right, method, index)
            for index in indices
        ]
        return result, outputs
    if operation in ("calibrate_base_select", "calibrate_branch_select"):
        from .calibration import select_base_cell, select_branch_cell
        item = task["item"]
        rep, left, right = (
            int(item["replicate"]), int(item["left_epoch"]),
            int(item["right_epoch"]),
        )
        if operation == "calibrate_base_select":
            result = select_base_cell(cfg, rep, left, right, stop)
            output = root(cfg) / "calibration" / f"r{rep}" / f"{left:03d}_{right:03d}" / "base_choice.json"
        else:
            result = select_branch_cell(cfg, rep, left, right, stop)
            output = root(cfg) / "calibration" / f"r{rep}" / f"{left:03d}_{right:03d}" / "choice.json"
        return result, [output]
    if operation == "freeze_choices":
        from .calibration import freeze_cell_choices
        result = freeze_cell_choices(cfg)
        return result, [root(cfg) / "hyperparameters.json", root(cfg) / "selections.json"]
    if operation == "endpoint_chunk":
        from .evaluation import endpoint_cache_path, evaluate_endpoint_chunk
        result = evaluate_endpoint_chunk(cfg, task["chunk"], stop)
        outputs = [endpoint_cache_path(cfg, int(i["seed"]), int(i["epoch"])) for i in task["chunk"]["items"]]
        return result, outputs
    if operation == "endpoints_complete":
        marker = root(cfg) / "endpoint_metrics" / "complete.json"
        write_json(marker, {"status": "complete", "test_access": "after frozen selections"})
        return {"endpoints": len(cfg["stages"]) * len(sum(cfg["seed_pairs"], []))}, [marker]
    if operation == "calibration_evaluation":
        from .evaluation import evaluate_full_chunk
        from .protocol import pair_dir
        result = evaluate_full_chunk(cfg, task["chunk"], stop)
        outputs = [
            pair_dir(cfg, int(i["replicate"]), int(i["left_epoch"]), int(i["right_epoch"])) / "full_profiles.json"
            for i in task["chunk"]["items"]
        ]
        return result, outputs
    if operation == "report":
        from .final_reporting import report
        result = report(cfg)
        return result, [root(cfg) / "report" / "summary.json", root(cfg) / "report" / "barriers.csv"]
    raise ValueError(operation)


def configure_torch(cfg):
    torch.set_num_threads(int(cfg.get("cpu_threads", 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_task(cfg, task, tasks):
    validate_config(cfg)
    configure_torch(cfg)
    status_dir = root(cfg) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{task['id']}.json"
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(cfg, task):
            print(f"Already complete: {task['id']}")
            return
        by_id = {value["id"]: value for value in tasks}
        for dependency in task["dependencies"]:
            if not completed(cfg, by_id[dependency]):
                raise RuntimeError(f"Unfinished prerequisite: {dependency}")
        stop, started = StopFlag(), time.monotonic()
        previous = json.loads(path.read_text()) if path.exists() else {}
        attempts = previous.get("attempts", [])
        record = dict(status="running", task=task, protocol_hash=protocol_hash(cfg), attempts=attempts)
        write_json(path, record)
        gpu = task["operation"] not in CPU_OPERATIONS
        try:
            if gpu:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable.")
                torch.cuda.reset_peak_memory_stats()
            result, outputs = dispatch(cfg, task, stop)
            stop.check()
            if any(not Path(value).exists() for value in outputs):
                raise RuntimeError("Operation finished without all declared outputs.")
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
            attempts.append(
                dict(
                    seconds=time.monotonic() - started,
                    status=record["status"],
                    code_hash=code_hash(),
                    process_peak_bytes=own,
                    worker_peak_bytes=children,
                    host_peak_bytes_upper_bound=own + children,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu and torch.cuda.is_available() else 0,
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu and torch.cuda.is_available() else 0,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                )
            )
            write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("operation")
    parser.add_argument("selector")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    cfg, tasks = manifest["config"], manifest["tasks"]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        identifier = f"{args.operation}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    else:
        identifier = args.selector
    task = next(value for value in tasks if value["id"] == identifier)
    run_task(cfg, task, tasks)


if __name__ == "__main__":
    main()
