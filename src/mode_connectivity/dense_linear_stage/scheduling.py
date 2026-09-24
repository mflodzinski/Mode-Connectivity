"""Resume-safe Slurm submission with measured-resource presets and bundled arrays."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

from .protocol import protocol_hash, root, verify_protocol, write_json
from .runner import CPU_OPERATIONS, completed, validate_config
from .tasks import build_dag, select_tasks


def queued_state(job_ref):
    result = subprocess.run(
        ["squeue", "-h", "-j", job_ref, "-o", "%T|%r"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def preflight(partition, qos, gres):
    commands = [
        ["sinfo", "-h", "-p", partition, "-o", "%P %G"],
        ["sacctmgr", "-n", "-P", "show", "qos", qos, "format=Name,MaxWall"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            raise RuntimeError(f"Scheduler setting unavailable: {shlex.join(command)}")
        print(result.stdout.strip())
    advertised = subprocess.run(commands[0], capture_output=True, text=True, check=True).stdout
    if ":".join(gres.split(":")[:2]) not in advertised:
        raise RuntimeError(f"Requested GRES {gres} is not advertised on {partition}.")


def dependency_expression(task, refs):
    corresponding, whole, exact = set(), set(), set()
    for dependency in task["dependencies"]:
        reference = refs.get(dependency)
        if not reference:
            continue
        job, index = reference
        if index is not None and index == task.get("index"):
            corresponding.add(job)
        elif "index" not in task:
            whole.add(job)
        else:
            exact.add(job if index is None else f"{job}_{index}")
    pieces = []
    if corresponding:
        pieces.append("aftercorr:" + ":".join(sorted(corresponding)))
    if whole or exact:
        pieces.append("afterok:" + ":".join(sorted(whole | exact)))
    return ",".join(pieces)


def resources(cfg, operation):
    table = cfg["slurm_resources"]
    key = operation if operation in table else (
        "cpu" if operation in CPU_OPERATIONS else "gpu"
    )
    return table[key]


def submit(cfg, tasks, *, dry_run=False, concurrency=3, partition="general", qos="short", gres="gpu:a40:1", account=None):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    project = Path(__file__).resolve().parents[3]
    directory, all_tasks = root(cfg), build_dag(cfg)
    manifest = directory / "tasks.json"
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        preflight(partition, qos, gres)
        if (directory / "protocol.json").exists():
            verify_protocol(cfg)
        if manifest.exists():
            previous = json.loads(manifest.read_text())["config"]
            if protocol_hash(previous) != protocol_hash(cfg):
                raise ValueError("Submission config changed; use a new output_root.")
        write_json(manifest, dict(config=cfg, tasks=all_tasks))
        (directory / "logs").mkdir(exist_ok=True)
    registry_path = directory / "submissions.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    refs, remaining, submitted = {}, [], []
    for task in tasks:
        if completed(cfg, task):
            refs[task["id"]] = None
            continue
        old = registry.get(task["id"])
        if old and not dry_run:
            ref = old["job"] + (f"_{old['index']}" if old["index"] is not None else "")
            state = queued_state(ref)
            if state and "DependencyNeverSatisfied" not in state and any(
                state.startswith(value) for value in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")
            ):
                refs[task["id"]] = (old["job"], old["index"])
                continue
        remaining.append(task)
    fake = 910000
    while remaining:
        ready = [task for task in remaining if all(dep in refs for dep in task["dependencies"])]
        if not ready:
            raise RuntimeError("Task graph has unresolved dependencies.")
        groups = defaultdict(list)
        for task in ready:
            expression = dependency_expression(task, refs)
            groups[(task["operation"], expression, None if "index" in task else task["id"])].append(task)
        for (operation, expression, _), group in groups.items():
            is_array = "index" in group[0]
            request = resources(cfg, operation)
            command = [
                "sbatch", "--parsable", f"--partition={partition}", f"--qos={qos}",
                "--ntasks=1", f"--cpus-per-task={request['cpus']}", f"--mem={request['mem']}",
                f"--time={request['time']}", "--signal=USR1@60", "--mail-type=FAIL",
                f"--job-name=dense_{cfg['dataset']}_{operation}",
                f"--output={directory}/logs/%x_%A_%a.out",
                f"--error={directory}/logs/%x_%A_%a.err",
            ]
            if account:
                command.append(f"--account={account}")
            if operation not in CPU_OPERATIONS:
                command.append(f"--gres={gres}")
            if expression:
                command.append(f"--dependency={expression}")
            if is_array:
                indices = ",".join(str(task["index"]) for task in group)
                command.append(f"--array={indices}%{concurrency}")
            script = project / "ops/slurm/dense_linear_stage" / (
                "run_cpu.sh" if operation in CPU_OPERATIONS else "run_gpu.sh"
            )
            command += [str(script), str(manifest), operation, group[0]["id"]]
            print(shlex.join(command), flush=True)
            if dry_run:
                fake += 1
                job = str(fake)
            else:
                job = subprocess.run(
                    command, check=True, capture_output=True, text=True,
                    env={**os.environ, "PROJECT_ROOT": str(project)},
                ).stdout.strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected sbatch output: {job!r}")
            for task in group:
                index = task.get("index")
                refs[task["id"]] = (job, index)
                registry[task["id"]] = dict(job=job, index=index)
                remaining.remove(task)
            submitted.append(dict(job=job, operation=operation, tasks=[t["id"] for t in group], command=command))
            if not dry_run:
                write_json(registry_path, registry)
    return submitted


def main():
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "pilot", "calibration", "main", "evaluate", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--partition", default=os.environ.get("MC_PARTITION", "general"))
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    parser.add_argument("--config-name", default="dense_linear_stage/vgg11")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.to_container(
        compose_experiment_config(
            default_config_name=args.config_name,
            caller_file=__file__, argv=overrides,
        ), resolve=True,
    )
    cfg["output_root"] = str(root(cfg))
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    cfg["source_roots"] = {str(k): str(Path(v).resolve()) for k, v in cfg["source_roots"].items()}
    reuse = cfg.get("reuse", {})
    for source in reuse.get("wm_sources", []):
        source["root"] = str(Path(source["root"]).resolve())
    if reuse.get("vgg_alignment_grid"):
        reuse["vgg_alignment_grid"] = str(Path(reuse["vgg_alignment_grid"]).resolve())
    validate_config(cfg)
    tasks = select_tasks(build_dag(cfg), args.mode)
    kwargs = dict(
        dry_run=args.dry_run, concurrency=args.concurrency, partition=args.partition,
        qos=args.qos, gres=args.gres, account=args.account,
    )
    if args.dry_run:
        submit(cfg, tasks, **kwargs)
    else:
        root(cfg).mkdir(parents=True, exist_ok=True)
        with (root(cfg) / ".submission.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(cfg, tasks, **kwargs)


if __name__ == "__main__":
    main()
