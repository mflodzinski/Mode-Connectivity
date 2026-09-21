"""Resumable Slurm DAG with validation-selected confirmation jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections import defaultdict
from pathlib import Path
import shlex
import subprocess

from .evaluation import EXTRA_SPECS, confirmation_targets
from .protocol import (
    pair_by_id,
    pilot_pair_ids,
    primary_pairs,
    protocol_hash,
    root,
    verify_protocol,
    write_json,
)
from .runner import completed, validate_config


TIMES = {
    "prepare": "00:20:00",
    "smoke": "00:20:00",
    "fit": "02:00:00",
    "validate": "00:30:00",
    "audit": "01:00:00",
    "gate": "00:20:00",
    "screen": "00:20:00",
    "freeze": "00:20:00",
    "selection_gate": "00:20:00",
    "test": "00:30:00",
    "endpoint_test": "00:30:00",
    "report": "00:20:00",
    "calibration_report": "00:20:00",
}
CPU_ONLY = {
    "prepare", "gate", "screen", "freeze", "selection_gate", "report",
    "calibration_report",
}


def build_dag(cfg, targets=None):
    tasks = []

    def add(task_id, operation, dependencies=(), **values):
        task = dict(
            id=task_id,
            operation=operation,
            dependencies=list(dict.fromkeys(dependencies)),
            **values,
        )
        tasks.append(task)
        return task_id

    def add_path(label, pair, family, restart, noise, dependencies):
        suffix = f"r{pair['replicate']}_{pair['index']}"
        fit_id = add(
            f"fit_{label}_{suffix}", "fit", dependencies,
            array_group=f"fit_{label}", replicate=pair["replicate"], index=pair["index"],
            pair_id=pair["id"], family=family, restart=restart, noise=noise,
        )
        validate_id = add(
            f"validate_{label}_{suffix}", "validate", [fit_id],
            array_group=f"validate_{label}", replicate=pair["replicate"], index=pair["index"],
            pair_id=pair["id"], family=family, restart=restart,
        )
        return fit_id, validate_id

    add("prepare", "prepare")
    add("smoke", "smoke", ["prepare"])
    pilot_ids = set(pilot_pair_ids(cfg))
    primary_validation = {}
    pilot_validation = []
    pairs = primary_pairs(cfg)
    for pair in pairs:
        if pair["id"] in pilot_ids:
            _, validation = add_path("primary", pair, "bezier", 0, 0.0, ["smoke"])
            primary_validation[pair["id"]] = validation
            pilot_validation.append(validation)
    final_pair = next(
        pair for pair in pairs
        if pair["replicate"] == 0 and pair["kind"] == "same"
        and pair["n"] == cfg["final_epoch"]
    )
    positive_validation = []
    for family, restart, noise in EXTRA_SPECS:
        label = f"positive_{family}_r{restart}"
        _, validation = add_path(label, final_pair, family, restart, noise, ["smoke"])
        positive_validation.append(validation)
    add("audit", "audit", pilot_validation)
    add("gate", "gate", ["audit"] + pilot_validation + positive_validation)
    for pair in pairs:
        if pair["id"] in pilot_ids:
            continue
        _, validation = add_path("primary", pair, "bezier", 0, 0.0, ["gate"])
        primary_validation[pair["id"]] = validation
    add("screen", "screen", list(primary_validation.values()) + ["gate"])

    if targets is None:
        return tasks
    targets = set(targets)
    confirmation_validation = defaultdict(list)
    for pair_id in sorted(targets):
        pair = pair_by_id(cfg, pair_id)
        if pair_id == final_pair["id"]:
            confirmation_validation[pair_id].extend(positive_validation)
            continue
        for family, restart, noise in EXTRA_SPECS:
            label = f"confirm_{family}_r{restart}"
            _, validation = add_path(label, pair, family, restart, noise, ["screen"])
            confirmation_validation[pair_id].append(validation)
    freeze_ids = []
    for pair in pairs:
        dependencies = ["screen", primary_validation[pair["id"]]]
        if pair["id"] == final_pair["id"]:
            dependencies += positive_validation
        dependencies += confirmation_validation[pair["id"]]
        freeze_ids.append(
            add(
                f"freeze_r{pair['replicate']}_{pair['index']}", "freeze", dependencies,
                array_group="freeze", replicate=pair["replicate"], index=pair["index"],
                pair_id=pair["id"],
            )
        )
    add("selection_gate", "selection_gate", freeze_ids)
    test_ids = []
    for pair in pairs:
        test_ids.append(
            add(
                f"test_r{pair['replicate']}_{pair['index']}", "test", ["selection_gate"],
                array_group="test", replicate=pair["replicate"], index=pair["index"],
                pair_id=pair["id"],
            )
        )
    endpoint_ids = [
        add(
            f"endpoint_test_{rep}", "endpoint_test", ["selection_gate"],
            array_group="endpoint_test", replicate=rep, index=rep,
        )
        for rep in range(len(cfg["source_pairs"]))
    ]
    add("report", "report", test_ids + endpoint_ids + ["audit"])
    return tasks


def prerequisite_tasks(tasks, target):
    by_id, wanted = {task["id"]: task for task in tasks}, set()

    def visit(task_id):
        if task_id in wanted:
            return
        wanted.add(task_id)
        for dependency in by_id[task_id]["dependencies"]:
            visit(dependency)

    visit(target)
    return [task for task in tasks if task["id"] in wanted]


def build_calibration_dag(cfg):
    """Focused equal-budget hyperparameter check for the final-final control."""
    tasks = []

    def add(task_id, operation, dependencies=(), **values):
        tasks.append(
            dict(
                id=task_id,
                operation=operation,
                dependencies=list(dict.fromkeys(dependencies)),
                **values,
            )
        )
        return task_id

    add("prepare", "prepare")
    add("smoke", "smoke", ["prepare"])
    pair = next(
        item
        for item in primary_pairs(cfg)
        if item["replicate"] == 0
        and item["kind"] == "same"
        and item["n"] == cfg["final_epoch"]
    )
    validations = []
    for index, spec in enumerate(cfg["calibration_specs"]):
        restart = 100 + index
        fit_id = add(
            f"fit_calibration_{index}",
            "fit",
            ["smoke"],
            array_group="fit_calibration",
            replicate=0,
            index=index,
            pair_id=pair["id"],
            family="bezier",
            restart=restart,
            noise=0.0,
            fit_overrides=dict(
                fit_subset="curve_fit",
                fit_passes=int(spec["passes"]),
                fit_lr=float(spec["lr"]),
                path_weight_decay=float(spec["weight_decay"]),
                validation_interval=5,
                seed=int(cfg["path_seed"]),
            ),
        )
        validations.append(
            add(
                f"validate_calibration_{index}",
                "validate",
                [fit_id],
                array_group="validate_calibration",
                replicate=0,
                index=index,
                pair_id=pair["id"],
                family="bezier",
                restart=restart,
            )
        )
    add("calibration_report", "calibration_report", validations)
    return tasks


def select_tasks(cfg, mode):
    if mode == "calibration":
        return build_calibration_dag(cfg)
    base = build_dag(cfg)
    if mode == "pilot":
        return prerequisite_tasks(base, "gate")
    if mode == "main":
        return prerequisite_tasks(base, "screen")
    target_file = root(cfg) / "confirmation_targets.json"
    if not target_file.exists():
        if mode == "confirm":
            raise RuntimeError("Run the main phase through screen before confirmation.")
        return prerequisite_tasks(base, "screen")
    extended = build_dag(cfg, confirmation_targets(cfg))
    return extended


def queued_state(job_ref):
    query = subprocess.run(
        ["squeue", "-h", "-j", job_ref, "-o", "%T|%r"],
        capture_output=True, text=True, check=False,
    )
    return query.stdout.strip() if query.returncode == 0 else ""


def dependency_expression(task, refs):
    corresponding, after = set(), set()
    for dependency in task["dependencies"]:
        reference = refs.get(dependency)
        if not reference:
            continue
        job, index = reference
        if "index" in task and index is not None and index == task["index"]:
            corresponding.add(job)
        else:
            after.add(job if "index" not in task else (job if index is None else f"{job}_{index}"))
    parts = []
    if corresponding:
        parts.append("aftercorr:" + ":".join(sorted(corresponding)))
    if after:
        parts.append("afterok:" + ":".join(sorted(after)))
    return ",".join(parts)


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
        if command[0] == "sinfo" and ":".join(gres.split(":")[:2]) not in result.stdout:
            raise RuntimeError(f"Requested GRES {gres} is unavailable on {partition}.")


def submit(
    cfg, tasks, *, dry_run=False, concurrency=4, partition="general", qos="short",
    gres="gpu:a40:1", account=None,
):
    project = Path(__file__).resolve().parents[3]
    directory, manifest = root(cfg), root(cfg) / "tasks.json"
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        preflight(partition, qos, gres)
        if (directory / "protocol.json").exists():
            verify_protocol(cfg)
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if protocol_hash(old["config"]) != protocol_hash(cfg):
                raise ValueError("Submission protocol changed; use another output_root.")
        write_json(manifest, dict(config=cfg, tasks=tasks))
        (directory / "logs").mkdir(exist_ok=True)
    registry_path = directory / "submissions.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    refs, remaining, submissions = {}, [], []
    for task in tasks:
        if completed(cfg, task):
            refs[task["id"]] = None
            continue
        old = registry.get(task["id"])
        if old and not dry_run:
            ref = old["job"] + (f"_{old['index']}" if old["index"] is not None else "")
            state = queued_state(ref)
            if state and "DependencyNeverSatisfied" not in state and any(
                state.startswith(value)
                for value in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]
            ):
                refs[task["id"]] = (old["job"], old["index"])
                continue
        remaining.append(task)
    fake_job = 900000
    while remaining:
        ready = [t for t in remaining if all(dep in refs for dep in t["dependencies"])]
        if not ready:
            raise RuntimeError("DAG has unresolved dependencies.")
        groups = defaultdict(list)
        for task in ready:
            expression = dependency_expression(task, refs)
            key = (
                task["operation"], task.get("array_group"), task.get("replicate"),
                expression, None if "index" in task else task["id"],
            )
            groups[key].append(task)
        for (operation, group_name, replicate, expression, _), group in groups.items():
            is_array = "index" in group[0]
            command = [
                "sbatch", "--parsable", f"--partition={partition}", f"--qos={qos}",
                "--ntasks=1", "--cpus-per-task=2", "--mem=4GB",
                f"--time={TIMES[operation]}", "--signal=USR1@120",
                f"--job-name=nonlinear_{operation}",
                f"--output={directory}/logs/%x_%A_%a.out",
                f"--error={directory}/logs/%x_%A_%a.err",
            ]
            if account:
                command.append(f"--account={account}")
            if operation not in CPU_ONLY:
                command.append(f"--gres={gres}")
            if expression:
                command.append(f"--dependency={expression}")
            if is_array:
                indices = ",".join(str(task["index"]) for task in group)
                command.append(f"--array={indices}%{concurrency}")
            script = project / "ops/slurm/nonlinear_stage" / (
                "run_cpu.sh" if operation in CPU_ONLY else "run_gpu.sh"
            )
            selector = group_name if is_array else group[0]["id"]
            command += [str(script), str(manifest), selector]
            if is_array:
                command += [str(replicate)]
            print(shlex.join(command), flush=True)
            if dry_run:
                fake_job += 1
                job = str(fake_job)
            else:
                job = subprocess.run(
                    command, capture_output=True, text=True, check=True,
                    env={**os.environ, "PROJECT_ROOT": str(project)},
                ).stdout.strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected sbatch output: {job!r}")
            for task in group:
                index = task.get("index")
                refs[task["id"]] = (job, index)
                registry[task["id"]] = dict(job=job, index=index)
                remaining.remove(task)
            submissions.append(dict(job=job, tasks=[task["id"] for task in group], command=command))
            if not dry_run:
                write_json(registry_path, registry)
    return submissions


def main():
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["calibration", "pilot", "main", "confirm", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--partition", default=os.environ.get("MC_PARTITION", "general"))
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="nonlinear_stage/default", caller_file=__file__, argv=overrides
        ), resolve=True,
    )
    cfg["output_root"] = str(Path(cfg["output_root"]).resolve())
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    for source in cfg["source_pairs"]:
        source["root"] = str(Path(source["root"]).resolve())
    validate_config(cfg)
    tasks = select_tasks(cfg, args.mode)
    kwargs = vars(args).copy()
    kwargs.pop("mode")
    if args.dry_run:
        submit(cfg, tasks, **kwargs)
        return
    root(cfg).mkdir(parents=True, exist_ok=True)
    with (root(cfg) / ".submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(cfg, tasks, **kwargs)


if __name__ == "__main__":
    main()
