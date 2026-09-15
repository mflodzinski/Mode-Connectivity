"""Stable task DAG and resumable Slurm array submission. No submission on import."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
from collections import defaultdict

from .protocol import root, protocol_hash, verify_protocol, write_json

TIMES = {
    "prepare": "00:20:00",
    "smoke": "00:20:00",
    "train": "03:00:00",
    "onset": "00:30:00",
    "onset_report": "00:20:00",
    "base": "03:00:00",
    "scale": "01:00:00",
    "continue": "01:00:00",
    "evaluate": "00:30:00",
    "controls": "01:00:00",
    "audit": "01:00:00",
    "gate": "00:20:00",
    "report": "00:20:00",
}
CPU_ONLY = {"prepare", "gate", "report", "onset_report"}


def build_dag(cfg):
    tasks = []

    def add(id, operation, deps=(), **kw):
        task = dict(
            id=id, operation=operation, dependencies=list(dict.fromkeys(deps)), **kw
        )
        tasks.append(task)
        return id

    add("prepare", "prepare")
    add("smoke", "smoke", ["prepare"])
    for seed in sum(cfg["seed_pairs"], []):
        add(f"train_{seed}", "train", ["smoke"], seed=seed)
    onset_gate_dependencies = []
    onset_tasks = []
    if cfg.get("onset_enabled", False):
        final_onset = max(cfg["onset_epochs"])
        for rep, seeds in enumerate(cfg["seed_pairs"]):
            training_dependencies = [f"train_{seed}" for seed in seeds]
            final_id = add(
                f"onset_{rep}_{final_onset}",
                "onset",
                training_dependencies,
                replicate=rep,
                epoch=final_onset,
                index=final_onset,
            )
            onset_tasks.append(final_id)
            if rep == 0 and final_onset in cfg["onset_pilot_epochs"]:
                onset_gate_dependencies.append(final_id)
            for epoch in cfg["onset_epochs"]:
                if epoch == final_onset:
                    continue
                task_id = add(
                    f"onset_{rep}_{epoch}",
                    "onset",
                    training_dependencies + [final_id],
                    replicate=rep,
                    epoch=epoch,
                    index=epoch,
                )
                onset_tasks.append(task_id)
                if rep == 0 and epoch in cfg["onset_pilot_epochs"]:
                    onset_gate_dependencies.append(task_id)
    stages = cfg["stages"]
    pairs = [(a, b) for a in stages for b in stages]
    pilot_indices = [
        pairs.index(p)
        for p in [
            (stages[0], stages[0]),
            (stages[0], stages[-1]),
            (stages[-1], stages[-1]),
        ]
    ]

    def alignment(rep, indices, pilot=False):
        for operation in ["base", "scale", "continue"]:
            for i in indices:
                ea, eb = pairs[i]
                if operation == "base":
                    deps = [f"train_{s}" for s in cfg["seed_pairs"][rep]]
                    if not pilot:
                        deps.append("gate")
                else:
                    deps = [f"base_{rep}_{i}"]
                add(
                    f"{operation}_{rep}_{i}",
                    operation,
                    deps,
                    replicate=rep,
                    ea=ea,
                    eb=eb,
                    index=i,
                )

    alignment(0, pilot_indices, pilot=True)
    add("audit", "audit", [f"scale_0_{i}" for i in pilot_indices])
    gate_deps = (
        ["audit"]
        + [f"continue_0_{i}" for i in pilot_indices]
        + onset_gate_dependencies
    )
    add("gate", "gate", gate_deps)
    for rep in range(len(cfg["seed_pairs"])):
        indices = [i for i in range(len(pairs)) if rep != 0 or i not in pilot_indices]
        alignment(rep, indices)
    for rep in range(len(cfg["seed_pairs"])):
        for i, (ea, eb) in enumerate(pairs):
            add(
                f"evaluate_{rep}_{i}",
                "evaluate",
                [
                    f"scale_{rep}_{i}",
                    f"continue_{rep}_{i}",
                    f"base_{rep}_{len(pairs)-1}",
                    "gate",
                ],
                replicate=rep,
                ea=ea,
                eb=eb,
                index=i,
            )
        add(
            f"controls_{rep}",
            "controls",
            [f"evaluate_{rep}_{i}" for i in range(len(pairs))],
            replicate=rep,
        )
    report_dependencies = [
        f"controls_{r}" for r in range(len(cfg["seed_pairs"]))
    ] + ["gate"]
    if cfg.get("onset_enabled", False):
        add("onset_report", "onset_report", onset_tasks)
        report_dependencies.append("onset_report")
    add(
        "report",
        "report",
        report_dependencies,
    )
    return tasks


def prerequisite_tasks(tasks, target):
    by_id = {t["id"]: t for t in tasks}
    wanted = set()

    def visit(id):
        wanted.add(id)
        for dep in by_id[id]["dependencies"]:
            if dep not in wanted:
                visit(dep)

    visit(target)
    return [t for t in tasks if t["id"] in wanted]


def select_tasks(tasks, mode):
    if mode in ("all", "main"):
        return tasks  # main also schedules missing prerequisites; completed pilot is reused.
    gate_ids = {t["id"] for t in prerequisite_tasks(tasks, "gate")}
    # Start every independent endpoint during the pilot submission. Only seed pair
    # zero participates in the gate, so slow endpoints never block pilot review.
    wanted = gate_ids | {t["id"] for t in tasks if t["operation"] == "train"}
    return [t for t in tasks if t["id"] in wanted]


def completed(cfg, task):
    path = root(cfg) / "status" / f"{task['id']}.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return (
        record.get("status") == "complete"
        and record.get("protocol_hash") == protocol_hash(cfg)
        and all(Path(p).exists() for p in record.get("outputs", []))
    )


def dependency_expression(task, refs):
    corr, after = set(), set()
    for dep in task["dependencies"]:
        ref = refs.get(dep)
        if not ref:
            continue
        job, index = ref
        if index is not None and index == task.get("index"):
            corr.add(job)
        else:
            # A scalar consumer can depend on completion of the whole parent
            # array. This is shorter and correctly covers every submitted index.
            after.add(job if index is None or "index" not in task else f"{job}_{index}")
    parts = []
    if corr:
        parts.append("aftercorr:" + ":".join(sorted(corr)))
    if after:
        parts.append("afterok:" + ":".join(sorted(after)))
    return ",".join(parts)


def preflight(partition, qos, gres):
    for command in [
        ["sinfo", "-h", "-p", partition, "-o", "%P %G"],
        ["sacctmgr", "-n", "-P", "show", "qos", qos, "format=Name,MaxWall"],
    ]:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if not result.stdout.strip():
            raise RuntimeError(f"Scheduler setting unavailable: {shlex.join(command)}")
        print(result.stdout.strip())
        if command[0] == "sinfo" and ":".join(gres.split(":")[:2]) not in result.stdout:
            raise RuntimeError(
                f"Requested GRES {gres} not advertised by partition {partition}."
            )


def submit(
    cfg,
    tasks,
    dry_run=False,
    concurrency=4,
    partition="general",
    qos="short",
    gres="gpu:a40:1",
    account=None,
):
    if concurrency < 1:
        raise ValueError("Concurrency must be positive.")
    project = Path(__file__).resolve().parents[3]
    directory = root(cfg)
    manifest = directory / "tasks.json"
    all_tasks = build_dag(cfg)
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        preflight(partition, qos, gres)
        if (directory / "protocol.json").exists():
            verify_protocol(cfg)
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if protocol_hash(previous["config"]) != protocol_hash(cfg):
                raise ValueError(
                    "Submission protocol changed: use another output_root."
                )
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
            job_ref = old["job"] + (
                f"_{old['index']}" if old["index"] is not None else ""
            )
            state = subprocess.run(
                ["squeue", "-h", "-j", job_ref, "-o", "%T|%r"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if (
                state
                and "DependencyNeverSatisfied" not in state
                and any(
                    state.startswith(s)
                    for s in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]
                )
            ):
                refs[task["id"]] = (old["job"], old["index"])
                continue
        remaining.append(task)
    counter = 0
    while remaining:
        ready = [t for t in remaining if all(d in refs for d in t["dependencies"])]
        if not ready:
            raise RuntimeError("DAG has unresolved dependencies.")
        groups = defaultdict(list)
        for task in ready:
            expression = dependency_expression(task, refs)
            key = (
                task["operation"],
                task.get("replicate"),
                expression,
                None if "index" in task else task["id"],
            )
            groups[key].append(task)
        for (operation, rep, expression, _), group in groups.items():
            is_array = "index" in group[0]
            command = [
                "sbatch",
                "--parsable",
                f"--partition={partition}",
                f"--qos={qos}",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=4GB",
                f"--time={TIMES[operation]}",
                "--signal=USR1@120",
                f"--job-name=stage_{operation}",
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
                indices = ",".join(str(t["index"]) for t in group)
                command.append(f"--array={indices}%{concurrency}")
            script = (
                project
                / "ops/slurm/training_stage"
                / ("run_cpu.sh" if operation in CPU_ONLY else "run_gpu.sh")
            )
            command += [
                str(script),
                str(manifest),
                operation,
                str(rep) if is_array else group[0]["id"],
            ]
            print(shlex.join(command), flush=True)
            counter += 1
            job = (
                str(900000 + counter)
                if dry_run
                else subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PROJECT_ROOT": str(project)},
                )
                .stdout.strip()
                .split(";")[0]
            )
            if not job.isdigit():
                raise RuntimeError(f"Unexpected sbatch output: {job!r}")
            for task in group:
                index = task.get("index")
                refs[task["id"]] = (job, index)
                registry[task["id"]] = dict(job=job, index=index)
                remaining.remove(task)
            submitted.append(
                dict(command=command, tasks=[t["id"] for t in group], job=job)
            )
            if not dry_run:
                write_json(registry_path, registry)
    return submitted


def main():
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "main", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--partition", default=os.environ.get("MC_PARTITION", "general")
    )
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="training_stage/default",
            caller_file=__file__,
            argv=overrides,
        ),
        resolve=True,
    )
    cfg["output_root"], cfg["data_root"] = (
        str(root(cfg)),
        str(Path(cfg["data_root"]).resolve()),
    )
    from .runner import validate_config

    validate_config(cfg)
    tasks = select_tasks(build_dag(cfg), args.mode)
    kwargs = vars(args).copy()
    kwargs.pop("mode")
    if args.dry_run:
        submit(cfg, tasks, **kwargs)
    else:
        root(cfg).mkdir(parents=True, exist_ok=True)
        with (root(cfg) / ".submission.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(cfg, tasks, **kwargs)


if __name__ == "__main__":
    main()
