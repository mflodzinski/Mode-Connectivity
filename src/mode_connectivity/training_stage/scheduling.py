"""Resume-safe Slurm submission for the paper's independent VGG endpoints."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path

from .protocol import protocol_hash, root, verify_protocol, write_json


def build_dag(cfg):
    tasks = [dict(id="prepare", operation="prepare", dependencies=[])]
    for seed in sum(cfg["seed_pairs"], []):
        tasks.append(dict(
            id=f"train_{seed}", operation="train", dependencies=["prepare"], seed=seed,
        ))
    return tasks


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


def queued_state(job_ref):
    result = subprocess.run(
        ["squeue", "-h", "-j", job_ref, "-o", "%T|%r"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def preflight(partition, qos, gres):
    for command in (
        ["sinfo", "-h", "-p", partition, "-o", "%P %G"],
        ["sacctmgr", "-n", "-P", "show", "qos", qos, "format=Name,MaxWall"],
    ):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if not result.stdout.strip():
            raise RuntimeError(f"Scheduler setting unavailable: {shlex.join(command)}")
        print(result.stdout.strip())
        if command[0] == "sinfo" and ":".join(gres.split(":")[:2]) not in result.stdout:
            raise RuntimeError(f"Requested GRES {gres} is not advertised on {partition}.")


def submit(cfg, tasks, *, dry_run=False, partition="general", qos="short", gres="gpu:a40:1", account=None):
    project = Path(__file__).resolve().parents[3]
    destination = root(cfg)
    manifest = destination / "tasks.json"
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        preflight(partition, qos, gres)
        if (destination / "protocol.json").exists():
            verify_protocol(cfg)
        if manifest.exists():
            previous = json.loads(manifest.read_text())["config"]
            if protocol_hash(previous) != protocol_hash(cfg):
                raise ValueError("Submission protocol changed; use another output_root.")
        write_json(manifest, dict(config=cfg, tasks=build_dag(cfg)))
        (destination / "logs").mkdir(exist_ok=True)
    registry_path = destination / "submissions.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    refs, submitted = {}, []
    fake = 900000
    for task in tasks:
        if completed(cfg, task):
            refs[task["id"]] = None
            continue
        previous = registry.get(task["id"])
        if previous and not dry_run:
            state = queued_state(previous["job"])
            if state and "DependencyNeverSatisfied" not in state and any(
                state.startswith(value)
                for value in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")
            ):
                refs[task["id"]] = previous["job"]
                continue
        dependencies = [refs[value] for value in task["dependencies"] if refs.get(value)]
        cpu = task["operation"] == "prepare"
        command = [
            "sbatch", "--parsable", f"--partition={partition}", f"--qos={qos}",
            "--ntasks=1", "--cpus-per-task=2", "--mem=4GB",
            f"--time={'00:20:00' if cpu else '03:00:00'}",
            "--signal=USR1@120", "--mail-type=FAIL",
            f"--job-name=vgg_{task['operation']}",
            f"--output={destination}/logs/%x_%j.out",
            f"--error={destination}/logs/%x_%j.err",
        ]
        if account:
            command.append(f"--account={account}")
        if not cpu:
            command.append(f"--gres={gres}")
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        script = project / "ops/slurm/training_stage" / ("run_cpu.sh" if cpu else "run_gpu.sh")
        command += [str(script), str(manifest), task["operation"], task["id"]]
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
        refs[task["id"]] = job
        registry[task["id"]] = dict(job=job)
        submitted.append(dict(job=job, task=task["id"], command=command))
        if not dry_run:
            write_json(registry_path, registry)
    return submitted


def main():
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf
    from .runner import validate_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--partition", default=os.environ.get("MC_PARTITION", "general"))
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.to_container(compose_experiment_config(
        default_config_name="training_stage/default", caller_file=__file__, argv=overrides,
    ), resolve=True)
    cfg["output_root"] = str(root(cfg))
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    validate_config(cfg)
    kwargs = vars(args)
    if args.dry_run:
        submit(cfg, build_dag(cfg), **kwargs)
    else:
        root(cfg).mkdir(parents=True, exist_ok=True)
        with (root(cfg) / ".submission.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(cfg, build_dag(cfg), **kwargs)


if __name__ == "__main__":
    main()
