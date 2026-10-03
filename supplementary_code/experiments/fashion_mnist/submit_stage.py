"""Submit linear or nonlinear Fashion-MNIST stage-connectivity jobs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.fashion_mnist.protocol import (
    checkpoint,
    root,
    verify_protocol,
    write_json,
)
from mode_connectivity.fashion_mnist.runner import validate_config
from omegaconf import OmegaConf


def _submit(command, dry_run, counter):
    print(shlex.join(command), flush=True)
    if dry_run:
        return str(930000 + counter)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


def _verify_checkpoints(cfg):
    verify_protocol(cfg)
    missing = [
        checkpoint(cfg, seed, epoch)
        for seed in sum(cfg["seed_pairs"], [])
        for epoch in cfg["stages"]
        if not checkpoint(cfg, seed, epoch).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run the endpoint submission first. Missing checkpoints: "
            + ", ".join(str(path) for path in missing[:8])
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["linear", "nonlinear"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--partition", default=os.environ.get("MC_PARTITION", "general")
    )
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    parser.add_argument("--concurrency", type=int, default=4)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="fashion_mnist/default",
            caller_file=__file__,
            argv=overrides,
        ),
        resolve=True,
    )
    validate_config(cfg)
    project = Path(__file__).resolve().parents[2]
    cfg["output_root"] = str(root(cfg))
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    destination = root(cfg)
    manifest = destination / "submissions" / f"{args.family}_stage_config.json"
    logs = destination / "logs" / f"{args.family}_stage"
    if not args.dry_run:
        _verify_checkpoints(cfg)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"config": cfg})
    tasks = [
        (replicate, epoch)
        for replicate in range(len(cfg["seed_pairs"]))
        for epoch in cfg["stages"]
    ]
    common = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=8GB",
        "--signal=USR1@120",
    ]
    if args.account:
        common.append(f"--account={args.account}")
    gpu = project / "ops/slurm/fashion_mnist/run_gpu.sh"
    cpu = project / "ops/slurm/fashion_mnist/run_cpu.sh"
    analysis = _submit(
        common
        + [
            "--time=03:50:00",
            f"--job-name=fmnist_{args.family}_stage",
            f"--output={logs}/%x_%A_%a.out",
            f"--error={logs}/%x_%A_%a.err",
            f"--array=0-{len(tasks) - 1}%{args.concurrency}",
            f"--gres={args.gres}",
            str(gpu),
            str(manifest),
            args.family,
        ],
        args.dry_run,
        1,
    )
    report = _submit(
        common
        + [
            "--time=00:30:00",
            f"--job-name=fmnist_{args.family}_report",
            f"--output={logs}/%x_%j.out",
            f"--error={logs}/%x_%j.err",
            f"--dependency=afterok:{analysis}",
            str(cpu),
            str(manifest),
            f"report_{args.family}",
        ],
        args.dry_run,
        2,
    )
    if not args.dry_run:
        write_json(
            destination / "submissions" / f"{args.family}_stage_jobs.json",
            dict(analysis=analysis, report=report, tasks=tasks),
        )


if __name__ == "__main__":
    main()
