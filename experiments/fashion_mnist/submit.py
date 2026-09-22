"""Submit the Fashion-MNIST endpoint and analysis arrays to Slurm."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.fashion_mnist.protocol import root, write_json
from mode_connectivity.fashion_mnist.runner import validate_config
from omegaconf import OmegaConf


def submit(command, dry_run):
    print(shlex.join(command), flush=True)
    if dry_run:
        return str(900000 + submit.counter)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


submit.counter = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    manifest = destination / "submission_config.json"
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"config": cfg})
        (destination / "logs").mkdir(exist_ok=True)
    manifest_arg = str(manifest)
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
    cpu = project / "ops/slurm/fashion_mnist/run_cpu.sh"
    gpu = project / "ops/slurm/fashion_mnist/run_gpu.sh"

    submit.counter += 1
    prepare = submit(
        common
        + [
            "--time=00:30:00",
            "--job-name=fmnist_prepare",
            f"--output={destination}/logs/%x_%j.out",
            f"--error={destination}/logs/%x_%j.err",
            str(cpu),
            manifest_arg,
            "prepare",
        ],
        args.dry_run,
    )
    seeds = sum(cfg["seed_pairs"], [])
    submit.counter += 1
    training = submit(
        common
        + [
            "--time=03:50:00",
            "--job-name=fmnist_train",
            f"--output={destination}/logs/%x_%A_%a.out",
            f"--error={destination}/logs/%x_%A_%a.err",
            f"--array=0-{len(seeds)-1}%{args.concurrency}",
            f"--dependency=afterok:{prepare}",
            f"--gres={args.gres}",
            str(gpu),
            manifest_arg,
            "train",
        ],
        args.dry_run,
    )
    analyses = [
        (replicate, epoch)
        for replicate in range(len(cfg["seed_pairs"]))
        for epoch in cfg["stages"]
    ]
    submit.counter += 1
    analysis = submit(
        common
        + [
            "--time=03:50:00",
            "--job-name=fmnist_analyze",
            f"--output={destination}/logs/%x_%A_%a.out",
            f"--error={destination}/logs/%x_%A_%a.err",
            f"--array=0-{len(analyses)-1}%{args.concurrency}",
            f"--dependency=afterok:{training}",
            f"--gres={args.gres}",
            str(gpu),
            manifest_arg,
            "analyze",
        ],
        args.dry_run,
    )
    submit.counter += 1
    submit(
        common
        + [
            "--time=00:30:00",
            "--job-name=fmnist_report",
            f"--output={destination}/logs/%x_%j.out",
            f"--error={destination}/logs/%x_%j.err",
            f"--dependency=afterok:{analysis}",
            str(cpu),
            manifest_arg,
            "report",
        ],
        args.dry_run,
    )
    if not args.dry_run:
        write_json(
            destination / "submission_plan.json",
            dict(seeds=seeds, analyses=analyses),
        )


if __name__ == "__main__":
    main()
