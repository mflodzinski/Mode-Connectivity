"""Submit Fashion-MNIST protocol preparation and endpoint training to Slurm."""

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


def _submit(command, dry_run, counter):
    print(shlex.join(command), flush=True)
    if dry_run:
        return str(920000 + counter)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


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
    manifest = destination / "submissions" / "endpoints_config.json"
    logs = destination / "logs" / "endpoints"
    if not args.dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"config": cfg})
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
    prepare = _submit(
        common
        + [
            "--time=00:30:00",
            "--job-name=fmnist_prepare",
            f"--output={logs}/%x_%j.out",
            f"--error={logs}/%x_%j.err",
            str(cpu),
            str(manifest),
            "prepare",
        ],
        args.dry_run,
        1,
    )
    seeds = sum(cfg["seed_pairs"], [])
    training = _submit(
        common
        + [
            "--time=03:50:00",
            "--job-name=fmnist_train",
            f"--output={logs}/%x_%A_%a.out",
            f"--error={logs}/%x_%A_%a.err",
            f"--array=0-{len(seeds) - 1}%{args.concurrency}",
            f"--dependency=afterok:{prepare}",
            f"--gres={args.gres}",
            str(gpu),
            str(manifest),
            "train",
        ],
        args.dry_run,
        2,
    )
    if not args.dry_run:
        write_json(
            destination / "submissions" / "endpoints_jobs.json",
            dict(prepare=prepare, training=training, seeds=seeds),
        )


if __name__ == "__main__":
    main()
