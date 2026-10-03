"""Submit the final Fashion-MNIST WM/Sinkhorn comparison to Slurm."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.fashion_mnist.final_alignment import final_alignment_root
from mode_connectivity.fashion_mnist.protocol import (
    checkpoint,
    root,
    verify_protocol,
    write_json,
)
from mode_connectivity.fashion_mnist.runner import validate_config
from omegaconf import OmegaConf

from .final_alignment_run import validate_final_alignment_config


def submit(command, dry_run):
    print(shlex.join(command), flush=True)
    if dry_run:
        submit.counter += 1
        return str(910000 + submit.counter)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


submit.counter = 0


def _verify_final_endpoints(cfg):
    verify_protocol(cfg)
    epoch = int(cfg["epochs"])
    missing = [
        checkpoint(cfg, seed, epoch)
        for seed in sum(cfg["seed_pairs"], [])
        if not checkpoint(cfg, seed, epoch).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run the Fashion-MNIST endpoint jobs first. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--partition", default=os.environ.get("MC_PARTITION", "general")
    )
    parser.add_argument("--qos", default=os.environ.get("MC_QOS", "short"))
    parser.add_argument("--gres", default=os.environ.get("MC_GRES", "gpu:a40:1"))
    parser.add_argument("--account", default=os.environ.get("MC_ACCOUNT"))
    parser.add_argument("--concurrency", type=int, default=3)
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
    validate_final_alignment_config(cfg)
    cfg["output_root"] = str(root(cfg))
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    project = Path(__file__).resolve().parents[2]
    destination = final_alignment_root(cfg)
    manifest = destination / "submission_config.json"
    if not args.dry_run:
        _verify_final_endpoints(cfg)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "logs").mkdir(exist_ok=True)
        write_json(manifest, {"config": cfg})

    common = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=16GB",
        "--signal=USR1@120",
    ]
    if args.account:
        common.append(f"--account={args.account}")
    gpu = project / "ops/slurm/fashion_mnist/run_final_alignment_gpu.sh"
    cpu = project / "ops/slurm/fashion_mnist/run_final_alignment_cpu.sh"
    array = f"0-{len(cfg['seed_pairs']) - 1}%{args.concurrency}"

    base = submit(
        common
        + [
            "--time=03:50:00",
            "--job-name=fmnist_final_sinkhorn",
            f"--output={destination}/logs/%x_%A_%a.out",
            f"--error={destination}/logs/%x_%A_%a.err",
            f"--array={array}",
            f"--gres={args.gres}",
            str(gpu),
            str(manifest),
            "base",
        ],
        args.dry_run,
    )
    scale = submit(
        common
        + [
            "--time=03:50:00",
            "--job-name=fmnist_final_scale",
            f"--output={destination}/logs/%x_%A_%a.out",
            f"--error={destination}/logs/%x_%A_%a.err",
            f"--array={array}",
            f"--dependency=afterok:{base}",
            f"--gres={args.gres}",
            str(gpu),
            str(manifest),
            "scale_evaluate",
        ],
        args.dry_run,
    )
    report = submit(
        common
        + [
            "--time=00:30:00",
            "--job-name=fmnist_final_report",
            f"--output={destination}/logs/%x_%j.out",
            f"--error={destination}/logs/%x_%j.err",
            f"--dependency=afterok:{scale}",
            str(cpu),
            str(manifest),
        ],
        args.dry_run,
    )
    if not args.dry_run:
        write_json(
            destination / "submission_jobs.json",
            dict(base=base, scale_evaluate=scale, report=report),
        )


if __name__ == "__main__":
    main()
