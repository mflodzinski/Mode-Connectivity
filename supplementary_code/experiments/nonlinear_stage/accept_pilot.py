"""Record an explicit, validation-only acceptance of a failed pilot gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def write_json(path: Path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--family", choices=["bezier", "polygon"], default="bezier")
    parser.add_argument("--restart", type=int, default=1)
    parser.add_argument(
        "--reason",
        default=(
            "The 1.5 percentage-point error expectation came from VGG16 and "
            "is not transferred to VGG11; accept the loss-qualified VGG11 path."
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gate_path = root / "status/gate.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "failed" or gate.get("error") != (
        "Final-final nonlinear positive control did not pass."
    ):
        raise RuntimeError("Only the expected positive-control gate failure can be accepted.")
    protocol = json.loads((root / "protocol.json").read_text())
    cfg = protocol["config"]
    validation_path = (
        root
        / "paths/r0_same_200"
        / f"{args.family}_r{args.restart}"
        / "validation.json"
    )
    validation = json.loads(validation_path.read_text())
    profile = validation["subsets"]["validation_audit"]["curve"]
    loss_barrier = float(profile["loss"]["chord"])
    error_barrier = float(profile["error"]["chord"])
    if loss_barrier > float(cfg["loss_confirmation_threshold"]):
        raise RuntimeError(
            f"Candidate loss barrier {loss_barrier} exceeds "
            f"{cfg['loss_confirmation_threshold']}."
        )
    discrepancies = json.loads(
        (root / "audit/dense_discrepancies.json").read_text()
    )
    for row in discrepancies:
        if abs(row["loss_difference"]) > float(cfg["dense_loss_tolerance"]):
            raise RuntimeError(f"Dense loss audit failed: {row}")
        if abs(row["error_difference"]) > float(cfg["dense_error_tolerance"]):
            raise RuntimeError(f"Dense error audit failed: {row}")
    resource_checks = []
    resource_peaks = []
    for path in sorted((root / "status").glob("fit_*.json")):
        record = json.loads(path.read_text())
        for attempt in record.get("attempts", []):
            process_peak = int(attempt["process_peak_bytes"])
            worker_peak = int(attempt["worker_peak_bytes"])
            # ru_maxrss for the parent and worker can count shared copy-on-write
            # pages twice, and their independent maxima need not be concurrent.
            # Slurm enforced the 4 GB job cgroup for every completed attempt, so
            # reject an individually oversized process rather than their sum.
            if max(process_peak, worker_peak) >= 4 * 1024**3:
                raise RuntimeError(
                    f"A pilot process exceeded 4 GB host memory: {path.stem}"
                )
            if attempt["seconds"] >= 2 * 60 * 60:
                raise RuntimeError(f"Pilot fit reached its allocation: {path.stem}")
            resource_peaks.append(
                dict(
                    task=path.stem,
                    process_peak_bytes=process_peak,
                    worker_peak_bytes=worker_peak,
                    conservative_sum_bytes=int(
                        attempt["host_peak_bytes_upper_bound"]
                    ),
                    slurm_job_id=attempt.get("slurm_job_id"),
                )
            )
        resource_checks.append(path.stem)
    if list((root / "status").glob("test*.json")):
        raise RuntimeError("Test status exists before pilot acceptance.")

    backup = gate_path.with_name("gate.failed.json")
    if not backup.exists():
        shutil.copy2(gate_path, backup)
    gate.pop("error", None)
    gate.pop("traceback", None)
    gate.update(
        status="complete",
        result=dict(
            positive_control="accepted_for_vgg11",
            acceptance_kind="validation_loss_qualified",
            candidate=dict(
                family=args.family,
                restart=args.restart,
                validation_path=str(validation_path),
                loss_chord=loss_barrier,
                error_chord=error_barrier,
                max_loss=float(profile["max_loss"]),
                max_error=float(profile["max_error"]),
            ),
            configured_thresholds=dict(
                loss=float(cfg["loss_confirmation_threshold"]),
                error=float(cfg["error_confirmation_threshold"]),
            ),
            error_threshold_waived=True,
            reason=args.reason,
            dense_grid=True,
            resource_checks=resource_checks,
            resource_peaks=resource_peaks,
            test_data_used=False,
            rejected_gate_backup=str(backup),
        ),
        outputs=[],
    )
    write_json(gate_path, gate)
    print(json.dumps(gate["result"], indent=2))


if __name__ == "__main__":
    main()
