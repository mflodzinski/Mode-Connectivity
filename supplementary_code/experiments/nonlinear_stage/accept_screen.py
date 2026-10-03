"""Record a validation-only waiver of the VGG16 error confirmation cutoff."""

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
    parser.add_argument(
        "--reason",
        default=(
            "The 1.5 percentage-point error cutoff came from VGG16 and is not "
            "transferred to VGG11; confirmation is triggered by validation loss."
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    screen_path = root / "status/screen.json"
    screen = json.loads(screen_path.read_text())
    if screen.get("status") != "complete":
        raise RuntimeError("Screening must be complete before its policy is accepted.")

    # This decision must precede confirmation fitting, selection, and test access.
    forbidden = [
        *sorted((root / "status").glob("fit_confirm_*.json")),
        *sorted((root / "status").glob("validate_confirm_*.json")),
        *sorted((root / "status").glob("freeze_*.json")),
        *sorted((root / "status").glob("test_*.json")),
    ]
    forbidden += [
        root / "status/selection_gate.json",
        root / "status/report.json",
    ]
    forbidden = [path for path in forbidden if path.exists()]
    if forbidden:
        raise RuntimeError(
            "Cannot change screening after confirmation/selection/test work exists: "
            + ", ".join(path.name for path in forbidden[:10])
        )

    target_path = root / "confirmation_targets.json"
    backup_path = root / "confirmation_targets.before_error_waiver.json"
    if backup_path.exists():
        raise RuntimeError(f"Screening waiver was already recorded at {backup_path}.")
    record = json.loads(target_path.read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    cfg = protocol["config"]
    loss_limit = float(cfg["loss_confirmation_threshold"])
    error_limit = float(cfg["error_confirmation_threshold"])

    original_targets = list(record["targets"])
    expected_original = [
        row["pair"]
        for row in record["rows"]
        if float(row["loss_chord"]) > loss_limit
        or float(row["error_chord"]) > error_limit
    ]
    if original_targets != expected_original:
        raise RuntimeError(
            "confirmation_targets.json does not match the configured screening rule."
        )

    retained = [
        row["pair"]
        for row in record["rows"]
        if float(row["loss_chord"]) > loss_limit
    ]
    waived = [pair for pair in original_targets if pair not in set(retained)]
    shutil.copy2(target_path, backup_path)
    record.update(
        targets=retained,
        original_targets=original_targets,
        screening_policy=dict(
            kind="validation_loss_only",
            loss_threshold=loss_limit,
            configured_error_threshold=error_limit,
            error_threshold_waived=True,
            reason=args.reason,
            original_target_count=len(original_targets),
            retained_target_count=len(retained),
            waived_target_count=len(waived),
            waived_targets=waived,
            test_data_used=False,
            original_file=str(backup_path),
        ),
    )
    write_json(target_path, record)
    print(json.dumps(record["screening_policy"], indent=2))


if __name__ == "__main__":
    main()
