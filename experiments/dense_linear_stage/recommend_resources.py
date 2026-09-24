"""Summarize pilot telemetry and recommend DAIC requests with 25% headroom."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def walltime(seconds):
    seconds = max(300, int(math.ceil(seconds / 60.0) * 60))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in sorted((args.root / "status").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete" or not record.get("attempts"):
            continue
        operation = record["task"]["operation"]
        attempt = record["attempts"][-1]
        grouped[operation].append(attempt)
    if not grouped:
        raise RuntimeError("No completed status telemetry was found.")
    print("# Paste reviewed values into slurm_resources. MaxRSS from sacct remains authoritative.")
    for operation, attempts in sorted(grouped.items()):
        elapsed = np.asarray([float(row["seconds"]) for row in attempts])
        memory = np.asarray([float(row["host_peak_bytes_upper_bound"]) for row in attempts])
        seconds = 1.25 * float(np.percentile(elapsed, 95))
        mebibytes = max(1024, int(math.ceil((1.25 * float(np.percentile(memory, 95)) / 2**20) / 256) * 256))
        print(
            f"{operation}: {{time: \"{walltime(seconds)}\", mem: {mebibytes}M}}"
            f"  # n={len(attempts)}, p95 elapsed={np.percentile(elapsed,95):.1f}s,"
            f" p95 process+children={np.percentile(memory,95)/2**20:.0f}MiB"
        )


if __name__ == "__main__":
    main()

