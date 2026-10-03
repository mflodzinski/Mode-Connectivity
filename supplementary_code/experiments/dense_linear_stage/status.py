"""Show completion and failure counts for a dense linear-stage root."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "tasks.json").read_text())
    totals, states = Counter(), defaultdict(Counter)
    problems = []
    for task in manifest["tasks"]:
        operation = task["operation"]
        totals[operation] += 1
        path = args.root / "status" / f"{task['id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            state = record.get("status", "unknown")
            if state not in ("complete", "running"):
                problems.append((task["id"], state, record.get("error", "")))
        else:
            state = "missing"
        states[operation][state] += 1
    print(f"root: {args.root}")
    for operation in totals:
        summary = " ".join(f"{key}={value}" for key, value in sorted(states[operation].items()))
        print(f"{operation:25} total={totals[operation]:4d}  {summary}")
    print(f"\nproblems: {len(problems)}")
    for identifier, state, error in problems[:30]:
        print(f"{identifier:45} {state:12} {error}")


if __name__ == "__main__":
    main()

