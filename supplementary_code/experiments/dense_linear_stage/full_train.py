"""Evaluate frozen final alignments on the complete endpoint-training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mode_connectivity.dense_linear_stage.full_train import (
    evaluate_task,
    report,
    status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("evaluate", "report", "status"))
    parser.add_argument("root", type=Path)
    parser.add_argument("selector", nargs="?", type=int)
    args = parser.parse_args()
    if args.operation == "evaluate":
        if args.selector is None:
            parser.error("evaluate requires a task index")
        result = evaluate_task(args.root, args.selector)
    elif args.operation == "report":
        result = report(args.root)
    else:
        result = status(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
