"""Evaluate the standard pytorch-vgg shared-split pair suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAIR_SPECS = [
    ("100/100", 100.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_100split"),
    ("80/120", 80.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_80split"),
    ("30/170", 30.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_30split"),
    ("8/192", 8.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_8split"),
    ("6/194", 6.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_6split"),
    ("5/195", 5.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_5split"),
    ("4/196", 4.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_4split"),
    ("3/197", 3.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_3split"),
    ("2/198", 2.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_2split"),
    ("1/199", 1.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_1split"),
    ("0/200", 0.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_0split"),
    ("independent", None, "results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pytorch-vgg shared-split suite.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=61)
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of labels to evaluate, e.g. '100/100' '80/120' 'independent'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_filter = set(args.labels) if args.labels else None

    evaluator = PROJECT_ROOT / "scripts" / "analysis" / "evaluate_pytorch_vgg_pair.py"

    for label, _, root_rel in PAIR_SPECS:
        if labels_filter and label not in labels_filter:
            continue

        root = PROJECT_ROOT / root_rel
        w0 = root / "seed0" / "checkpoint-200.pt"
        w1 = root / "seed1" / "checkpoint-200.pt"
        out_dir = root / "evaluation"

        if not w0.exists() or not w1.exists():
            print(f"Skipping {label}: missing checkpoints under {root}")
            continue

        print("=" * 72)
        print(f"Evaluating {label}")
        print(f"  w0: {w0}")
        print(f"  w1: {w1}")
        print(f"  out: {out_dir}")
        print("=" * 72)

        cmd = [
            sys.executable,
            str(evaluator),
            "--w0",
            str(w0.relative_to(PROJECT_ROOT)),
            "--w1",
            str(w1.relative_to(PROJECT_ROOT)),
            "--data-root",
            args.data_root,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--num-points",
            str(args.num_points),
            "--output-dir",
            str(out_dir.relative_to(PROJECT_ROOT)),
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
