"""Config-backed runner for the XOR trained-endpoints linear-barrier study.

The wrapper keeps the repo-level experiment surface simple while delegating the
actual training and evaluation pipeline to the reusable XOR module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.xor._cli import invoke_main
from mode_connectivity.xor import xor_train_linear_barriers


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/xor/runners/train_linear_barriers.yaml"


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        argv = list(cfg.get("argv", []))
    invoke_main(xor_train_linear_barriers.main, argv)


if __name__ == "__main__":
    main()
