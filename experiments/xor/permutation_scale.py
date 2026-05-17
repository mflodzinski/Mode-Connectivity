"""Config-backed runner for the XOR permutation-and-scale experiment.

It optionally loads a default YAML argument list and then invokes the retained
CLI implementation without changing the underlying experiment logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.xor._cli import invoke_main
from mode_connectivity.xor import xor_permutation_scale_experiment


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/xor/runners/permutation_scale.yaml"


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        argv = list(cfg.get("argv", []))
    invoke_main(xor_permutation_scale_experiment.main, argv)


if __name__ == "__main__":
    main()
