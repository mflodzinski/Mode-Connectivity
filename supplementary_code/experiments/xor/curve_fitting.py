"""Config-backed runner for the XOR curve-fitting experiment.

The module keeps the public entrypoint stable while forwarding the concrete
argument list to the retained argparse implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.xor._cli import invoke_main
from mode_connectivity.xor import xor_curve_fitting


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/xor/runners/curve_fitting.yaml"


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        argv = list(cfg.get("argv", []))
    invoke_main(xor_curve_fitting.main, argv)


if __name__ == "__main__":
    main()
