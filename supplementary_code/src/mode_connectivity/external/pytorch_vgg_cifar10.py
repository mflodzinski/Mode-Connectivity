"""Adapter helpers for the vendored ``pytorch-vgg-cifar10`` repository.

The module hides the path manipulation required to import the upstream VGG
implementation from the rest of the codebase.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from mode_connectivity.common.paths import PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT / "external" / "pytorch-vgg-cifar10"


def add_to_path() -> Path:
    """Ensure the vendored repo is importable and return its root."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def load_vgg_module():
    add_to_path()
    return importlib.import_module("vgg")
