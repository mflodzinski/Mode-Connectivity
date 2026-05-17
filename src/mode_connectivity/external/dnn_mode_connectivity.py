"""Adapter helpers for the vendored ``dnn-mode-connectivity`` repository.

These functions centralize import-path setup and expose the upstream modules and
script locations that the retained curve workflows still depend on.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from mode_connectivity.common.paths import PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT / "external" / "dnn-mode-connectivity"


def add_to_path() -> Path:
    """Ensure the vendored repo is importable and return its root."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def load_data_module():
    add_to_path()
    return importlib.import_module("data")


def load_models_module():
    add_to_path()
    return importlib.import_module("models")


def load_curves_module():
    add_to_path()
    return importlib.import_module("curves")


def train_script_path() -> Path:
    add_to_path()
    return REPO_ROOT / "train.py"


def eval_curve_script_path() -> Path:
    add_to_path()
    return REPO_ROOT / "eval_curve.py"
