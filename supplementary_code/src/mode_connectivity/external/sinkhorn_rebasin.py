"""Load vendored Sinkhorn-rebasin and related upstream VGG components.

This module is the main adapter boundary to ``external/sinkhorn-rebasin`` and
the extra upstream modules needed by the retained scale-aware alignment code.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from mode_connectivity.common.paths import PROJECT_ROOT


project_root = PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))


def ensure_upstream_paths() -> tuple[Path, Path, Path]:
    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    dnn_root = project_root / "external" / "dnn-mode-connectivity"

    if not sinkhorn_root.exists():
        raise RuntimeError(
            f"Missing vendored repo at {sinkhorn_root}. "
            "Initialize the git submodule on this checkout first."
        )

    for path in (str(examples_root), str(sinkhorn_root), str(dnn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    return sinkhorn_root, examples_root, dnn_root


def load_upstream_vgg_class():
    _, examples_root, _ = ensure_upstream_paths()
    sinkhorn_vgg_path = examples_root / "models" / "vgg.py"
    if not sinkhorn_vgg_path.exists():
        raise RuntimeError(
            f"Expected external VGG file at {sinkhorn_vgg_path}, but it does not exist. "
            "This usually means the sinkhorn-rebasin submodule was not checked out correctly."
        )

    spec = importlib.util.spec_from_file_location("_sinkhorn_rebasin_examples_vgg", sinkhorn_vgg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sinkhorn-rebasin VGG definition from {sinkhorn_vgg_path}.")

    vgg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgg_module)
    return vgg_module.VGG


def import_external_sinkhorn():
    """Import the vendored external Sinkhorn implementation and VGG model."""

    ensure_upstream_paths()
    VGG = load_upstream_vgg_class()

    try:
        rebasin_module = importlib.import_module("rebasin")
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import external/sinkhorn-rebasin. "
            "This is usually either a missing dependency in the vendored repo "
            "(notably `torchviz` and `graphviz`) or a module-path collision."
        ) from exc

    return VGG, rebasin_module.RebasinNet, rebasin_module.matching


def import_vgg_rebasin_components() -> tuple[Any, ...]:
    """Import the full upstream component bundle used by active VGG/CIFAR flows."""

    VGG, RebasinNet, matching = import_external_sinkhorn()

    dnn_data = importlib.import_module("data")
    rebasin_loss = importlib.import_module("rebasin.loss")
    dnn_utils = importlib.import_module("utils")

    return (
        VGG,
        RebasinNet,
        matching,
        dnn_data,
        rebasin_loss.DistL1Loss,
        rebasin_loss.DistL2Loss,
        rebasin_loss.MidLoss,
        rebasin_loss.RndLoss,
        dnn_utils.eval_loss_acc,
        dnn_utils.lerp,
    )
