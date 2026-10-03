"""Adapter layer for vendored upstream repositories.

These helpers centralize imports and path setup for the retained external code
so the rest of the library does not scatter ``sys.path`` mutations.
"""

from .dnn_mode_connectivity import (
    add_to_path as add_dnn_mode_connectivity_to_path,
    eval_curve_script_path,
    load_curves_module,
    load_data_module,
    load_models_module,
    train_script_path,
)
from .pytorch_vgg_cifar10 import load_vgg_module
from .sinkhorn_rebasin import import_external_sinkhorn, import_vgg_rebasin_components

__all__ = [
    "add_dnn_mode_connectivity_to_path",
    "eval_curve_script_path",
    "import_external_sinkhorn",
    "import_vgg_rebasin_components",
    "load_curves_module",
    "load_data_module",
    "load_models_module",
    "load_vgg_module",
    "train_script_path",
]
