"""Small cross-cutting helpers shared across many modules.

These exports mostly cover stable path constants and utility functions that are
cheap to depend on from anywhere in the library.
"""

from .paths import EXTERNAL_ROOT, PROJECT_ROOT, SRC_ROOT
from .utils import get_git_commit, set_global_seed, worker_init_fn

__all__ = [
    "EXTERNAL_ROOT",
    "PROJECT_ROOT",
    "SRC_ROOT",
    "get_git_commit",
    "set_global_seed",
    "worker_init_fn",
]
