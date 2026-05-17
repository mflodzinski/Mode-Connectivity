"""Stable path constants for the repository layout.

These values give the rest of the codebase a single source of truth for the
project root, source tree, and vendored external directories.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
EXTERNAL_ROOT = PROJECT_ROOT / "external"

__all__ = ["PROJECT_ROOT", "SRC_ROOT", "EXTERNAL_ROOT"]
