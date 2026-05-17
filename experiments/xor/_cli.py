"""Small helpers for wrapping retained XOR CLIs.

The XOR experiments still use argparse-heavy implementations, so these helpers
temporarily rewrite ``sys.argv`` to reuse them behind cleaner entrypoints.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def invoke_main(main_fn, argv: Sequence[str]) -> None:
    """Invoke an argparse-based main with a temporary argv."""

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0], *list(argv)]
    try:
        main_fn()
    finally:
        sys.argv = original_argv
