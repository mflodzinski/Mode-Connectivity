"""Small plotting helpers that do not belong to a specific experiment.

These functions mainly standardize figure and text-summary output for the
repo-level plotting scripts.
"""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, output_path: str | Path, **savefig_kwargs) -> Path:
    """Save a Matplotlib figure and ensure the parent directory exists."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
    return path


def save_summary_text(lines: list[str], output_path: str | Path) -> Path:
    """Write a plaintext plotting summary next to a generated artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
