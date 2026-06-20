"""Shared utility functions for the realtime_eval pipeline.

All four pipeline scripts originally maintained local copies of these helpers.
This module is the single source of truth. Import from here instead of
duplicating across build_stream_episodes, evaluate_realtime_detect,
export_window_scores, and inference_backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


# Path helpers

def resolve_project_path(project_root: Path, raw: str | Path) -> Path:
    """Return *raw* as an absolute path, resolved relative to *project_root* if needed."""
    path = Path(raw)
    return path if path.is_absolute() else (project_root / path)


def path_for_output(project_root: Path, path: Path) -> str:
    """Return a portable, project-relative string representation of *path*.

    Canonical version: uses ``path.resolve()`` and ``project_root.resolve()``
    (from export_window_scores.py:96) so that symlinks and relative CWDs cannot
    produce diverged strings across pipeline stages.
    """
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def canonical_path(text: str) -> str:
    """Normalise a path string for cross-platform dictionary lookups.

    Converts backslashes to forward slashes and lowercases the result so that
    Windows and Linux paths compare equal when used as manifest index keys.
    """
    return text.replace("\\", "/").lower()


# Sequence helpers

def max_consecutive_true(mask: Sequence[bool]) -> int:
    """Return the length of the longest run of ``True`` values in *mask*."""
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)
