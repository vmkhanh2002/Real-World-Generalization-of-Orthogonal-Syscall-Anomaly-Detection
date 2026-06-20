"""Shared defaults for the syscall_abi workspace and CLI helpers."""

from __future__ import annotations

from pathlib import Path


WORKSPACE_ROOT = Path("tests") / "syscall_abi"
DEFAULT_INBOX_ROOT = WORKSPACE_ROOT / "inbox"
DEFAULT_ORGANIZED_ROOT = WORKSPACE_ROOT / "organized"
DEFAULT_TOP_K = 20
DEFAULT_SYNTHETIC_SEED = 20260325


def default_manifest_path(prefix: str) -> Path:
    return DEFAULT_INBOX_ROOT / f"{prefix}.manifest.json"
