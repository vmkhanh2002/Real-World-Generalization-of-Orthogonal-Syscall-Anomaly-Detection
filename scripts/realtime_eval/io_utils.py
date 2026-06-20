"""Shared text/JSON readers for realtime_eval scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text_allow_bom(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def load_json(path: Path) -> Any:
    return json.loads(read_text_allow_bom(path))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_text_allow_bom(path).splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
