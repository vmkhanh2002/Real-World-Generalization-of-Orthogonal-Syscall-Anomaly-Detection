"""Shared defaults for the realtime evaluation pipeline.

These defaults are centralized here so individual CLIs do not hardcode
workspace paths or canonical artifact names in multiple places.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATASET = os.getenv("REALTIME_EVAL_DEFAULT_DATASET", "32bit")
DEFAULT_ARTIFACTS_DIR = Path(
    os.getenv("REALTIME_EVAL_ARTIFACTS_DIR", "artifacts/realtime_eval")
)
DEFAULT_DATASET_ROOT = Path(
    os.getenv("REALTIME_EVAL_DATASET_ROOT", f"datasets/{DEFAULT_DATASET}/ADFA-LD/ADFA-LD")
)
CANONICAL_FULL_TAG = os.getenv("REALTIME_EVAL_CANONICAL_LABEL", "full_100_manifest")

DEFAULT_HALF_MANIFEST = Path(
    os.getenv(
 "REALTIME_EVAL_HALF_MANIFEST",
        str(DEFAULT_ARTIFACTS_DIR / "adfa_ld_realtime_half_manifest.json"),
    )
)
DEFAULT_HALF_INPUTS = Path(
    os.getenv(
 "REALTIME_EVAL_HALF_INPUTS",
        str(DEFAULT_ARTIFACTS_DIR / "adfa_ld_realtime_half_inputs.txt"),
    )
)
DEFAULT_FULL_MANIFEST = Path(
    os.getenv(
 "REALTIME_EVAL_FULL_MANIFEST",
        str(DEFAULT_ARTIFACTS_DIR / "adfa_ld_realtime_full_manifest.json"),
    )
)
DEFAULT_FULL_INPUTS = Path(
    os.getenv(
 "REALTIME_EVAL_FULL_INPUTS",
        str(DEFAULT_ARTIFACTS_DIR / "adfa_ld_realtime_full_inputs.txt"),
    )
)
DEFAULT_CANONICAL_POLICY = Path(
    os.getenv(
 "REALTIME_EVAL_CANONICAL_POLICY",
        str(DEFAULT_ARTIFACTS_DIR / "POLICY.canonical.json"),
    )
)
DEFAULT_CANONICAL_FAMILY_ONSET = Path(
    os.getenv(
 "REALTIME_EVAL_CANONICAL_FAMILY_ONSET",
        str(DEFAULT_ARTIFACTS_DIR / f"family_onset_calibration.{CANONICAL_FULL_TAG}.json"),
    )
)


def artifact_path(name: str) -> Path:
    return DEFAULT_ARTIFACTS_DIR / name


def canonical_artifact_path(stem: str, suffix: str) -> Path:
    return artifact_path(f"{stem}.{CANONICAL_FULL_TAG}.{suffix}")
