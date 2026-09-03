"""Atomic per-run persistence and validated resume support."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fair_lending.sensitivity.design import ANALYSIS_VERSION
from fair_lending.simulation.config import PROJECT_ROOT
from fair_lending.simulation.generator import SYNTHETIC_SCHEMA_VERSION


def run_directory(project_root: Path | str | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "results" / "metrics" / "sensitivity_runs"


def record_path(run_id: str, project_root: Path | str | None = None) -> Path:
    return run_directory(project_root) / f"{run_id}.json"


def write_record_atomic(record: dict[str, Any], path: Path) -> None:
    """Replace one JSON record atomically after a complete serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_matching_record(
    path: Path,
    *,
    expected_run_id: str,
    expected_config_fingerprint: str,
) -> dict[str, Any] | None:
    """Return only a complete record matching the current scientific identity."""
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("status") != "completed"
        or record.get("analysis_version") != ANALYSIS_VERSION
        or record.get("schema_version") != SYNTHETIC_SCHEMA_VERSION
        or record.get("run_id") != expected_run_id
        or record.get("config_fingerprint") != expected_config_fingerprint
    ):
        return None
    return record
