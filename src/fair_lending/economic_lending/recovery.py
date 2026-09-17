"""Transparent provenance and immutable operational evidence (no scientific settings)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from fair_lending.economic_lending.config import PROJECT_ROOT


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scientific_provenance(root: Path = PROJECT_ROOT) -> dict:
    """Hash actual files, including untracked/dirty source, not merely HEAD."""
    paths = sorted(set(root.glob("src/fair_lending/economic_lending/*.py"))
                   | set(root.glob("configs/economic_lending/*.yaml"))
                   | set(root.glob("experiments/run_systemic_monte_carlo.py")))
    result = {
        "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "dependencies": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow", "pyyaml"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return {**result, "fingerprint": digest(result)}


def save_attempt(path: Path, record: dict) -> None:
    """Keep every attempt before updating a latest-status pointer."""
    stamp = datetime.now(timezone.utc).isoformat()
    attempt_id = uuid.uuid4().hex
    value = {**record, "attempt_id": attempt_id, "recorded_at_utc": stamp}
    atomic_json(path.parent / "attempts" / path.stem / f"{attempt_id}.json", value)
    atomic_json(path, record)
