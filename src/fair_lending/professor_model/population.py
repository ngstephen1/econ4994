"""Vectorized applicant draws and information-separated Step 2 exports."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.professor_model.config import (
    BANK_COLUMNS, DEFAULT_CONFIG_PATH, PREVIEW_COLUMNS, PREVIEW_SCOPE,
    PRIVATE_COLUMNS, PROJECT_ROOT, SCHEMA_VERSION, STREAM_NAMES,
    configuration_fingerprint, resolve_config,
)


def generate_applicants(
    config: dict[str, Any], *, n_samples: int | None = None, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate independent primitives and C; keep simulator variables private.

    All Normal scales are standard deviations. D is deliberately unbounded.
    Group affects only its own Bernoulli draw. No older scientific DGP is used.
    """
    resolved = resolve_config(config, n_samples=n_samples, seed=seed)
    n = resolved["simulation"]["n_samples"]
    master_seed = resolved["simulation"]["random_seed"]
    children = np.random.SeedSequence(master_seed).spawn(len(STREAM_NAMES))
    streams = {name: np.random.default_rng(child) for name, child in zip(STREAM_NAMES, children, strict=True)}
    draws: dict[str, np.ndarray] = {}
    for name in STREAM_NAMES:
        spec = resolved["population"][name]
        if name == "B":
            draws[name] = streams[name].binomial(1, spec["probability"], size=n).astype(np.int8)
        else:
            draws[name] = streams[name].normal(spec["mean"], spec["sd"], size=n)
    income = np.exp(draws["log_Y"])
    signal = resolved["credit_signal"]["epsilon_coefficient"] * draws["epsilon"] + draws["omega"]
    ids = [f"prof_{i:06d}" for i in range(1, n + 1)]
    bank = pd.DataFrame({"applicant_id": ids, "Y": income, "D": draws["D"], "B": draws["B"], "C": signal})
    private = pd.DataFrame({"applicant_id": ids, "log_Y": draws["log_Y"], "epsilon": draws["epsilon"], "omega": draws["omega"]})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "resolved_configuration": resolved,
        "configuration_sha256": configuration_fingerprint(resolved),
        "n_samples": n,
        "random_seed": master_seed,
        "random_streams": {
            name: {"spawn_key": list(child.spawn_key), "bit_generator": streams[name].bit_generator.__class__.__name__}
            for name, child in zip(STREAM_NAMES, children, strict=True)
        },
        "independence_assumption": "independent log_Y, D, B, epsilon, omega; C = coefficient * epsilon + omega",
        "normal_scale_interpretation": "standard_deviation",
        "dti_policy": "untruncated; no clipping or resampling",
        "bank_columns": list(BANK_COLUMNS),
        "private_columns": list(PRIVATE_COLUMNS),
        "audit_preview_scope": PREVIEW_SCOPE,
    }
    return bank, private, metadata


def evaluator_preview(bank: pd.DataFrame, private: pd.DataFrame, n_rows: int = 150) -> pd.DataFrame:
    """Join only the first n rows for an explicitly labeled evaluator preview."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or not 1 <= n_rows <= 200:
        raise ValueError("preview row count must be an integer between 1 and 200")
    preview = bank.head(n_rows).merge(private, on="applicant_id", validate="one_to_one")
    preview.insert(0, "information_scope", PREVIEW_SCOPE)
    return preview.loc[:, PREVIEW_COLUMNS]


def _source_provenance() -> dict[str, Any]:
    """Record actual new source contents, including uncommitted implementation."""
    files = sorted((PROJECT_ROOT / "src/fair_lending/professor_model").glob("*.py"))
    files.extend([DEFAULT_CONFIG_PATH, PROJECT_ROOT / "experiments/generate_professor_population.py"])
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files if path.is_file()
    }
    dependencies = {name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML")}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"source_sha256": source_hashes, "dependencies": dependencies, "python": platform.python_version(), "git_revision": revision, "git_dirty": dirty}


def save_population(
    bank: pd.DataFrame,
    private: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_root: Path = PROJECT_ROOT,
    reproducibility_checks: dict[str, bool] | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Validate first, then save CSV/Parquet data and complete regeneration metadata.

    Existing exports require an explicit overwrite flag. The output paths are
    restricted to the professor-model namespaces by the configuration contract.
    """
    from fair_lending.professor_model.validation import validate_population

    config = resolve_config(metadata["resolved_configuration"])
    report = validate_population(bank, private, config)
    if reproducibility_checks is not None:
        report["reproducibility_checks"] = reproducibility_checks
        if not all(reproducibility_checks.values()):
            report["errors"].append("reproducibility validation failed")
            report["valid"] = False
    if not report["valid"]:
        raise ValueError(f"population validation failed: {report['errors']}")
    root = Path(output_root)
    directory = root / config["outputs"]["data_directory"]
    paths = {
        "bank_csv": directory / "applicants_bank_view.csv",
        "bank_parquet": directory / "applicants_bank_view.parquet",
        "private_csv": directory / "applicants_simulator_private.csv",
        "private_parquet": directory / "applicants_simulator_private.parquet",
        "audit_preview_csv": directory / "applicants_audit_preview.csv",
        "metadata": directory / "population_metadata.json",
        "validation_report": root / config["outputs"]["validation_report"],
        "group_comparison": root / config["outputs"]["group_comparison"],
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"outputs already exist; use --overwrite to regenerate: {existing}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(paths["bank_csv"], index=False)
    private.to_csv(paths["private_csv"], index=False)
    bank.to_parquet(paths["bank_parquet"], index=False)
    private.to_parquet(paths["private_parquet"], index=False)
    evaluator_preview(bank, private, config["simulation"]["audit_preview_rows"]).to_csv(paths["audit_preview_csv"], index=False)
    pd.DataFrame(report["group_comparison"]).to_csv(paths["group_comparison"], index=False)
    paths["validation_report"].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    saved_metadata = {
        **metadata, **_source_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "information_scope": "BANK_OBSERVED" if name.startswith("bank_") else "EVALUATOR_ONLY",
            } for name, path in paths.items() if name != "metadata"
        },
    }
    paths["metadata"].write_text(json.dumps(saved_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return paths, report
