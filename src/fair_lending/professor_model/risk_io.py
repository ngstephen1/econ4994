"""Read frozen Step 2 inputs and persist new truth with separate provenance."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from fair_lending.professor_model.config import PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.risk import (
    TRUTH_PREVIEW_COLUMNS, TRUE_RISK_CONFIG_PATH, validate_true_risk_config,
)
from fair_lending.professor_model.validation import validate_population


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_input_hashes(hashes: dict[str, str], root: Path = PROJECT_ROOT) -> bool:
    """Verify byte-level preservation, including the original Step 2 metadata."""
    return all((root / path).is_file() and file_hash(root / path) == expected for path, expected in hashes.items())


def load_existing_applicants(
    config: dict[str, Any], root: Path = PROJECT_ROOT
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load existing Parquet only; validate the Step 2 manifest without regeneration."""
    validate_true_risk_config(config)
    directory = root / config["input_directory"]
    metadata_path = directory / "population_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_sha256"]:
        raise ValueError("Step 2 resolved configuration fingerprint mismatch")
    hashes = {str(metadata_path.relative_to(root)): file_hash(metadata_path)}
    for artifact in metadata["artifacts"].values():
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts or not any(relative.is_relative_to(prefix) for prefix in ("data/synthetic/professor_model", "results/metrics/professor_model", "results/tables/professor_model")):
            raise ValueError("unexpected Step 2 artifact path")
        path = root / relative
        if file_hash(path) != artifact["sha256"]:
            raise ValueError(f"Step 2 artifact checksum mismatch: {relative}")
        hashes[str(relative)] = artifact["sha256"]
    for filename in ("applicants_bank_view.parquet", "applicants_simulator_private.parquet"):
        relative = str((directory / filename).relative_to(root))
        if relative not in hashes:
            raise ValueError(f"missing primary input checksum: {filename}")
    bank = pd.read_parquet(directory / "applicants_bank_view.parquet")
    private = pd.read_parquet(directory / "applicants_simulator_private.parquet")
    report = validate_population(bank, private, metadata["resolved_configuration"])
    if not report["valid"] or len(bank) != config["expected_input_rows"]:
        raise ValueError(f"existing population failed validation: {report['errors']}")
    return bank.merge(private, on="applicant_id", validate="one_to_one"), hashes


def save_true_risk_artifacts(
    applicants: pd.DataFrame,
    truth: pd.DataFrame,
    report: dict[str, Any],
    config: dict[str, Any],
    input_hashes: dict[str, str],
    *, root: Path = PROJECT_ROOT, overwrite: bool = False,
    config_path: Path = TRUE_RISK_CONFIG_PATH,
) -> dict[str, Path]:
    """Save only new Step 3 files, with independent metadata and verified inputs."""
    from fair_lending.professor_model.risk_figures import save_risk_figures
    from fair_lending.professor_model.risk_validation import validate_true_risk

    if not report["valid"] or not validate_true_risk(truth, applicants, config)["valid"]:
        raise ValueError("true-risk validation failed; no artifacts saved")
    if not verify_input_hashes(input_hashes, root):
        raise ValueError("original Step 2 inputs changed before export")
    directory = root / config["input_directory"]
    figure_directory = root / config["outputs"]["figures_directory"]
    paths = {
        "truth_csv": directory / "simulation_truth.csv",
        "truth_parquet": directory / "simulation_truth.parquet",
        "preview": directory / "simulation_truth_audit_preview_EVALUATOR_ONLY.csv",
        "metadata": directory / "true_risk_metadata.json",
        "validation_report": root / config["outputs"]["validation_report"],
        "distribution": root / config["outputs"]["summary"],
        "group_comparison": root / config["outputs"]["group_comparison"],
        "correlations": root / config["outputs"]["correlations"],
        **{name: figure_directory / f"{name}.png" for name in ("true_pd_histogram", "true_pd_vs_log_income", "true_pd_vs_dti")},
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 3 outputs already exist; use --overwrite explicitly")
    if {str(path.relative_to(root)) for path in paths.values()} & set(input_hashes):
        raise ValueError("output path would overwrite a Step 2 input")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(paths["truth_csv"], index=False)
    truth.to_parquet(paths["truth_parquet"], index=False)
    applicants.head(config["preview_rows"]).merge(truth, on="applicant_id", validate="one_to_one").loc[:, TRUTH_PREVIEW_COLUMNS].to_csv(paths["preview"], index=False)
    pd.DataFrame([{"variable": name, **{key: value for key, value in stats.items() if key != "quantiles"}, **stats["quantiles"]} for name, stats in report["distributions"].items()]).to_csv(paths["distribution"], index=False)
    pd.DataFrame(report["group_comparison"]).to_csv(paths["group_comparison"], index=False)
    pd.DataFrame(report["correlations"]).to_csv(paths["correlations"], index=False)
    save_risk_figures(applicants, truth, figure_directory)
    if not verify_input_hashes(input_hashes, root):
        raise RuntimeError("original Step 2 inputs changed during Step 3")
    report = {**report, "original_step2_files_unchanged": True, "input_data_checksums": input_hashes}
    paths["validation_report"].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    sources = [*sorted((PROJECT_ROOT / "src/fair_lending/professor_model").glob("*.py")), PROJECT_ROOT / "experiments/generate_professor_true_risk.py"]
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in sources}
    metadata = {
        "schema_version": config["schema_version"], "information_scope": "EVALUATOR_ONLY",
        "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
        "configuration_file_sha256": file_hash(config_path),
        "source_code_hashes": source_hashes,
        "source_code_fingerprint": configuration_fingerprint(source_hashes),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "pyarrow", "matplotlib", "PyYAML")},
        "python": platform.python_version(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_data_checksums": input_hashes, "original_step2_files_unchanged": True,
        "deterministic": True, "random_draws": False, "probability_floor_applied": False,
        "preview_scope": "EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING",
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": file_hash(path)} for name, path in paths.items() if name != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return paths
