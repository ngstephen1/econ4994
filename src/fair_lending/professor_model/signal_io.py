"""Read-only Step 2/3 loading and separate Step 4 export/provenance."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.professor_model.config import BANK_COLUMNS, PRIVATE_COLUMNS, PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.noisy_signal import (
    SIGNAL_CONFIG_PATH, noisy_audit_preview, validate_noisy_signal, validate_signal_config,
)
from fair_lending.professor_model.risk import TRUTH_COLUMNS
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


def join_signal_inputs(
    bank: pd.DataFrame, private: pd.DataFrame, truth: pd.DataFrame, expected_rows: int
) -> pd.DataFrame:
    """Reject imperfect joins; inspect saved probabilities without recalculating risk."""
    for name, frame, columns in (
        ("bank", bank, BANK_COLUMNS), ("private", private, PRIVATE_COLUMNS), ("truth", truth, TRUTH_COLUMNS),
    ):
        if tuple(frame.columns) != columns or len(frame) != expected_rows:
            raise ValueError(f"{name} requires exact schema and row count")
        if frame.isna().any().any() or not frame.applicant_id.is_unique:
            raise ValueError(f"{name} has missing values or duplicate IDs")
        if set(frame.applicant_id) != set(bank.applicant_id):
            raise ValueError(f"{name} applicant IDs do not match bank IDs")
        try:
            finite = np.isfinite(frame.drop(columns="applicant_id").to_numpy(dtype=float)).all()
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{name} has nonnumeric or infinite values")
    if not bank.B.isin([0, 1]).all() or not (bank.Y > 0).all():
        raise ValueError("saved bank input has invalid group/income values")
    for column in TRUTH_COLUMNS[1:]:
        if not truth[column].between(0, 1, inclusive="neither").all():
            raise ValueError("saved true-risk components require strict probability bounds")
    return bank.merge(private, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one")


def load_signal_inputs(
    config: dict[str, Any], root: Path = PROJECT_ROOT
) -> tuple[pd.DataFrame, dict[str, str], list[str], list[dict[str, Any]]]:
    """Validate old manifests and load existing tables; no generation calls."""
    validate_signal_config(config)
    directory = root / config["input_directory"]
    hashes: dict[str, str] = {}

    def register(relative: str, expected: str | None = None) -> None:
        path = Path(relative)
        prefixes = (
            "data/synthetic/professor_model", "results/metrics/professor_model",
            "results/tables/professor_model", "results/figures/professor_model",
            "configs/professor_model", "src/fair_lending/professor_model", "experiments",
        )
        if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(prefix) for prefix in prefixes):
            raise ValueError("unexpected old-artifact path")
        actual = file_hash(root / path)
        if expected is not None and actual != expected:
            raise ValueError(f"original input checksum mismatch: {relative}")
        if relative in hashes and hashes[relative] != actual:
            raise ValueError(f"inconsistent original artifact: {relative}")
        hashes[relative] = actual

    for name, fingerprint_key in (("population_metadata.json", "configuration_sha256"), ("true_risk_metadata.json", "configuration_fingerprint")):
        path = directory / name
        metadata = json.loads(path.read_text(encoding="utf-8"))
        register(str(path.relative_to(root)))
        if configuration_fingerprint(metadata["resolved_configuration"]) != metadata[fingerprint_key]:
            raise ValueError(f"configuration fingerprint mismatch in {name}")
        for artifact in metadata["artifacts"].values():
            register(artifact["path"], artifact["sha256"])
        for field in ("input_data_checksums", "source_sha256", "source_code_hashes"):
            for relative, expected in metadata.get(field, {}).items():
                register(relative, expected)
        if name == "true_risk_metadata.json":
            register("configs/professor_model/true_risk.yaml", metadata["configuration_file_sha256"])
    register("configs/professor_model/baseline.yaml")
    frames, schemas = [], []
    for filename in ("applicants_bank_view.parquet", "applicants_simulator_private.parquet", "simulation_truth.parquet"):
        path = directory / filename
        if str(path.relative_to(root)) not in hashes:
            raise ValueError(f"primary input is not covered by saved checksums: {filename}")
        frame = pd.read_parquet(path)
        frames.append(frame)
        schemas.append({"filename": filename, "shape": list(frame.shape), "columns": list(frame.columns)})
    joined = join_signal_inputs(*frames, config["expected_input_rows"])
    preview_path = directory / "simulation_truth_audit_preview_EVALUATOR_ONLY.csv"
    if str(preview_path.relative_to(root)) not in hashes:
        raise ValueError("Step 3 preview is not covered by saved checksums")
    preview = pd.read_csv(preview_path)
    ids = preview.applicant_id.tolist()
    if len(ids) != min(config["preview_rows"], len(joined)) or len(set(ids)) != len(ids) or not set(ids).issubset(joined.applicant_id):
        raise ValueError("Step 3 preview has invalid IDs or size")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original inputs changed while loading")
    return joined, hashes, ids, schemas


def save_signal_artifacts(
    applicants: pd.DataFrame, targets: pd.DataFrame, noise: pd.DataFrame,
    config: dict[str, Any], input_hashes: dict[str, str], preview_ids: list[str],
    *, root: Path = PROJECT_ROOT, overwrite: bool = False,
    config_path: Path = SIGNAL_CONFIG_PATH,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Persist exact raw tables and only new Step 4 files after validation."""
    from fair_lending.professor_model.signal_figures import save_signal_figures

    report = validate_noisy_signal(applicants, targets, noise, config)
    if not report["valid"]:
        raise ValueError(f"signal validation failed: {report['errors']}")
    if not verify_input_hashes(input_hashes, root):
        raise ValueError("original inputs changed before export")
    preview = noisy_audit_preview(applicants, targets, noise, preview_ids)
    directory = root / config["input_directory"]
    figures = root / config["outputs"]["figures_directory"]
    paths = {
        "targets_csv": directory / "training_signals.csv",
        "targets_parquet": directory / "training_signals.parquet",
        "noise_csv": directory / "risk_measurement_noise_private.csv",
        "noise_parquet": directory / "risk_measurement_noise_private.parquet",
        "preview": directory / "noisy_signal_audit_preview_EVALUATOR_ONLY.csv",
        "metadata": directory / "noisy_signal_metadata.json",
        "validation_report": root / config["outputs"]["validation_report"],
        "summary": root / config["outputs"]["summary"],
        "group_comparison": root / config["outputs"]["group_comparison"],
        "histogram": figures / "pn_histogram.png",
        "scatter": figures / "pn_vs_true_pd.png",
    }
    if {str(path.relative_to(root)) for path in paths.values()} & set(input_hashes):
        raise ValueError("output would overwrite an original input")
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 4 outputs already exist; use --overwrite explicitly")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(paths["targets_csv"], index=False)
    targets.to_parquet(paths["targets_parquet"], index=False)
    noise.to_csv(paths["noise_csv"], index=False)
    noise.to_parquet(paths["noise_parquet"], index=False)
    preview.to_csv(paths["preview"], index=False)
    # Round-trip floats are verified without rounding the source values.
    for name, frame in (("targets", targets), ("noise", noise)):
        pd.testing.assert_frame_equal(pd.read_parquet(paths[f"{name}_parquet"]), frame, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_csv(paths[f"{name}_csv"], float_precision="round_trip"), frame, check_exact=True)
    pd.DataFrame([{"variable": variable, **{key: value for key, value in stats.items() if key != "quantiles"}, **stats["quantiles"]} for variable, stats in report["distributions"].items()]).to_csv(paths["summary"], index=False)
    pd.DataFrame(report["group_comparison"]).to_csv(paths["group_comparison"], index=False)
    save_signal_figures(applicants, targets, figures)
    if not verify_input_hashes(input_hashes, root):
        raise RuntimeError("original inputs changed during Step 4")
    report["checks"].update({"CSV_Parquet_agreement": True, "original_inputs_unchanged": True, "same_Step3_preview_ids": preview.applicant_id.tolist() == preview_ids})
    report["input_checksums"] = input_hashes
    paths["validation_report"].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    source_paths = [*sorted((PROJECT_ROOT / "src/fair_lending/professor_model").glob("*.py")), PROJECT_ROOT / "experiments/generate_professor_noisy_signal.py"]
    sources = {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in source_paths}
    roles = {"targets_csv": "BANK_FUTURE_REGRESSION_TARGET", "targets_parquet": "BANK_FUTURE_REGRESSION_TARGET"}
    metadata = {
        "schema_version": config["schema_version"], "resolved_configuration": config,
        "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path),
        "source_code_hashes": sources, "source_code_fingerprint": configuration_fingerprint(sources),
        "random_stream": report["random_stream"], "input_checksums": input_hashes,
        "original_inputs_unchanged": True, "preview_applicant_ids": preview_ids,
        "schemas": {"targets": list(targets.columns), "noise_private": list(noise.columns), "evaluator_preview": list(preview.columns)},
        "transformations": {"signal_scale": 0.8, "clipping": False, "sigmoid": False, "noise_recentered": False, "noise_redrawn": False},
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "matplotlib", "PyYAML")},
        "python": platform.python_version(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": file_hash(path), "information_role": roles.get(name, "EVALUATOR_ONLY")} for name, path in paths.items() if name != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return paths, report
