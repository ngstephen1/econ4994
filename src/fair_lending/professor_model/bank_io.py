"""Read existing artifacts, freeze the bank model, then evaluate hidden truth.

Old manifests are verified by byte hash without parsing any private data table.
The evaluator truth loader requires a saved, verified freeze checkpoint.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fair_lending.professor_model.bank_regression import (
    DESIGN_COLUMNS, FEATURES, MODEL_ID, evaluate_bank_risk, ids_fingerprint,
    join_bank_inputs, predict_bank_ols,
)
from fair_lending.professor_model.config import PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/bank_regression.yaml"


def load_bank_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_bank_config(config)
    return config


def validate_bank_config(config: dict[str, Any]) -> None:
    """Keep the requested fit fixed; small split counts are supported in tests."""
    expected_model = {
        "family": "statsmodels_OLS", "solver": "pinv", "intercept": True,
        "predictors": list(FEATURES), "target": "PN", "regularization": None,
        "target_transform": "none", "prediction_rescaling": "none",
        "bounding": "clip_0_1_separate_column",
    }
    try:
        if config["schema_version"] != "professor-bank-regression-v1" or config["model_id"] != MODEL_ID or config["model"] != expected_model:
            raise ValueError("Step 5 requires the exact unregularized raw-PN OLS specification")
        split = config["split"]
        if split["spawn_key"] != [6] or split["bit_generator"] != "PCG64" or split["applicant_order"] != "ascending_lexicographic_applicant_id":
            raise ValueError("incorrect split stream/order")
        for key in ("train_rows", "evaluation_rows", "master_seed"):
            if type(split[key]) is not int or split[key] < (0 if key == "master_seed" else 1):
                raise ValueError("invalid split count or seed")
        if type(config["expected_input_rows"]) is not int or split["train_rows"] + split["evaluation_rows"] != config["expected_input_rows"]:
            raise ValueError("split must cover all input rows")
        if config["evaluation"]["scaled_true_risk_reference"] != .8 or type(config["evaluation"]["preview_rows"]) is not int or not 1 <= config["evaluation"]["preview_rows"] <= 150:
            raise ValueError("invalid evaluator reference/preview")
        for relative, prefix in (
            (config["input_directory"], "data/synthetic/professor_model"),
            *[(config["outputs"][f"{key}_directory"], f"results/{key}/professor_model") for key in ("tables", "metrics", "figures")],
        ):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
                raise ValueError("paths must remain within professor-model namespaces")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed bank configuration: {exc}") from exc


def load_bank_inputs(config: dict[str, Any], root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Read only bank-view/PN Parquet. Private tables are byte-hashed, not read."""
    validate_bank_config(config)
    directory = root / config["input_directory"]
    hashes: dict[str, str] = {}

    def register(relative: str, expected: str | None = None) -> None:
        path = Path(relative)
        allowed = ("data/synthetic/professor_model", "results/metrics/professor_model", "results/tables/professor_model", "results/figures/professor_model", "configs/professor_model", "src/fair_lending/professor_model", "experiments")
        if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(prefix) for prefix in allowed):
            raise ValueError("unexpected original-artifact path")
        actual = file_hash(root / path)
        if expected is not None and actual != expected:
            raise ValueError(f"original input checksum mismatch: {relative}")
        if relative in hashes and hashes[relative] != actual:
            raise ValueError("inconsistent prior manifests")
        hashes[relative] = actual

    for name, fingerprint_key, config_name in (
        ("population_metadata.json", "configuration_sha256", "baseline.yaml"),
        ("true_risk_metadata.json", "configuration_fingerprint", "true_risk.yaml"),
        ("noisy_signal_metadata.json", "configuration_fingerprint", "noisy_signal.yaml"),
    ):
        path = directory / name
        metadata = json.loads(path.read_text(encoding="utf-8"))
        register(str(path.relative_to(root)))
        if configuration_fingerprint(metadata["resolved_configuration"]) != metadata[fingerprint_key]:
            raise ValueError(f"configuration fingerprint mismatch: {name}")
        register(f"configs/professor_model/{config_name}", metadata.get("configuration_file_sha256"))
        for artifact in metadata["artifacts"].values():
            register(artifact["path"], artifact["sha256"])
        for field in ("source_sha256", "source_code_hashes", "input_data_checksums", "input_checksums"):
            for relative, expected in metadata.get(field, {}).items():
                register(relative, expected)
    schemas, frames = [], []
    for filename in ("applicants_bank_view.parquet", "training_signals.parquet"):
        path = directory / filename
        if str(path.relative_to(root)) not in hashes:
            raise ValueError(f"unregistered input: {filename}")
        frame = pd.read_parquet(path)
        frames.append(frame)
        schemas.append({"filename": filename, "shape": list(frame.shape), "columns": list(frame.columns)})
    joined = join_bank_inputs(*frames, config["expected_input_rows"])
    if not verify_input_hashes(hashes, root):
        raise ValueError("original inputs changed during loading")
    return joined, hashes, schemas


def output_paths(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Path]:
    directory = root / config["input_directory"]
    tables, metrics, figures = (root / config["outputs"][f"{key}_directory"] for key in ("tables", "metrics", "figures"))
    return {
        "splits_csv": directory / "applicant_splits.csv",
        "estimates_csv": directory / "bank_risk_estimates.csv",
        "estimates_parquet": directory / "bank_risk_estimates.parquet",
        "preview": directory / "bank_regression_audit_preview_EVALUATOR_ONLY.csv",
        "metadata": directory / "bank_regression_metadata.json",
        "model": metrics / "bank_regression_model.json",
        "validation": metrics / "bank_regression_validation.json",
        "coefficients": tables / "bank_regression_coefficients.csv",
        "evaluation_metrics": tables / "bank_regression_evaluation_metrics.csv",
        "evaluation_groups": tables / "bank_regression_evaluation_groups.csv",
        "prediction_summary": tables / "bank_regression_prediction_summary.csv",
        "scatter": figures / "bank_pd_raw_vs_true_pd.png",
        "histogram": figures / "bank_pd_raw_histogram.png",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def freeze_bank_artifacts(
    applicants: pd.DataFrame, splits: pd.DataFrame, predictions: pd.DataFrame,
    model: dict[str, Any], config: dict[str, Any], input_hashes: dict[str, str],
    *, root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH, overwrite: bool = False,
) -> dict[str, Any]:
    """Save bank-side artifacts first. No evaluator data is accepted or read."""
    validate_bank_config(config)
    paths = output_paths(config, root)
    if any(str(path.relative_to(root)) in input_hashes for path in paths.values()):
        raise ValueError("Step 5 output would overwrite an original artifact")
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 5 outputs exist; explicitly use --overwrite to replace only Step 5")
    if not verify_input_hashes(input_hashes, root):
        raise ValueError("original inputs changed before model freeze")
    if len(applicants) != config["expected_input_rows"] or splits.split.value_counts().to_dict() != {"train": config["split"]["train_rows"], "evaluation": config["split"]["evaluation_rows"]}:
        raise ValueError("incorrect full-population split")
    replay = predict_bank_ols(applicants, model)
    np.testing.assert_array_equal(predictions.bank_PD_raw.to_numpy(), replay)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(paths["splits_csv"], index=False)
    predictions.to_csv(paths["estimates_csv"], index=False)
    predictions.to_parquet(paths["estimates_parquet"], index=False)
    pd.DataFrame([{"term": term, "coefficient": model["coefficients"][term], "predictor_units": model["units"][term]} for term in DESIGN_COLUMNS]).to_csv(paths["coefficients"], index=False)
    frozen_model = {
        **model, "frozen_before_evaluator_truth": True,
        "configuration_fingerprint": configuration_fingerprint(config),
        "training_cohort": "train", "evaluation_cohort": "evaluation",
        "split_random_stream": config["split"],
        "evaluation_count": config["split"]["evaluation_rows"],
        "evaluation_ID_fingerprint": ids_fingerprint(applicants.loc[applicants.split == "evaluation", "applicant_id"]),
        "input_checksums": input_hashes,
    }
    _write_json(paths["model"], frozen_model)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["estimates_parquet"]), predictions, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(paths["estimates_csv"], float_precision="round_trip"), predictions, check_exact=True)
    saved = json.loads(paths["model"].read_text(encoding="utf-8"))
    np.testing.assert_array_equal(predict_bank_ols(applicants, saved), replay)
    names = ("splits_csv", "estimates_csv", "estimates_parquet", "coefficients", "model")
    return {"status": "frozen_before_evaluator_truth", "artifacts": {name: {"path": str(paths[name].relative_to(root)), "sha256": file_hash(paths[name])} for name in names}, "configuration_file_sha256": file_hash(config_path)}


def load_evaluator_truth(config: dict[str, Any], checkpoint: dict[str, Any], input_hashes: dict[str, str], root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """This is the first numeric truth read, gated on saved model/predictions."""
    if checkpoint.get("status") != "frozen_before_evaluator_truth" or set(checkpoint.get("artifacts", {})) != {"splits_csv", "estimates_csv", "estimates_parquet", "coefficients", "model"}:
        raise ValueError("model and predictions must be frozen before loading truth")
    for artifact in checkpoint["artifacts"].values():
        if file_hash(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen bank artifact changed before truth loading")
    model = json.loads((root / checkpoint["artifacts"]["model"]["path"]).read_text(encoding="utf-8"))
    if model.get("frozen_before_evaluator_truth") is not True or model["configuration_fingerprint"] != configuration_fingerprint(config):
        raise ValueError("invalid frozen model checkpoint")
    path = root / config["input_directory"] / "simulation_truth.parquet"
    if str(path.relative_to(root)) not in input_hashes or not verify_input_hashes(input_hashes, root):
        raise ValueError("unregistered/changed evaluator truth or original inputs")
    return pd.read_parquet(path)


def _save_figures(evaluation: pd.DataFrame, paths: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.1), layout="constrained")
    ax.scatter(evaluation.true_PD, evaluation.bank_PD_raw, s=5, alpha=.18, color="#235b79", rasterized=True)
    low, high = float(evaluation.true_PD.min()), float(evaluation.true_PD.max())
    ax.plot([low, high], [low, high], linestyle="--", color="#a94c40", linewidth=1.5, label="Identity: raw estimate = true PD")
    ax.axhline(0, color="0.4", linewidth=.8)
    ax.set(xlabel="Hidden true_PD (raw probability units)", ylabel="bank_PD_raw (raw prediction units)", title=f"Held-out risk recovery: {len(evaluation):,} applicants")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(paths["scatter"], dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.7), layout="constrained")
    ax.hist(evaluation.bank_PD_raw, bins=70, color="#235b79", edgecolor="white", linewidth=.3)
    ax.axvline(0, color="#a94c40", linestyle="--", linewidth=1.5, label="Zero")
    ax.set(xlabel="bank_PD_raw (raw prediction units)", ylabel="Evaluation applicants", title="Raw OLS predictions: all held-out observations")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(paths["histogram"], dpi=180)
    plt.close(fig)


def save_bank_evaluation(
    applicants: pd.DataFrame, predictions: pd.DataFrame, truth: pd.DataFrame,
    model: dict[str, Any], config: dict[str, Any], checkpoint: dict[str, Any], input_hashes: dict[str, str],
    *, root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Persist evaluator-only reports; never refit or rewrite the frozen files."""
    paths = output_paths(config, root)
    metrics, groups, preview, summary = evaluate_bank_risk(applicants, predictions, truth, model, config["evaluation"]["preview_rows"])
    metrics.to_csv(paths["evaluation_metrics"], index=False)
    groups.to_csv(paths["evaluation_groups"], index=False)
    pd.DataFrame([{"split": "evaluation", **summary}]).to_csv(paths["prediction_summary"], index=False)
    preview.to_csv(paths["preview"], index=False)
    evaluation = applicants.loc[applicants.split == "evaluation", ["applicant_id"]].merge(predictions, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one")
    _save_figures(evaluation, paths)
    unchanged_frozen = all(file_hash(root / a["path"]) == a["sha256"] for a in checkpoint["artifacts"].values())
    original_unchanged = verify_input_hashes(input_hashes, root)
    if not original_unchanged or not unchanged_frozen or file_hash(config_path) != checkpoint["configuration_file_sha256"]:
        raise RuntimeError("original inputs, configuration, or frozen artifacts changed during evaluation")
    report = {
        "valid": True, "checks": {"original_inputs_unchanged": original_unchanged, "frozen_bank_artifacts_unchanged": unchanged_frozen, "CSV_Parquet_exact_agreement": True, "coefficient_artifact_replay": True},
        "split_counts": applicants.split.value_counts().to_dict(),
        "design_rank": model["design_rank"], "design_condition_number": model["design_condition_number"],
        "evaluation_metrics": metrics.astype(object).where(metrics.notna(), None).to_dict(orient="records"), "group_summary": groups.to_dict(orient="records"),
        "raw_prediction_summary": summary,
        "notices": ["PN noise is large relative to the underlying risk signal.", "Y levels differ from the nonlinear log-income true-risk equation.", "Bounding is a safeguard, not calibration; boundary ties are retained.", "Group coefficients/gaps do not establish discrimination.", "Repository-wide checkpoint failures are separate from this validation."],
    }
    _write_json(paths["validation"], report)
    sources = [PROJECT_ROOT / f"src/fair_lending/professor_model/{name}.py" for name in ("bank_regression", "bank_io", "config", "risk_io")]
    sources.append(PROJECT_ROOT / "experiments/fit_professor_bank_regression.py")
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in sources}
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout)
    metadata = {
        "schema_version": config["schema_version"], "model_id": MODEL_ID, "status": "Step_5_validated",
        "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
        "configuration_file_sha256": file_hash(config_path), "source_code_hashes": source_hashes,
        "source_code_fingerprint": configuration_fingerprint(source_hashes), "input_checksums": input_hashes,
        "git_revision": git_revision, "git_dirty": git_dirty,
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow", "matplotlib", "PyYAML")},
        "python": platform.python_version(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "split_random_stream": config["split"], "freeze_checkpoint": checkpoint,
        "evaluation_ID_fingerprint": ids_fingerprint(applicants.loc[applicants.split == "evaluation", "applicant_id"]),
        "original_inputs_unchanged": True,
        "information_boundary": "Only bank view and PN read before freeze; true-risk table read afterward for evaluation only.",
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": file_hash(path)} for name, path in paths.items() if name != "metadata"},
    }
    _write_json(paths["metadata"], metadata)
    return report
