"""Read frozen Step 5 artifacts; export Step 6A bank/evaluator tables separately."""

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

from fair_lending.professor_model.bank_regression import ESTIMATE_COLUMNS, MODEL_ID, ids_fingerprint, validate_ids
from fair_lending.professor_model.config import BANK_COLUMNS, PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.contracts import CONTRACT_COLUMNS, PAYMENT_RULE_ID, make_loan_contracts, scheduled_payments, validate_terms
from fair_lending.professor_model.profit import (
    BANK_PROFIT_COLUMNS, TRUE_PROFIT_COLUMNS, break_even_probability,
    economic_diagnostics, expected_cashflows,
)
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/loan_economics.yaml"


def load_economics_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_economics_config(config)
    return config


def validate_economics_config(config: dict[str, Any]) -> None:
    try:
        if config["schema_version"] != "professor-loan-economics-v1" or config["model_id"] != MODEL_ID:
            raise ValueError("unsupported economics schema or bank model")
        for name in ("expected_input_rows", "evaluation_rows", "preview_rows"):
            if type(config[name]) is not int or config[name] < 1:
                raise ValueError("row counts must be positive integers")
        if config["evaluation_rows"] >= config["expected_input_rows"] or config["preview_rows"] > 150:
            raise ValueError("invalid evaluation/preview size")
        contract = config["contract"]
        validate_terms(contract["T"], contract["theta"], contract["r"])
        if set(contract) != {"T", "theta", "r", "rate_basis", "payment_rule_id", "schedule", "principal"} or contract["rate_basis"] != "per_period" or contract["payment_rule_id"] != PAYMENT_RULE_ID or contract["schedule"] != "ell_times_one_plus_r_power_t" or contract["principal"] != "T_times_ell":
            raise ValueError("unsupported working contract interpretation")
        if config["risk"] != {"interpretation": "constant_conditional_per_period_default", "default_absorbing": True, "bank_probability": "bank_PD_bounded", "evaluator_probability": "true_PD"}:
            raise ValueError("survival must use conditional per-period default probabilities")
        if config["accounting"] != {"transaction_cost": 0, "discounting": False, "prepayment": False, "recovery": 0}:
            raise ValueError("Step 6A does not include costs, discounting, prepayment, or recovery")
        for relative, prefix in (
            (config["input_directory"], "data/synthetic/professor_model"),
            (config["outputs"]["data_directory"], "data/processed/professor_model"),
            (config["outputs"]["summary"], "results/tables/professor_model"),
            (config["outputs"]["distributions"], "results/tables/professor_model"),
            (config["outputs"]["validation"], "results/metrics/professor_model"),
        ):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
                raise ValueError("paths must stay in the separate professor-model namespaces")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed economics configuration: {exc}") from exc


def load_economic_inputs(config: dict[str, Any], root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """No generation, model refitting, PN, or numeric hidden-truth reads."""
    validate_economics_config(config)
    directory = root / config["input_directory"]
    metadata_path = directory / "bank_regression_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}

    def register(relative: str, expected: str | None = None) -> None:
        path = Path(relative)
        allowed = ("data/synthetic/professor_model", "results/metrics/professor_model", "results/tables/professor_model", "results/figures/professor_model", "configs/professor_model", "src/fair_lending/professor_model", "experiments")
        if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(prefix) for prefix in allowed):
            raise ValueError("unexpected existing-artifact path")
        actual = file_hash(root / path)
        if expected is not None and actual != expected:
            raise ValueError(f"existing input checksum mismatch: {relative}")
        if relative in hashes and hashes[relative] != actual:
            raise ValueError("inconsistent existing checksums")
        hashes[relative] = actual

    register(str(metadata_path.relative_to(root)))
    if metadata["status"] != "Step_5_validated" or metadata["model_id"] != MODEL_ID or configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_fingerprint"]:
        raise ValueError("Step 5 metadata does not describe a validated, unchanged configuration")
    register("configs/professor_model/bank_regression.yaml", metadata["configuration_file_sha256"])
    for field in ("input_checksums", "source_code_hashes"):
        for relative, expected in metadata[field].items():
            register(relative, expected)
    for artifact in metadata["artifacts"].values():
        register(artifact["path"], artifact["sha256"])
    frames, schemas = [], []
    for filename, columns in (("applicants_bank_view.parquet", BANK_COLUMNS), ("applicant_splits.csv", ("applicant_id", "split")), ("bank_risk_estimates.parquet", ESTIMATE_COLUMNS)):
        path = directory / filename
        if str(path.relative_to(root)) not in hashes:
            raise ValueError(f"unregistered primary input: {filename}")
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        validate_ids(frame, filename)
        if tuple(frame.columns) != columns or len(frame) != config["expected_input_rows"] or frame.isna().any().any():
            raise ValueError(f"incorrect schema/count or missing values: {filename}")
        frames.append(frame)
        schemas.append({"filename": filename, "shape": list(frame.shape), "columns": list(frame.columns)})
    bank, splits, estimates = frames
    if any(set(frame.applicant_id) != set(bank.applicant_id) for frame in frames):
        raise ValueError("input applicant ID sets do not match")
    if not np.isfinite(bank[["Y", "D", "B", "C"]].to_numpy(dtype=float)).all() or not (bank.Y > 0).all() or not bank.B.isin([0, 1]).all():
        raise ValueError("invalid observed applicant values")
    if not estimates.model_id.eq(MODEL_ID).all() or not estimates.split.isin(["train", "evaluation"]).all() or not splits.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid model or split labels")
    if not np.isfinite(estimates[["bank_PD_raw", "bank_PD_bounded"]].to_numpy()).all() or not np.array_equal(estimates.bank_PD_bounded, np.clip(estimates.bank_PD_raw, 0, 1)):
        raise ValueError("bounded predictions do not match frozen raw scores")
    if not np.array_equal(estimates.prediction_out_of_bounds, (estimates.bank_PD_raw < 0) | (estimates.bank_PD_raw > 1)):
        raise ValueError("invalid prediction range flags")
    joined = bank.merge(splits, on="applicant_id", validate="one_to_one").merge(estimates, on=["applicant_id", "split"], validate="one_to_one")
    if len(joined) != config["expected_input_rows"]:
        raise ValueError("prediction and split labels do not align; no rows may be dropped")
    evaluation = joined.loc[joined.split == "evaluation"].sort_values("applicant_id").reset_index(drop=True)
    if len(evaluation) != config["evaluation_rows"] or ids_fingerprint(evaluation.applicant_id) != metadata["evaluation_ID_fingerprint"]:
        raise ValueError("evaluation IDs/count differ from frozen Step 5")
    if not verify_input_hashes(hashes, root):
        raise ValueError("existing inputs changed while loading")
    return evaluation, hashes, schemas


def load_true_risk_for_economics(evaluation: pd.DataFrame, config: dict[str, Any], input_hashes: dict[str, str], root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Evaluator-only lookup of saved PD, never a risk recalculation."""
    path = root / config["input_directory"] / "simulation_truth.parquet"
    if str(path.relative_to(root)) not in input_hashes or not verify_input_hashes(input_hashes, root):
        raise ValueError("unregistered or changed original truth inputs")
    truth = pd.read_parquet(path)
    validate_ids(truth, "saved truth")
    if tuple(truth.columns) != ("applicant_id", "P_income", "P_other", "true_PD") or len(truth) != config["expected_input_rows"]:
        raise ValueError("saved truth has the wrong schema/count")
    if not np.isfinite(truth.true_PD.to_numpy()).all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid saved default probabilities")
    if not set(evaluation.applicant_id).issubset(truth.applicant_id):
        raise ValueError("evaluation IDs are absent from truth")
    return evaluation[["applicant_id"]].merge(truth[["applicant_id", "true_PD"]], on="applicant_id", validate="one_to_one")


def output_paths(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Path]:
    directory = root / config["outputs"]["data_directory"]
    return {
        **{f"{name}_{suffix}": directory / f"{name}.{suffix}" for name in ("loan_contracts", "bank_expected_profit", "true_expected_profit") for suffix in ("csv", "parquet")},
        "preview": directory / "loan_economics_preview_EVALUATOR_ONLY.csv",
        "metadata": directory / "loan_economics_metadata.json",
        **{key: root / config["outputs"][key] for key in ("summary", "distributions", "validation")},
    }


def validate_economic_outputs(contracts: pd.DataFrame, bank: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    if tuple(contracts.columns) != CONTRACT_COLUMNS or tuple(bank.columns) != BANK_PROFIT_COLUMNS or tuple(truth.columns) != TRUE_PROFIT_COLUMNS:
        raise ValueError("incorrect output schemas or hidden-truth leakage")
    for name, frame in (("contracts", contracts), ("bank profit", bank), ("true profit", truth)):
        validate_ids(frame, name)
        if set(frame.applicant_id) != set(contracts.applicant_id) or frame.isna().any().any():
            raise ValueError("output IDs/missing values are invalid")
    aligned_bank = contracts[["applicant_id"]].merge(bank, on="applicant_id", validate="one_to_one")
    aligned_true = contracts[["applicant_id"]].merge(truth, on="applicant_id", validate="one_to_one")
    for frame, probability, suffix in ((aligned_bank, "bank_PD_bounded", "bank"), (aligned_true, "true_PD", "true")):
        receipts, profit = expected_cashflows(contracts, frame[probability].to_numpy())
        np.testing.assert_allclose(frame[f"expected_receipts_{suffix}"], receipts, rtol=1e-13, atol=1e-8)
        np.testing.assert_allclose(frame[f"expected_profit_{suffix}"], profit, rtol=1e-13, atol=1e-8)
    n = len(contracts)
    p_break_even = break_even_probability(float(contracts.r.iloc[0]))
    p0_receipts, _ = expected_cashflows(contracts, np.zeros(n))
    p1_receipts, p1_profit = expected_cashflows(contracts, np.ones(n))
    _, break_even_profit = expected_cashflows(contracts, np.full(n, p_break_even))
    zero_rate = make_loan_contracts(contracts[["applicant_id", "Y"]], T=int(contracts["T"].iloc[0]), theta=float(contracts.theta.iloc[0]), r=0.)
    _, zero_rate_profit = expected_cashflows(zero_rate, np.zeros(n))
    np.testing.assert_allclose(p0_receipts, scheduled_payments(contracts).sum(axis=1), rtol=1e-14, atol=1e-8)
    np.testing.assert_array_equal(p1_receipts, np.zeros(n))
    np.testing.assert_array_equal(p1_profit, -contracts.L_i.to_numpy())
    tolerance = 1e-12 * contracts.L_i.to_numpy()
    if not (np.abs(break_even_profit) <= tolerance).all() or not (np.abs(zero_rate_profit) <= tolerance).all():
        raise ValueError("break-even/zero-interest accounting check failed")
    return {"valid": True, "checks": {"principal_identity": True, "survival_weighted_receipts": True, "principal_subtracted_once": True, "nonnegative_receipts": True, "p_zero_all_payments": True, "p_one_zero_receipts": True, "zero_rate_zero_default_zero_profit": True, "break_even": True, "bank_truth_information_separation": True}, "break_even_probability_per_period": p_break_even, "break_even_max_abs_profit_USD": float(np.abs(break_even_profit).max()), "break_even_relative_tolerance": 1e-12}


def save_economic_artifacts(
    evaluation: pd.DataFrame, contracts: pd.DataFrame, bank: pd.DataFrame, truth: pd.DataFrame,
    config: dict[str, Any], input_hashes: dict[str, str], *, root: Path = PROJECT_ROOT,
    config_path: Path = CONFIG_PATH, overwrite: bool = False,
) -> tuple[dict[str, Path], dict[str, Any], pd.DataFrame]:
    """Write only new Step 6A outputs after accounting and preservation checks."""
    validate_economics_config(config)
    paths = output_paths(config, root)
    if any(str(path.relative_to(root)) in input_hashes for path in paths.values()) or len(set(paths.values())) != len(paths):
        raise ValueError("output path collision or original-input overwrite")
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 6A outputs exist; use --overwrite explicitly for Step 6A only")
    if not verify_input_hashes(input_hashes, root):
        raise ValueError("original inputs changed before export")
    if len(evaluation) != config["evaluation_rows"] or not evaluation.split.eq("evaluation").all():
        raise ValueError("only the complete held-out evaluation cohort is allowed")
    if not contracts["T"].eq(config["contract"]["T"]).all() or not contracts.theta.eq(config["contract"]["theta"]).all() or not contracts.r.eq(config["contract"]["r"]).all():
        raise ValueError("contracts do not match the configured terms")
    report = validate_economic_outputs(contracts, bank, truth)
    summary, distributions, joined = economic_diagnostics(evaluation, contracts, bank, truth)
    preview = joined.head(config["preview_rows"])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in (("loan_contracts", contracts), ("bank_expected_profit", bank), ("true_expected_profit", truth)):
        frame.to_csv(paths[f"{name}_csv"], index=False)
        frame.to_parquet(paths[f"{name}_parquet"], index=False)
        pd.testing.assert_frame_equal(pd.read_csv(paths[f"{name}_csv"], float_precision="round_trip"), frame, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_parquet(paths[f"{name}_parquet"]), frame, check_exact=True)
    summary.to_csv(paths["summary"], index=False)
    distributions.to_csv(paths["distributions"], index=False)
    preview.to_csv(paths["preview"], index=False)
    if not verify_input_hashes(input_hashes, root):
        raise RuntimeError("original inputs changed during export")
    report["checks"].update(original_inputs_unchanged=True, CSV_Parquet_exact_agreement=True, evaluation_only=True)
    report["summary"] = summary.to_dict(orient="records")
    report["n_evaluation"] = len(evaluation)
    report["notices"] = ["Stylized growing payments, not conventional mortgage amortization.", "r=0.05 is per period; no annual conversion or calendar duration is inferred.", "Existing PD is interpreted as constant conditional per-period default for Step 6A only.", "Expected potential profits are not lending decisions or realized profits.", "No discrimination conclusion follows from group differences."]
    paths["validation"].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    sources = [PROJECT_ROOT / f"src/fair_lending/professor_model/{name}.py" for name in ("contracts", "profit", "economics_io", "config", "risk_io", "bank_regression")]
    sources.append(PROJECT_ROOT / "experiments/calculate_professor_loan_economics.py")
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in sources}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout)
    metadata = {
        "schema_version": config["schema_version"], "status": "Step_6A_validated", "information_scope": "EVALUATOR_ONLY",
        "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path),
        "source_code_hashes": source_hashes, "source_code_fingerprint": configuration_fingerprint(source_hashes), "input_checksums": input_hashes,
        "evaluation_ID_fingerprint": ids_fingerprint(evaluation.applicant_id), "n_evaluation": len(evaluation),
        "random_draws": 0, "regression_refitted": False, "raw_predictions_modified": False,
        "git_revision": revision, "git_dirty": dirty, "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML")},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schemas": {"contracts": list(contracts.columns), "bank": list(bank.columns), "truth": list(truth.columns), "preview": list(preview.columns)},
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": file_hash(path), "information_role": "BANK_OBSERVABLE" if name.startswith(("loan_contracts_", "bank_expected_profit_")) else "EVALUATOR_ONLY"} for name, path in paths.items() if name != "metadata"},
        "original_inputs_unchanged": True,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return paths, report, preview
