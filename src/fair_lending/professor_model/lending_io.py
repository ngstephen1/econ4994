"""Step 6B saved-input validation, bank-decision freeze, and post-freeze audit."""

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

from fair_lending.professor_model.bank_regression import MODEL_ID, ids_fingerprint, validate_ids
from fair_lending.professor_model.config import PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.contracts import validate_contracts
from fair_lending.professor_model.lending import BANK_INPUT_COLUMNS, capital_budget, evaluate_portfolio, validate_bank_inputs, validate_solution
from fair_lending.professor_model.profit import BANK_PROFIT_COLUMNS, TRUE_PROFIT_COLUMNS
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/lending_threshold.yaml"


def load_lending_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        if config["schema_version"] != "professor-lending-threshold-v1" or config["model_id"] != MODEL_ID:
            raise ValueError("unsupported threshold schema/model")
        policy = config["policy"]
        expected = {"risk_column": "bank_PD_bounded", "budget_fraction": .4, "tied_risks": "fund_all_or_none", "no_lending": "explicit_null_cutoff", "profit_tie_break": "lowest_cutoff_reject_all_first", "summation": "exact_binary_float_integer_prefixes"}
        if policy != expected:
            raise ValueError("Step 6B requires the declared 40% common-cutoff policy")
        if any(type(config[key]) is not int or config[key] < 1 for key in ("expected_population_rows", "evaluation_rows")) or config["evaluation_rows"] >= config["expected_population_rows"]:
            raise ValueError("invalid cohort counts")
        for relative, prefix in (
            (config["input_directory"], "data/processed/professor_model"),
            (config["synthetic_directory"], "data/synthetic/professor_model"),
            *[(config["outputs"][f"{key}_directory"], f"{base}/professor_model") for key, base in (("data", "data/processed"), ("tables", "results/tables"), ("metrics", "results/metrics"))],
            (config["outputs"]["figure"], "results/figures/professor_model"),
        ):
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or not p.is_relative_to(prefix):
                raise ValueError("paths must remain in professor-model namespaces")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid lending configuration: {exc}") from exc
    return config


def load_lending_inputs(config: dict[str, Any], root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    """Read saved contracts, bank profits, and splits; no numeric truth reads."""
    directory = root / config["input_directory"]
    metadata_path = directory / "loan_economics_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    def register(relative: str, expected: str | None = None) -> None:
        path = Path(relative)
        prefixes = ("data/synthetic/professor_model", "data/processed/professor_model", "results/metrics/professor_model", "results/tables/professor_model", "results/figures/professor_model", "configs/professor_model", "src/fair_lending/professor_model", "experiments")
        if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(prefix) for prefix in prefixes):
            raise ValueError("unexpected original-artifact path")
        actual = file_hash(root / path)
        if expected is not None and actual != expected:
            raise ValueError(f"original checksum mismatch: {relative}")
        if relative in hashes and hashes[relative] != actual:
            raise ValueError("inconsistent prior checksums")
        hashes[relative] = actual
    register(str(metadata_path.relative_to(root)))
    if metadata["status"] != "Step_6A_validated" or configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_fingerprint"]:
        raise ValueError("Step 6A must be validated with its original configuration")
    register("configs/professor_model/loan_economics.yaml", metadata["configuration_file_sha256"])
    for field in ("input_checksums", "source_code_hashes"):
        for relative, expected in metadata[field].items():
            register(relative, expected)
    for artifact in metadata["artifacts"].values():
        register(artifact["path"], artifact["sha256"])
    sources = (directory / "loan_contracts.parquet", directory / "bank_expected_profit.parquet", root / config["synthetic_directory"] / "applicant_splits.csv")
    frames, schemas = [], []
    for path in sources:
        if str(path.relative_to(root)) not in hashes:
            raise ValueError("primary input is not covered by original checksums")
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        validate_ids(frame, path.name)
        if frame.isna().any().any():
            raise ValueError("missing values in saved lending inputs")
        frames.append(frame)
        schemas.append({"filename": path.name, "shape": list(frame.shape), "columns": list(frame.columns)})
    contracts, bank, splits = frames
    validate_contracts(contracts)
    if len(contracts) != config["evaluation_rows"] or tuple(bank.columns) != BANK_PROFIT_COLUMNS or set(bank.applicant_id) != set(contracts.applicant_id):
        raise ValueError("saved contract/bank schemas, counts or IDs do not match")
    if tuple(splits.columns) != ("applicant_id", "split") or len(splits) != config["expected_population_rows"] or not splits.split.isin(["train", "evaluation"]).all():
        raise ValueError("invalid saved cohort assignments")
    if set(splits.loc[splits.split == "evaluation", "applicant_id"]) != set(contracts.applicant_id) or ids_fingerprint(contracts.applicant_id) != metadata["evaluation_ID_fingerprint"]:
        raise ValueError("contracts are not exactly the authoritative evaluation cohort")
    joined = contracts[["applicant_id", "L_i"]].merge(bank, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"})
    if not np.isfinite(joined[["expected_receipts_bank", "expected_profit_bank"]].to_numpy()).all() or (joined.expected_receipts_bank < 0).any():
        raise ValueError("invalid saved economic values")
    np.testing.assert_allclose(joined.expected_profit_bank, joined.expected_receipts_bank - joined.requested_principal, rtol=1e-13, atol=1e-8)
    bank_input = joined.loc[:, BANK_INPUT_COLUMNS].sort_values("applicant_id").reset_index(drop=True)
    validate_bank_inputs(bank_input)
    if not verify_input_hashes(hashes, root):
        raise ValueError("original artifacts changed while loading")
    return bank_input, hashes, schemas


def output_paths(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Path]:
    directory = root / config["outputs"]["data_directory"]
    tables = root / config["outputs"]["tables_directory"]
    metrics = root / config["outputs"]["metrics_directory"]
    return {"decisions_csv": directory / "lending_decisions.csv", "decisions_parquet": directory / "lending_decisions.parquet", "metadata": directory / "lending_metadata.json", "portfolio_summary": tables / "lending_portfolio_summary.csv", "candidates": tables / "lending_threshold_candidates.csv", "group_audit": tables / "lending_group_audit.csv", "evaluator_summary": tables / "lending_evaluator_summary.csv", "selection": metrics / "lending_selection.json", "validation": metrics / "lending_validation.json", "figure": root / config["outputs"]["figure"]}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def freeze_lending_decisions(bank: pd.DataFrame, decisions: pd.DataFrame, candidates: pd.DataFrame, summary: dict[str, Any], config: dict[str, Any], hashes: dict[str, str], *, root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH, overwrite: bool = False) -> dict[str, Any]:
    """Persist and verify the bank's complete choice before evaluator truth loads."""
    if len(bank) != config["evaluation_rows"] or capital_budget(bank, config["policy"]["budget_fraction"])[1] != summary["available_bank_capital"]:
        raise ValueError("bank budget must equal the configured 40% of saved evaluation requests")
    checks = validate_solution(bank, summary["available_bank_capital"], decisions, candidates, summary)
    paths = output_paths(config, root)
    if len(set(paths.values())) != len(paths) or any(str(path.relative_to(root)) in hashes for path in paths.values()):
        raise ValueError("output collision or original-input overwrite")
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 6B outputs exist; use --overwrite explicitly for Step 6B only")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original inputs changed before decision freeze")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(paths["decisions_csv"], index=False)
    decisions.to_parquet(paths["decisions_parquet"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    pd.DataFrame([summary]).to_csv(paths["portfolio_summary"], index=False)
    _write_json(paths["selection"], {"status": "frozen_before_truth", "selection": summary, "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path), "decision_input_columns": list(BANK_INPUT_COLUMNS), "no_lending_tau": "null denotes reject-all, equivalent to -infinity", "profit_tie_break": config["policy"]["profit_tie_break"], "summation": config["policy"]["summation"], "checks": checks})
    pd.testing.assert_frame_equal(pd.read_parquet(paths["decisions_parquet"]), decisions, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(paths["decisions_csv"], float_precision="round_trip"), decisions, check_exact=True)
    names = ("decisions_csv", "decisions_parquet", "candidates", "portfolio_summary", "selection")
    return {"status": "frozen_before_truth", "artifacts": {key: {"path": str(paths[key].relative_to(root)), "sha256": file_hash(paths[key])} for key in names}}


def verify_frozen_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
    if checkpoint.get("status") != "frozen_before_truth" or set(checkpoint.get("artifacts", {})) != {"decisions_csv", "decisions_parquet", "candidates", "portfolio_summary", "selection"}:
        raise ValueError("bank decisions must be saved and frozen before truth loads")
    for item in checkpoint["artifacts"].values():
        if file_hash(root / item["path"]) != item["sha256"]:
            raise ValueError("frozen bank decision artifact changed")
    saved = json.loads((root / checkpoint["artifacts"]["selection"]["path"]).read_text(encoding="utf-8"))
    if saved["status"] != "frozen_before_truth" or saved["configuration_fingerprint"] != configuration_fingerprint(config):
        raise ValueError("incompatible frozen bank selection")


def load_evaluator_inputs(bank: pd.DataFrame, checkpoint: dict[str, Any], config: dict[str, Any], hashes: dict[str, str], root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First numeric true-profit read, permitted only after decision-file freeze."""
    verify_frozen_checkpoint(checkpoint, config, root)
    if not verify_input_hashes(hashes, root):
        raise ValueError("original evaluator inputs changed")
    paths = [root / config["input_directory"] / "true_expected_profit.parquet", root / config["synthetic_directory"] / "applicants_bank_view.parquet", root / config["synthetic_directory"] / "bank_risk_estimates.parquet"]
    for path in paths:
        if str(path.relative_to(root)) not in hashes:
            raise ValueError("unregistered evaluator input")
    truth, applicants, estimates = [pd.read_parquet(path) for path in paths]
    for name, frame in (("true expected profit", truth), ("observed applicants", applicants), ("saved estimates", estimates)):
        validate_ids(frame, name)
    if tuple(truth.columns) != TRUE_PROFIT_COLUMNS or set(truth.applicant_id) != set(bank.applicant_id):
        raise ValueError("true profit must exactly cover the evaluation applicants")
    if len(applicants) != config["expected_population_rows"] or len(estimates) != len(applicants) or set(estimates.applicant_id) != set(applicants.applicant_id):
        raise ValueError("invalid observed audit population")
    audit = bank[["applicant_id"]].merge(applicants[["applicant_id", "B"]], on="applicant_id", validate="one_to_one").merge(estimates[["applicant_id", "bank_PD_raw"]], on="applicant_id", validate="one_to_one")
    if len(audit) != len(bank) or not np.array_equal(np.clip(audit.bank_PD_raw, 0, 1), bank.bank_PD_bounded):
        raise ValueError("saved raw and bounded probabilities do not align")
    return audit, truth


def _save_figure(candidates: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = candidates.loc[candidates.candidate_type == "cutoff"]
    feasible = finite.loc[finite.budget_feasible]
    infeasible = finite.loc[~finite.budget_feasible]
    fig, ax = plt.subplots(figsize=(8.4, 5.1), layout="constrained")
    if len(feasible):
        ax.axvspan(float(feasible.tau.min()) * 100, float(feasible.tau.max()) * 100, color="#3a927e", alpha=.12)
    ax.plot(feasible.tau * 100, feasible.bank_expected_portfolio_profit / 1e9, color="#217b67", linewidth=2, label="Capital-feasible cutoffs")
    ax.plot(infeasible.tau * 100, infeasible.bank_expected_portfolio_profit / 1e9, color="#b7bdc4", linewidth=1.6, label="Capital-infeasible cutoffs")
    x = 0. if summary["optimal_tau"] is None else 100 * summary["optimal_tau"]
    y = summary["bank_expected_portfolio_profit"] / 1e9
    ax.scatter([x], [y], color="#172e4b", s=55, zorder=5)
    ax.axvline(x, color="#172e4b", linestyle="--", linewidth=1)
    ax.annotate("Optimal cutoff" if summary["optimal_tau"] is not None else "Reject all", xy=(x, y), xytext=(16, 18), textcoords="offset points", fontsize=10)
    ax.scatter([0], [0], marker="D", s=25, facecolors="none", edgecolors="#172e4b", label="Separate reject-all candidate")
    ax.set(xlabel="Bounded default-risk cutoff (%)", ylabel="Bank-expected portfolio profit (USD billions)", title="One common lending cutoff under a 40% capital budget")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_evaluator_results(bank: pd.DataFrame, decisions: pd.DataFrame, candidates: pd.DataFrame, summary: dict[str, Any], audit: pd.DataFrame, truth: pd.DataFrame, checkpoint: dict[str, Any], config: dict[str, Any], hashes: dict[str, str], *, root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    verify_frozen_checkpoint(checkpoint, config, root)
    paths = output_paths(config, root)
    groups, evaluation = evaluate_portfolio(bank, decisions, audit, truth)
    if evaluation["bank_expected_portfolio_profit"] != summary["bank_expected_portfolio_profit"]:
        raise ValueError("evaluator bank objective differs from frozen selection")
    groups.to_csv(paths["group_audit"], index=False)
    pd.DataFrame([evaluation]).to_csv(paths["evaluator_summary"], index=False)
    _save_figure(candidates, summary, paths["figure"])
    verify_frozen_checkpoint(checkpoint, config, root)
    if not verify_input_hashes(hashes, root):
        raise RuntimeError("original research artifacts changed")
    frozen_selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    if file_hash(config_path) != frozen_selection["configuration_file_sha256"]:
        raise RuntimeError("configuration changed after the bank decision")
    report = {"valid": True, "checks": {**frozen_selection["checks"], "decisions_saved_before_truth_read": True, "frozen_bank_artifacts_unchanged": True, "original_inputs_unchanged": True, "CSV_Parquet_exact_agreement": True}, "bank_summary": summary, "evaluator_summary": evaluation, "group_audit": groups.astype(object).where(groups.notna(), None).to_dict("records")}
    _write_json(paths["validation"], report)
    sources = [PROJECT_ROOT / f"src/fair_lending/professor_model/{name}.py" for name in ("lending", "lending_io", "bank_regression", "config", "risk_io", "contracts")]
    sources.append(PROJECT_ROOT / "experiments/run_professor_lending_threshold.py")
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in sources}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"schema_version": config["schema_version"], "status": "Step_6B_validated", "information_scope": "EVALUATOR_ONLY", "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path), "source_code_hashes": source_hashes, "source_code_fingerprint": configuration_fingerprint(source_hashes), "input_checksums": hashes, "evaluation_ID_fingerprint": ids_fingerprint(bank.applicant_id), "freeze_checkpoint": checkpoint, "random_draws": 0, "risk_models_refitted": False, "git_revision": revision, "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout), "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML", "matplotlib")}, "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {key: {"path": str(path.relative_to(root)), "sha256": file_hash(path), "information_role": "BANK_OBSERVABLE" if key in checkpoint["artifacts"] or key == "figure" else "EVALUATOR_ONLY"} for key, path in paths.items() if key != "metadata"}}
    _write_json(paths["metadata"], metadata)
    return report
