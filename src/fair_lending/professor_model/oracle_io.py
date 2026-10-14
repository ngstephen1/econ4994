"""Separate evaluator-only persistence for the Step 7A oracle benchmark."""

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

from fair_lending.professor_model.bank_regression import ids_fingerprint, validate_ids
from fair_lending.professor_model.config import PROJECT_ROOT, configuration_fingerprint
from fair_lending.professor_model.contracts import validate_contracts
from fair_lending.professor_model.lending import DECISION_COLUMNS
from fair_lending.professor_model.lending_io import (
    load_evaluator_inputs, load_lending_config, load_lending_inputs, verify_frozen_checkpoint,
)
from fair_lending.professor_model.oracle import ORACLE_ID, SCOPE, compare_policies, optimize_oracle
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/oracle_comparison.yaml"


def load_oracle_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        if config["schema_version"] != "professor-oracle-comparison-v1" or config["policy_id"] != ORACLE_ID or config["information_scope"] != SCOPE or config["oracle_score"] != "true_PD" or config["oracle_objective"] != "expected_profit_true" or config["tie_policy"] != "lowest_cutoff_reject_all_first":
            raise ValueError("unsupported oracle information/decision contract")
        if type(config["evaluation_rows"]) is not int or config["evaluation_rows"] < 1 or config["budget_field"] != "selection.available_bank_capital":
            raise ValueError("oracle must read the saved absolute bank budget")
        for relative, prefix in (
            (config["input_directory"], "data/processed/professor_model"),
            (config["budget_source"], "results/metrics/professor_model"),
            *[(config["outputs"][f"{name}_directory"], f"{base}/professor_model") for name, base in (("data", "data/processed"), ("tables", "results/tables"), ("metrics", "results/metrics"), ("figures", "results/figures"))],
        ):
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or not p.is_relative_to(prefix):
                raise ValueError("oracle paths must remain in professor-model namespaces")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed oracle configuration: {exc}") from exc
    return config


def load_oracle_inputs(config: dict[str, Any], root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate saved Step 6B evidence; do not choose a bank cutoff again."""
    bank_config = load_lending_config(root / "configs/professor_model/lending_threshold.yaml")
    bank, hashes, schemas = load_lending_inputs(bank_config, root)
    directory = root / config["input_directory"]
    metadata_path = directory / "lending_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    def register(relative: str, expected: str | None = None) -> None:
        p = Path(relative)
        prefixes = ("data/processed/professor_model", "results/metrics/professor_model", "results/tables/professor_model", "results/figures/professor_model", "configs/professor_model", "src/fair_lending/professor_model", "experiments")
        if p.is_absolute() or ".." in p.parts or not any(p.is_relative_to(prefix) for prefix in prefixes):
            raise ValueError("unexpected prior oracle-input path")
        digest = file_hash(root / p)
        if expected is not None and digest != expected:
            raise ValueError(f"original Step 6B checksum mismatch: {relative}")
        if relative in hashes and hashes[relative] != digest:
            raise ValueError("inconsistent prior checksums")
        hashes[relative] = digest
    register(str(metadata_path.relative_to(root)))
    if metadata["status"] != "Step_6B_validated" or metadata["configuration_fingerprint"] != configuration_fingerprint(bank_config) or metadata["source_code_fingerprint"] != configuration_fingerprint(metadata["source_code_hashes"]):
        raise ValueError("Step 6B is not validated or its provenance differs")
    for field in ("source_code_hashes",):
        for relative, expected in metadata[field].items():
            register(relative, expected)
    # Prior input hashes have already been verified by the Step 6B loader.
    for relative, expected in metadata["input_checksums"].items():
        if hashes.get(relative) != expected:
            raise ValueError("bank metadata references different original inputs")
    register("configs/professor_model/lending_threshold.yaml", metadata["configuration_file_sha256"])
    for item in metadata["artifacts"].values():
        register(item["path"], item["sha256"])
    checkpoint = metadata["freeze_checkpoint"]
    verify_frozen_checkpoint(checkpoint, bank_config, root)
    selection_path = root / config["budget_source"]
    if str(selection_path.relative_to(root)) != checkpoint["artifacts"]["selection"]["path"]:
        raise ValueError("budget source must be the frozen bank selection")
    saved = json.loads(selection_path.read_text(encoding="utf-8"))
    selection = saved["selection"]
    decisions_path = directory / "lending_decisions.parquet"
    if str(decisions_path.relative_to(root)) != checkpoint["artifacts"]["decisions_parquet"]["path"]:
        raise ValueError("oracle must compare the original frozen bank decisions")
    decisions = pd.read_parquet(decisions_path)
    validate_ids(decisions, "saved bank decisions")
    if tuple(decisions.columns) != DECISION_COLUMNS or len(decisions) != config["evaluation_rows"] or set(decisions.applicant_id) != set(bank.applicant_id):
        raise ValueError("saved bank decision schema/cohort mismatch")
    # Reuse the already-frozen evaluator lookup, not a fit/search function.
    audit, truth = load_evaluator_inputs(bank, checkpoint, bank_config, hashes, root)
    contracts = pd.read_parquet(directory / "loan_contracts.parquet")
    validate_contracts(contracts)
    if len(contracts) != config["evaluation_rows"] or set(contracts.applicant_id) != set(bank.applicant_id):
        raise ValueError("evaluation contracts do not match bank allocation")
    if not contracts["T"].eq(30).all() or not contracts.theta.eq(.1).all() or not contracts.r.eq(.05).all():
        raise ValueError("original professor contract terms must remain T=30, theta=.10, r=.05")
    aligned = bank.merge(decisions, on="applicant_id", validate="one_to_one", suffixes=("_input", "_decision"))
    for column in ("model_id", "bank_PD_bounded", "requested_principal"):
        np.testing.assert_array_equal(aligned[f"{column}_input"], aligned[f"{column}_decision"])
    tau = selection["optimal_tau"]
    expected = np.zeros(len(aligned), dtype=bool) if tau is None else aligned.bank_PD_bounded_input <= tau
    np.testing.assert_array_equal(aligned.approved, expected)
    if not np.isfinite(selection["available_bank_capital"]) or selection["available_bank_capital"] < 0:
        raise ValueError("invalid saved capital budget")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original artifacts changed during oracle loading")
    return {"bank": bank, "contracts": contracts, "truth": truth, "audit": audit, "bank_decisions": decisions, "bank_selection": selection, "schemas": schemas}, hashes


def output_paths(config: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Path]:
    data, tables, metrics, figures = (root / config["outputs"][f"{key}_directory"] for key in ("data", "tables", "metrics", "figures"))
    return {"oracle_csv": data / "oracle_lending_decisions.csv", "oracle_parquet": data / "oracle_lending_decisions.parquet", "summary": tables / "bank_vs_oracle_summary.csv", "groups": tables / "bank_vs_oracle_group_audit.csv", "disagreements": tables / "bank_vs_oracle_disagreements.csv", "candidates": tables / "oracle_threshold_candidates.csv", "validation": metrics / "oracle_comparison_validation.json", "metadata": data / "oracle_comparison_metadata.json", "profit_figure": figures / "bank_vs_oracle_profit.png", "group_figure": figures / "bank_vs_oracle_group_approval.png"}


def _save_figures(summary: pd.DataFrame, groups: pd.DataFrame, paths: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#244962", "#278575")
    indexed = summary.set_index("policy").loc[["bank", "oracle"]]
    fig, ax = plt.subplots(figsize=(6.7, 4.7), layout="constrained")
    values = indexed.true_expected_portfolio_profit.to_numpy() / 1e9
    bars = ax.bar(["Bank (frozen)", "True-risk oracle"], values, color=colors, width=.58)
    ax.bar_label(bars, labels=[f"${value:,.4f}B" for value in values], padding=5, fontsize=11)
    ax.set(ylabel="True expected portfolio profit (USD billions)", title="Same applicants, contracts, and capital budget")
    if np.all(values >= 0):
        ax.set_ylim(0, max(values.max() * 1.16, .01))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(paths["profit_figure"], dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.7), layout="constrained")
    x = np.arange(2)
    maximum = 0.
    for index, policy in enumerate(("bank", "oracle")):
        values = groups.loc[groups.policy == policy].set_index("B").reindex([0, 1]).approval_rate.to_numpy() * 100
        bars = ax.bar(x + (index - .5) * .34, values, width=.32, color=colors[index], label="Bank (frozen)" if policy == "bank" else "True-risk oracle")
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
        maximum = max(maximum, float(np.nanmax(values)))
    ax.set(xticks=x, xticklabels=["B=0", "B=1"], ylabel="Approved applicants (%)", title="Group approval rates under matched policies", ylim=(0, max(maximum * 1.27, 1)))
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(paths["group_figure"], dpi=180)
    plt.close(fig)


def save_oracle_results(inputs: dict[str, Any], decisions: pd.DataFrame, candidates: pd.DataFrame, selection: dict[str, Any], config: dict[str, Any], hashes: dict[str, str], *, root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH, overwrite: bool = False) -> dict[str, Any]:
    """Persist only new evaluator artifacts; old bank outputs are never written."""
    paths = output_paths(config, root)
    if len(set(paths.values())) != len(paths) or any(str(p.relative_to(root)) in hashes for p in paths.values()):
        raise ValueError("output collision with original evidence")
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Step 7A outputs exist; --overwrite applies to Step 7A only")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original artifacts changed before export")
    repeated, repeated_candidates, repeated_selection = optimize_oracle(inputs["contracts"], inputs["truth"], inputs["bank_selection"]["available_bank_capital"])
    pd.testing.assert_frame_equal(decisions, repeated, check_exact=True)
    pd.testing.assert_frame_equal(candidates, repeated_candidates, check_exact=True)
    if selection != repeated_selection:
        raise ValueError("oracle selection is not the exact true-risk threshold optimum")
    summary, groups, disagreements, comparison = compare_policies(inputs["contracts"], inputs["truth"], inputs["audit"], inputs["bank_decisions"], inputs["bank_selection"], decisions, selection)
    oracle_profit = summary.loc[summary.policy == "oracle", "true_expected_portfolio_profit"].iloc[0]
    if oracle_profit != selection["true_expected_portfolio_profit"]:
        raise ValueError("oracle true-profit objective does not match its selected loans")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(paths["oracle_csv"], index=False)
    decisions.to_parquet(paths["oracle_parquet"], index=False)
    pd.testing.assert_frame_equal(pd.read_csv(paths["oracle_csv"], float_precision="round_trip"), decisions, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["oracle_parquet"]), decisions, check_exact=True)
    for key, frame in (("summary", summary), ("groups", groups), ("disagreements", disagreements), ("candidates", candidates)):
        frame.to_csv(paths[key], index=False)
    _save_figures(summary, groups, paths)
    if not verify_input_hashes(hashes, root):
        raise RuntimeError("original scientific artifacts changed during oracle evaluation")
    report = {
        "valid": True, "information_scope": SCOPE, "oracle_selection": selection,
        "comparison": comparison,
        "checks": {"bank_decisions_unchanged": True, "bank_not_reoptimized": True, "same_evaluation_applicants_and_contracts": True, "identical_saved_absolute_budget": True, "both_policies_obey_own_common_cutoff": True, "budgets_feasible": True, "exact_ties_reject_all_and_objective": True, "CSV_Parquet_exact_agreement": True, "original_inputs_unchanged": True},
        "notices": [comparison["limitation"], "Oracle knows probabilities, not future realized outcomes.", "Oracle is not automatically fair; group-gap difference is descriptive, not an identified causal effect."],
    }
    paths["validation"].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    names = ("oracle", "oracle_io", "lending", "lending_io", "bank_regression", "config", "contracts", "risk_io")
    sources = [PROJECT_ROOT / f"src/fair_lending/professor_model/{name}.py" for name in names]
    sources.append(PROJECT_ROOT / "experiments/run_professor_oracle_comparison.py")
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in sources}
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    metadata = {"schema_version": config["schema_version"], "status": "Step_7A_validated", "information_scope": SCOPE, "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path), "source_code_hashes": source_hashes, "source_code_fingerprint": configuration_fingerprint(source_hashes), "input_checksums": hashes, "evaluation_ID_fingerprint": ids_fingerprint(inputs["contracts"].applicant_id), "exact_saved_budget": inputs["bank_selection"]["available_bank_capital"], "frozen_bank_tau": inputs["bank_selection"]["optimal_tau"], "bank_reoptimized": False, "models_refitted": False, "random_draws": 0, "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")), "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML", "matplotlib")}, "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {key: {"path": str(p.relative_to(root)), "sha256": file_hash(p), "information_role": SCOPE} for key, p in paths.items() if key != "metadata"}}
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report
