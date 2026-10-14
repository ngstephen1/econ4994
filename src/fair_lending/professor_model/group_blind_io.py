"""Saved-input/provenance checks and freeze-before-truth lifecycle for Step 7B."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import group_blind as gb
from .bank_io import load_bank_config, load_bank_inputs
from .bank_regression import attach_splits, ids_fingerprint, validate_ids
from .config import PROJECT_ROOT, configuration_fingerprint
from .contracts import validate_contracts
from .profit import TRUE_PROFIT_COLUMNS
from .risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/group_blind_comparison.yaml"
SYNTHETIC = Path("data/synthetic/professor_model/baseline")
PROCESSED = Path("data/processed/professor_model")
TABLES = Path("results/tables/professor_model")
METRICS = Path("results/metrics/professor_model")
FIGURES = Path("results/figures/professor_model")
FROZEN = ("model", "coefficients", "predictions_csv", "predictions_parquet", "profit_csv", "profit_parquet", "decisions_csv", "decisions_parquet", "candidates", "selection")


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    expected = {
        "family": "statsmodels_OLS", "solver": "pinv", "predictors": list(gb.FEATURES),
        "intercept": True, "target": "PN", "regularization": None,
        "target_transform": "none", "prediction_rescaling": "none", "bounding": "clip_0_1_separate_column",
    }
    if config.get("schema_version") != "professor-group-blind-comparison-v1" or config.get("model_id") != gb.MODEL_ID or config.get("model") != expected:
        raise ValueError("Step 7B requires the exact raw-PN group-blind OLS")
    for key in ("expected_population_rows", "training_rows", "evaluation_rows"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("positive cohort counts required")
    if config["training_rows"] + config["evaluation_rows"] != config["expected_population_rows"]:
        raise ValueError("cohorts must cover saved population")
    if config["split_source"] != str(SYNTHETIC / "applicant_splits.csv") or config["budget_source"] != str(METRICS / "lending_selection.json") or config["budget_field"] != "selection.available_bank_capital":
        raise ValueError("saved split and original absolute budget are authoritative")
    return config


def paths(root=PROJECT_ROOT):
    relative = {
        "model": METRICS / "group_blind_regression_model.json",
        "coefficients": TABLES / "group_blind_regression_coefficients.csv",
        "predictions_csv": SYNTHETIC / "bank_risk_estimates_group_blind.csv",
        "predictions_parquet": SYNTHETIC / "bank_risk_estimates_group_blind.parquet",
        "profit_csv": PROCESSED / "bank_expected_profit_group_blind.csv",
        "profit_parquet": PROCESSED / "bank_expected_profit_group_blind.parquet",
        "decisions_csv": PROCESSED / "lending_decisions_group_blind.csv",
        "decisions_parquet": PROCESSED / "lending_decisions_group_blind.parquet",
        "candidates": TABLES / "group_blind_threshold_candidates.csv",
        "selection": METRICS / "group_blind_lending_selection.json",
        "freeze": METRICS / "group_blind_freeze_checkpoint.json",
        "risk": TABLES / "group_aware_vs_blind_risk.csv",
        "risk_groups": TABLES / "group_aware_vs_blind_risk_group_audit.csv",
        "prediction_changes": TABLES / "group_aware_vs_blind_prediction_changes.csv",
        "portfolio": TABLES / "group_aware_vs_blind_portfolio.csv",
        "groups": TABLES / "group_aware_vs_blind_group_audit.csv",
        "overlap": TABLES / "group_aware_vs_blind_allocation_overlap.csv",
        "validation": METRICS / "group_blind_comparison_validation.json",
        "metadata": PROCESSED / "group_blind_comparison_metadata.json",
        "group_figure": FIGURES / "group_aware_vs_blind_group_approval.png",
        "profit_figure": FIGURES / "group_aware_vs_blind_true_profit.png",
    }
    return {key: root / value for key, value in relative.items()}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _registered_originals(root):
    """Hash prior evidence, including private bytes, without numerical truth reads."""
    metadata_path = PROCESSED / "oracle_comparison_metadata.json"
    metadata = json.loads((root / metadata_path).read_text())
    if metadata["status"] != "Step_7A_validated" or configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_fingerprint"] or configuration_fingerprint(metadata["source_code_hashes"]) != metadata["source_code_fingerprint"]:
        raise ValueError("original oracle provenance is invalid")
    hashes = {**metadata["input_checksums"], **metadata["source_code_hashes"]}
    for item in metadata["artifacts"].values():
        if item["path"] in hashes and hashes[item["path"]] != item["sha256"]:
            raise ValueError("inconsistent original artifact identities")
        hashes[item["path"]] = item["sha256"]
    hashes["configs/professor_model/oracle_comparison.yaml"] = metadata["configuration_file_sha256"]
    hashes[str(metadata_path)] = file_hash(root / metadata_path)
    allowed = ("data/synthetic/professor_model", "data/processed/professor_model", "results/tables/professor_model", "results/metrics/professor_model", "results/figures/professor_model", "configs/professor_model", "src/fair_lending/professor_model", "experiments")
    for relative in hashes:
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or not any(p.is_relative_to(prefix) for prefix in allowed):
            raise ValueError("unexpected original evidence path")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original bank/oracle/source checksum mismatch")
    return hashes


def load_inputs(config, root=PROJECT_ROOT):
    """No truth/oracle Parquet reads: bank observables, PN, split and contracts."""
    hashes = _registered_originals(root)
    observed, bank_hashes, schemas = load_bank_inputs(load_bank_config(root / "configs/professor_model/bank_regression.yaml"), root)
    if any(hashes.get(key) != value for key, value in bank_hashes.items()):
        raise ValueError("inconsistent prior bank/oracle manifests")

    def read(relative):
        if str(relative) not in hashes:
            raise ValueError("unregistered bank input")
        p = root / relative
        return pd.read_csv(p, float_precision="round_trip") if p.suffix == ".csv" else pd.read_parquet(p)

    splits = read(Path(config["split_source"]))
    observed = attach_splits(observed, splits)
    counts = observed.split.value_counts().to_dict()
    if len(observed) != config["expected_population_rows"] or counts != {"train": config["training_rows"], "evaluation": config["evaluation_rows"]}:
        raise ValueError("saved cohorts differ from declared counts")
    aware = read(SYNTHETIC / "bank_risk_estimates.parquet")
    contracts = read(PROCESSED / "loan_contracts.parquet")
    bank_profit = read(PROCESSED / "bank_expected_profit.parquet")
    bank_decisions = read(PROCESSED / "lending_decisions.parquet")
    validate_contracts(contracts)
    if not contracts["T"].eq(30).all() or not contracts.theta.eq(.1).all() or not contracts.r.eq(.05).all():
        raise ValueError("saved contract terms changed")
    eval_ids = set(observed.loc[observed.split == "evaluation", "applicant_id"])
    for frame in (contracts, bank_profit, bank_decisions):
        validate_ids(frame, "evaluation artifacts")
        if set(frame.applicant_id) != eval_ids:
            raise ValueError("saved evaluation IDs differ")
    validate_ids(aware, "original predictions")
    expected_labels = observed[["applicant_id", "split"]].sort_values("applicant_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(aware[["applicant_id", "split"]].sort_values("applicant_id").reset_index(drop=True), expected_labels)
    model = json.loads((root / METRICS / "bank_regression_model.json").read_text())
    train = observed.loc[observed.split == "train"].sort_values("applicant_id")
    if model["training_ID_fingerprint"] != ids_fingerprint(train.applicant_id) or model["training_count"] != len(train) or model["training_mean_PN"] != float(train.PN.mean()):
        raise ValueError("training IDs/target differ from original bank")
    selection = json.loads((root / config["budget_source"]).read_text())["selection"]
    if selection["available_bank_capital"] != json.loads((root / PROCESSED / "oracle_comparison_metadata.json").read_text())["exact_saved_budget"]:
        raise ValueError("saved bank/oracle budgets differ")
    return {"observed": observed, "aware": aware, "contracts": contracts, "bank_profit": bank_profit, "bank_decisions": bank_decisions, "bank_selection": selection, "original_model": model, "schemas": schemas}, hashes


def source_hashes():
    names = ("group_blind", "group_blind_io", "bank_regression", "bank_io", "lending", "contracts", "profit", "config", "risk_io")
    files = [f"src/fair_lending/professor_model/{name}.py" for name in names]
    files.append("experiments/run_professor_group_blind_comparison.py")
    return {name: file_hash(PROJECT_ROOT / name) for name in files}


def freeze(inputs, model, predictions, profit, decisions, candidates, selection, config, hashes, *, root=PROJECT_ROOT, config_path=CONFIG_PATH, overwrite=False):
    """Freeze all bank outputs and decision identity before numeric truth loading."""
    output = paths(root)
    if not overwrite and any(path.exists() for path in output.values()):
        raise FileExistsError("Step 7B outputs exist; --overwrite replaces Step 7B only")
    if any(str(path.relative_to(root)) in hashes for path in output.values()) or not verify_input_hashes(hashes, root):
        raise ValueError("original evidence changed or output collides")
    pd.testing.assert_frame_equal(predictions, gb.risk_estimates(inputs["observed"], model), check_exact=True)
    if model["training_ID_fingerprint"] != inputs["original_model"]["training_ID_fingerprint"]:
        raise ValueError("training ID mismatch")
    evaluation = predictions.loc[predictions.split == "evaluation", ["applicant_id", "model_id", gb.BOUNDED]]
    pd.testing.assert_frame_equal(profit, gb.assess_profit(inputs["contracts"], evaluation), check_exact=True)
    rebuilt = gb.optimize_cutoff(inputs["contracts"], profit, inputs["bank_selection"]["available_bank_capital"])
    for actual, expected in zip((decisions, candidates), rebuilt[:2]):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if selection != rebuilt[2]:
        raise ValueError("new bank selection does not match exact search")
    for path in output.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    model = {**model, "frozen_before_evaluator_truth": True, "configuration_fingerprint": configuration_fingerprint(config)}
    _json(output["model"], model)
    pd.DataFrame([{"term": term, "coefficient": model["coefficients"][term], "predictor_units": model["units"][term]} for term in gb.DESIGN]).to_csv(output["coefficients"], index=False)
    for name, frame in (("predictions", predictions), ("profit", profit), ("decisions", decisions)):
        frame.to_csv(output[f"{name}_csv"], index=False)
        frame.to_parquet(output[f"{name}_parquet"], index=False)
        pd.testing.assert_frame_equal(pd.read_csv(output[f"{name}_csv"], float_precision="round_trip"), frame, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_parquet(output[f"{name}_parquet"]), frame, check_exact=True)
    candidates.to_csv(output["candidates"], index=False)
    _json(output["selection"], {"status": "frozen_before_truth", "selection": selection, "decision_input_columns": ["applicant_id", "model_id", gb.BOUNDED, "requested_principal", "expected_profit_bank"]})
    checkpoint = {"status": "frozen_before_truth", "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path), "source_code_hashes": source_hashes(), "artifacts": {key: {"path": str(output[key].relative_to(root)), "sha256": file_hash(output[key])} for key in FROZEN}}
    _json(output["freeze"], checkpoint)
    return checkpoint


def verify_freeze(config, hashes, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    output = paths(root)
    checkpoint = json.loads(output["freeze"].read_text())
    if checkpoint["status"] != "frozen_before_truth" or set(checkpoint["artifacts"]) != set(FROZEN) or checkpoint["configuration_fingerprint"] != configuration_fingerprint(config) or checkpoint["configuration_file_sha256"] != file_hash(config_path) or checkpoint["source_code_hashes"] != source_hashes():
        raise ValueError("incompatible group-blind freeze/configuration/source")
    for key, item in checkpoint["artifacts"].items():
        if item["path"] != str(output[key].relative_to(root)) or item["sha256"] != file_hash(output[key]):
            raise ValueError("frozen group-blind artifact changed")
    if not verify_input_hashes(hashes, root):
        raise ValueError("original bank/oracle evidence changed")
    return checkpoint


def load_truth(config, hashes, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    """First numeric truth/oracle read in the Step 7B workflow, after freeze."""
    verify_freeze(config, hashes, root=root, config_path=config_path)
    truth = pd.read_parquet(root / PROCESSED / "true_expected_profit.parquet")
    oracle = pd.read_parquet(root / PROCESSED / "oracle_lending_decisions.parquet")
    if tuple(truth.columns) != TRUE_PROFIT_COLUMNS or not np.isfinite(truth.iloc[:, 1:].to_numpy()).all() or not truth.true_PD.between(0, 1).all():
        raise ValueError("invalid evaluator truth schema/values")
    if not oracle.information_scope.eq("EVALUATOR_ONLY").all() or not oracle.policy_id.eq("professor_true_risk_oracle_v1").all():
        raise ValueError("invalid saved oracle policy")
    for frame in (truth, oracle):
        validate_ids(frame, "evaluator inputs")
        if len(frame) != config["evaluation_rows"]:
            raise ValueError("invalid evaluator cohort size")
    oracle_selection = json.loads((root / METRICS / "oracle_comparison_validation.json").read_text())["oracle_selection"]
    return truth, oracle, oracle_selection


def _figures(summary, groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Group-aware", "Group-blind", "True-risk oracle"]
    policies = ["group_aware", "group_blind", "oracle"]
    colors = ["#243e59", "#288679", "#9a724d"]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = np.arange(2)
    for i, policy in enumerate(policies):
        values = 100 * groups.loc[groups.policy == policy].set_index("B").loc[[0, 1], "approval_rate"].to_numpy()
        bars = ax.bar(x + (i - 1) * .25, values, width=.23, color=colors[i], label=labels[i])
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set(xticks=x, xticklabels=["B=0", "B=1"], ylabel="Approved applicants (%)", title="Group approval rates: one matched synthetic population", ylim=(0, max(groups.approval_rate.max() * 130, 1)))
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output["group_figure"], dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    values = summary.set_index("policy").loc[policies, "true_expected_portfolio_profit"].to_numpy() / 1e9
    bars = ax.bar(labels, values, color=colors, width=.6)
    ax.bar_label(bars, labels=[f"${v:.6f}B" for v in values], padding=5)
    ax.set(ylabel="True expected portfolio profit (USD billions)", title="Identical contracts and saved capital budget")
    if np.all(values >= 0):
        ax.set_ylim(0, max(values.max() * 1.18, .01))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output["profit_figure"], dpi=180)
    plt.close(fig)


def save_evaluation(inputs, predictions, profit, decisions, selection, config, hashes, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    truth, oracle, oracle_selection = load_truth(config, hashes, root=root, config_path=config_path)
    risk, risk_groups, changes = gb.compare_risk(inputs["observed"], inputs["aware"], predictions, truth)
    policy_data = {
        "group_aware": (inputs["bank_decisions"], inputs["bank_selection"], "bank_PD_bounded", inputs["bank_profit"][["applicant_id", "expected_profit_bank"]]),
        "group_blind": (decisions, selection, gb.BOUNDED, profit[["applicant_id", "expected_profit_bank"]]),
        "oracle": (oracle, oracle_selection, "true_PD", None),
    }
    audit = inputs["observed"].loc[inputs["observed"].split == "evaluation", ["applicant_id", "B"]]
    summary, groups, overlap, comparison = gb.compare_allocations(inputs["contracts"], truth, audit, policy_data, inputs["bank_selection"]["available_bank_capital"])
    output = paths(root)
    for key, frame in (("risk", risk), ("risk_groups", risk_groups), ("prediction_changes", changes), ("portfolio", summary), ("groups", groups), ("overlap", overlap)):
        frame.to_csv(output[key], index=False)
    _figures(summary, groups, output)
    checkpoint = verify_freeze(config, hashes, root=root, config_path=config_path)
    report = {"valid": True, "information_scope": "EVALUATOR_ONLY", "comparison": comparison, "group_blind_selection": selection, "checks": {"same_saved_train_evaluation_split": True, "same_raw_PN_target": True, "exact_Y_D_C_feature_allowlist": True, "new_decisions_frozen_before_truth": True, "same_exact_saved_budget": True, "old_bank_and_oracle_not_reoptimized": True, "original_inputs_unchanged": True, "CSV_Parquet_exact_agreement": True}, "notices": ["Clipping is not calibration.", "B removal re-estimates all slopes and changes the cutoff; it does not isolate the B coefficient.", "Score-dependent cutoff families differ; signed oracle comparisons are not unrestricted optimal regret.", "One noisy synthetic realization does not identify discrimination or establish equal treatment."]}
    _json(output["validation"], report)
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    metadata = {"schema_version": config["schema_version"], "status": "Step_7B_validated", "information_scope": "EVALUATOR_ONLY", "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path), "source_code_hashes": checkpoint["source_code_hashes"], "source_code_fingerprint": configuration_fingerprint(checkpoint["source_code_hashes"]), "input_checksums": hashes, "freeze_checkpoint": checkpoint, "training_ID_fingerprint": ids_fingerprint(inputs["observed"].loc[inputs["observed"].split == "train", "applicant_id"]), "evaluation_ID_fingerprint": ids_fingerprint(audit.applicant_id), "exact_saved_budget": inputs["bank_selection"]["available_bank_capital"], "new_model_fits": 1, "original_models_refitted": False, "original_policies_reoptimized": False, "random_draws": 0, "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")), "python": platform.python_version(), "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow", "PyYAML", "matplotlib")}, "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {key: {"path": str(path.relative_to(root)), "sha256": file_hash(path), "information_role": "BANK_OBSERVABLE" if key in FROZEN or key == "freeze" else "EVALUATOR_ONLY"} for key, path in output.items() if key != "metadata"}}
    _json(output["metadata"], metadata)
    return report
