"""Step 9B frozen-risk inputs, bank freeze, then evaluator-only release."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import yaml

from . import revised_lending as rl
from . import revised_scenarios as rs
from . import revised_scenarios_io as risk_io
from .bank_regression import ids_fingerprint
from .config import PROJECT_ROOT, configuration_fingerprint
from .risk_io import file_hash

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/revised_lending.yaml"
DATA = Path("data/processed/professor_model/revised_lending")
TABLES = Path("results/tables/professor_model/revised_lending")
METRICS = Path("results/metrics/professor_model/revised_lending")
FIGURES = Path("results/figures/professor_model/revised_lending")
BANK_OUTPUTS = ("decisions_csv", "decisions_parquet", "assessments_csv", "assessments_parquet", "candidates", "selections")


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    expected = {
        "schema_version": rl.SCHEMA_VERSION, "evaluation_rows": config.get("evaluation_rows"),
        "scenario_metadata": "data/processed/professor_model/revised_scenarios/scenario_risk_metadata.json",
        "budget_source": "results/metrics/professor_model/lending_selection.json",
        "expected_capital_USD": rs.EXACT_BUDGET, "scenarios": list(rl.SCENARIO_IDS),
        "policy": {"score": "estimated_PD_bounded", "comparison": "less_than_or_equal",
                   "tied_scores": "indivisible_exact_blocks", "objective": "existing_bank_expected_profit",
                   "tie_break": "lowest_cutoff_reject_all_first", "summation": "existing_exact_binary_float_prefixes",
                   "independent_verification": "existing_Fraction_enumeration"},
    }
    if config != expected or type(config["evaluation_rows"]) is not int or config["evaluation_rows"] < 1:
        raise ValueError("Step 9B requires frozen scenarios, original capital and unchanged cutoff policy")
    return config


def paths(root=PROJECT_ROOT):
    return {k: root / v for k, v in {
        "decisions_csv": DATA / "scenario_lending_decisions.csv",
        "decisions_parquet": DATA / "scenario_lending_decisions.parquet",
        "assessments_csv": DATA / "scenario_bank_assessments.csv",
        "assessments_parquet": DATA / "scenario_bank_assessments.parquet",
        "candidates": TABLES / "cutoff_candidates.csv", "selections": METRICS / "bank_selections.json",
        "freeze": METRICS / "bank_freeze_checkpoint.json", "preservation": METRICS / "preservation_manifest.json",
        "portfolio": TABLES / "portfolio_comparison.csv", "groups": TABLES / "group_outcome_audit.csv",
        "overlap": TABLES / "pairwise_allocation_overlap.csv", "ties": TABLES / "zero_score_tie_diagnostics.csv",
        "validation": METRICS / "validation.json", "metadata": METRICS / "run_manifest.json",
        "profit_figure": FIGURES / "true_expected_portfolio_profit.png",
        "approval_figure": FIGURES / "group_approval_rates.png",
    }.items()}


def require_new_outputs(root=PROJECT_ROOT):
    existing = [str(p) for k, p in paths(root).items() if k != "preservation" and p.exists()]
    if existing:
        raise FileExistsError(f"Step 9B never overwrites prior outputs: {existing}")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_hashes(hashes, root=PROJECT_ROOT):
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).resolve().is_relative_to(root.resolve()):
            raise ValueError("provenance path escapes repository")
        if not (root / relative).is_file() or file_hash(root / relative) != expected:
            raise ValueError(f"preservation/provenance checksum mismatch: {name}")


def preservation_manifest(root=PROJECT_ROOT):
    path = paths(root)["preservation"]
    if not path.exists():
        # A new reproduction records the files that exist before its first write.
        files = set()
        for folder in ("src", "configs", "experiments", "tests", "docs",
                       "data/synthetic/professor_model", "data/processed/professor_model",
                       "results/tables/professor_model", "results/metrics/professor_model", "results/figures/professor_model"):
            files.update(p for p in (root / folder).rglob("*") if p.is_file()
                         and p.name != ".DS_Store" and "__pycache__" not in p.parts
                         and p != root / "docs/PROJECT_STATUS.md"
                         and not any(p.is_relative_to(root / d) for d in (DATA, TABLES, METRICS, FIGURES)))
        write_json(path, {"schema_version": "professor-revised-lending-preservation-v1",
                          "checksums": {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}})
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != "professor-revised-lending-preservation-v1" or not manifest["checksums"]:
        raise ValueError("invalid preservation manifest")
    verify_hashes(manifest["checksums"], root)
    return manifest["checksums"]


def source_hashes():
    modules = ("revised_lending", "revised_lending_io", "revised_lending_figures", "lending",
               "profit", "contracts", "noise_pilot", "revised_scenarios", "revised_scenarios_io",
               "config", "bank_regression", "group_blind", "risk_io")
    files = [f"src/fair_lending/professor_model/{m}.py" for m in modules]
    files.append("experiments/run_professor_revised_lending.py")
    return {p: file_hash(PROJECT_ROOT / p) for p in files}


def load_risk_inputs(config, originals, root=PROJECT_ROOT):
    """Read frozen scores/contracts only; do not parse truth or reference masks."""
    metadata_path = root / config["scenario_metadata"]
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "Step_9A_validated_risk_only" or metadata["schema_version"] != rs.SCHEMA_VERSION or configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_fingerprint"] or configuration_fingerprint(metadata["source_code_hashes"]) != metadata["source_code_fingerprint"]:
        raise ValueError("invalid frozen Step 9A provenance")
    prior_config_path = root / "configs/professor_model/revised_scenarios.yaml"
    prior_config = risk_io.load_config(prior_config_path)
    if prior_config != metadata["resolved_configuration"] or file_hash(prior_config_path) != metadata["configuration_file_sha256"]:
        raise ValueError("frozen Step 9A configuration changed")
    hashes = {**originals, str(metadata_path.relative_to(root)): file_hash(metadata_path)}
    for field in ("original_checksums", "input_checksums", "source_code_hashes"):
        verify_hashes(metadata[field], root)
        hashes.update(metadata[field])
    for item in metadata["artifacts"].values():
        verify_hashes({item["path"]: item["sha256"]}, root)
        hashes[item["path"]] = item["sha256"]
    for name in ("risk_csv", "risk_parquet"):
        item = metadata["artifacts"].get(name)
        if item is None or item["path"] != str(risk_io.output_paths(root)[name].relative_to(root)):
            raise ValueError("frozen Step 9A primary risk artifact is missing or misidentified")
    # Reuse the tested old loader to verify fits/IDs; it never reads PN/truth.
    old, registered = risk_io.load_frozen_inputs(prior_config, root)
    hashes.update(registered)
    risk_path = root / "data/processed/professor_model/revised_scenarios/scenario_risk_estimates.parquet"
    risks = pd.read_parquet(risk_path)
    pd.testing.assert_frame_equal(risks, pd.read_csv(risk_path.with_suffix(".csv"), float_precision="round_trip"), check_exact=True)
    checks = rs.validate_scenario_estimates(risks, old["applicants"], old["aware"], old["blind"])
    if len(old["contracts"]) != config["evaluation_rows"] or old["budget"] != config["expected_capital_USD"]:
        raise ValueError("evaluation size or original absolute budget changed")
    verify_hashes(hashes, root)
    return {"risks": risks, "contracts": old["contracts"], "budget": old["budget"],
            "audit": old["applicants"][["applicant_id", "B"]].copy(), "hashes": hashes,
            "evaluation_ID_fingerprint": ids_fingerprint(old["contracts"].applicant_id),
            "risk_checks": checks}, hashes


def write_csv(path, frame):
    # Deliberate missing statistics (e.g. approved-risk mean for reject-all)
    # export as empty cells without NumPy's NaN-to-string cast warning.
    frame.astype(object).where(frame.notna(), None).to_csv(path, index=False, mode="x")
    # CSV lacks integer-width metadata and represents all-null numbers as NaN.
    expected = frame.copy()
    for column in expected:
        if expected[column].isna().all(): expected[column] = np.nan
    pd.testing.assert_frame_equal(pd.read_csv(path, float_precision="round_trip"), expected,
                                  check_exact=True, check_dtype=False)


def freeze_banks(inputs, banks, config, hashes, originals, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    require_new_outputs(root)
    verify_hashes(hashes, root)
    verify_hashes(originals, root)
    if set(banks) != set(rl.SCENARIO_IDS):
        raise ValueError("all three scenario policies must freeze together")
    for scenario, bank in banks.items():
        original = inputs["risks"].loc[inputs["risks"].scenario_id == scenario].sort_values("applicant_id")
        decisions = bank["decisions"].sort_values("applicant_id")
        np.testing.assert_array_equal(original.applicant_id, decisions.applicant_id)
        np.testing.assert_array_equal(original.estimated_PD_bounded, decisions.estimated_PD_bounded)
        if tuple(decisions.columns) != rl.DECISION_COLUMNS or not decisions.information_scope.eq(rl.BANK_SCOPE).all() or not decisions.scenario_id.eq(scenario).all() or not bank["checks"]["independent_exact_enumeration_passed"]:
            raise ValueError("invalid bank-only decision freeze")
        # Independently recheck the actual in-memory bundle, not just a stale
        # successful flag returned before a caller might have changed it.
        bounded = inputs["contracts"][["applicant_id"]].merge(original[["applicant_id", "estimated_PD_bounded"]], on="applicant_id", validate="one_to_one")
        bounded = bounded.rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
        bounded.insert(1, "model_id", rl.MODEL_ID)
        expected_assessment = rl.assess_bank_profit(inputs["contracts"], bounded)
        adapter = inputs["contracts"][["applicant_id", "L_i"]].merge(expected_assessment, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, rl.BANK_INPUT_COLUMNS]
        internal_decisions = decisions.drop(columns=["information_scope", "scenario_id", "policy_id"]).rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
        internal_candidates = bank["candidates"].drop(columns=["information_scope", "scenario_id", "policy_id"])
        rl.exact_cutoff_diagnostics(adapter, internal_decisions, internal_candidates, bank["selection"], inputs["budget"])
        actual_assessment = bank["assessments"].drop(columns=["information_scope", "scenario_id", "policy_id"]).rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
        expected_assessment = expected_assessment.drop(columns="model_id")
        pd.testing.assert_frame_equal(actual_assessment.sort_values("applicant_id").reset_index(drop=True), expected_assessment.sort_values("applicant_id").reset_index(drop=True), check_exact=True)
    output = paths(root)
    if any(str(p.relative_to(root)) in hashes for k, p in output.items() if k != "preservation"):
        raise ValueError("Step 9B output would overwrite a frozen input")
    for path in output.values(): path.parent.mkdir(parents=True, exist_ok=True)
    combined = {}
    for name in ("decisions", "assessments", "candidates"):
        sort_columns = ["scenario_id", "candidate_id" if name == "candidates" else "applicant_id"]
        combined[name] = pd.concat([banks[s][name] for s in rl.SCENARIO_IDS], ignore_index=True).sort_values(sort_columns).reset_index(drop=True)
        if name == "candidates":
            write_csv(output[name], combined[name])
        else:
            write_csv(output[f"{name}_csv"], combined[name])
            with output[f"{name}_parquet"].open("xb") as stream:
                combined[name].to_parquet(stream, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(output[f"{name}_parquet"]), combined[name], check_exact=True)
    selections = {s: {"selection": banks[s]["selection"], "checks": banks[s]["checks"]} for s in rl.SCENARIO_IDS}
    write_json(output["selections"], {"status": "frozen_before_truth", "policies": selections})
    sources = source_hashes()
    checkpoint = {"schema_version": rl.SCHEMA_VERSION, "status": "all_three_policies_frozen_before_evaluation",
                  "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                  "configuration_fingerprint": configuration_fingerprint(config), "configuration_file_sha256": file_hash(config_path),
                  "source_code_hashes": sources, "source_fingerprint": configuration_fingerprint(sources),
                  "preservation_manifest_sha256": file_hash(output["preservation"]),
                  "input_checksums": hashes,
                  "artifacts": {k: {"path": str(output[k].relative_to(root)), "sha256": file_hash(output[k])} for k in BANK_OUTPUTS}}
    write_json(output["freeze"], checkpoint)
    verify_freeze(config, root=root, config_path=config_path)
    return checkpoint


def verify_freeze(config, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    output = paths(root)
    if not output["freeze"].is_file():
        raise ValueError("all scenario decisions must be saved and frozen before evaluation")
    checkpoint = json.loads(output["freeze"].read_text())
    if checkpoint["status"] != "all_three_policies_frozen_before_evaluation" or checkpoint["schema_version"] != rl.SCHEMA_VERSION or checkpoint["configuration_fingerprint"] != configuration_fingerprint(config) or checkpoint["configuration_file_sha256"] != file_hash(config_path) or set(checkpoint["artifacts"]) != set(BANK_OUTPUTS):
        raise ValueError("invalid bank freeze identity")
    if checkpoint["source_fingerprint"] != configuration_fingerprint(checkpoint["source_code_hashes"]):
        raise ValueError("invalid freeze source fingerprint")
    verify_hashes(checkpoint["source_code_hashes"], PROJECT_ROOT)
    verify_hashes(checkpoint["input_checksums"], root)
    if file_hash(output["preservation"]) != checkpoint["preservation_manifest_sha256"]:
        raise ValueError("preservation manifest changed")
    for name, item in checkpoint["artifacts"].items():
        if item["path"] != str(output[name].relative_to(root)):
            raise ValueError("frozen bank output path changed")
        verify_hashes({item["path"]: item["sha256"]}, root)
    return checkpoint


def load_evaluator_inputs(inputs, config, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    """The first numeric truth/reference read occurs only after verified freeze."""
    checkpoint = verify_freeze(config, root=root, config_path=config_path)
    def read(relative):
        if str(relative) not in checkpoint["input_checksums"]:
            raise ValueError("unregistered evaluator reference")
        path = root / relative
        return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_parquet(path)
    truth = read(Path("data/processed/professor_model/true_expected_profit.parquet"))
    selections = json.loads(paths(root)["selections"].read_text())["policies"]
    combined = pd.read_parquet(paths(root)["decisions_parquet"])
    policies = {s: rl.evaluation_policy(combined.loc[combined.scenario_id == s], selections[s]["selection"],
                                      score_column="estimated_PD_bounded", role="step9b_scenario") for s in rl.SCENARIO_IDS}
    reference_specs = (
        (rl.REFERENCE_IDS[0], "lending_decisions.parquet", "bank_PD_bounded", "lending_selection.json", False),
        (rl.REFERENCE_IDS[1], "lending_decisions_group_blind.parquet", "bank_PD_group_blind_bounded", "group_blind_lending_selection.json", False),
        (rl.REFERENCE_IDS[2], "oracle_lending_decisions.parquet", "true_PD", "oracle_comparison_validation.json", True),
    )
    for label, filename, score, selected_file, oracle in reference_specs:
        original = read(Path("data/processed/professor_model") / filename)
        report = read(Path("results/metrics/professor_model") / selected_file)
        selection = report["oracle_selection" if oracle else "selection"]
        policies[label] = rl.evaluation_policy(original, selection, score_column=score, role="frozen_reference", oracle=oracle)
    # These two new policies must reproduce the prior decisions, not just totals.
    for scenario, reference in ((rl.SCENARIO_IDS[0], rl.REFERENCE_IDS[0]), (rl.SCENARIO_IDS[2], rl.REFERENCE_IDS[1])):
        pd.testing.assert_frame_equal(policies[scenario]["decisions"].sort_values("applicant_id").reset_index(drop=True),
                                      policies[reference]["decisions"].sort_values("applicant_id").reset_index(drop=True), check_exact=True)
        for key in ("optimal_tau", "number_approved", "total_funded_principal", "bank_expected_portfolio_profit"):
            if policies[scenario]["selection"][key] != policies[reference]["selection"][key]:
                raise ValueError("unchanged-risk policy does not reproduce frozen reference")
    return policies, truth, checkpoint


def save_evaluation(inputs, policies, truth, config, originals, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    checkpoint = verify_freeze(config, root=root, config_path=config_path)
    output = paths(root)
    portfolio, groups, overlap = rl.evaluate_policies(policies, inputs["contracts"], inputs["audit"], truth, inputs["budget"])
    decisions = pd.read_parquet(output["decisions_parquet"])
    ties = rl.zero_tie_outcomes(inputs["risks"], decisions, inputs["contracts"], inputs["budget"])
    prior_path = Path("results/tables/professor_model/revised_scenarios_zero_tie_diagnostics.csv")
    if str(prior_path) not in checkpoint["input_checksums"]:
        raise ValueError("unregistered Step 9A tie diagnostic")
    prior = pd.read_csv(root / prior_path, float_precision="round_trip").set_index("scenario_id")
    for row in ties.itertuples(index=False):
        saved = prior.loc[row.scenario_id]
        if row.zero_score_count != saved.zero_score_count or row.B0_zero_count != saved.B0_zero_score_count or row.B1_zero_count != saved.B1_zero_score_count or row.zero_block_requested_principal_USD != saved.zero_score_requested_principal_USD or row.zero_block_feasible != saved.entire_zero_score_group_feasible:
            raise ValueError("recomputed tie block differs from frozen Step 9A diagnostic")
    for key, table in (("portfolio", portfolio), ("groups", groups), ("overlap", overlap), ("ties", ties)):
        write_csv(output[key], table)
    from .revised_lending_figures import save_figures
    save_figures(portfolio, groups, output["profit_figure"], output["approval_figure"])
    # Independently aggregate saved decisions without reselecting any cutoffs.
    for scenario in rl.SCENARIO_IDS:
        issued_ids = decisions.loc[(decisions.scenario_id == scenario) & decisions.approved, "applicant_id"]
        issued_truth = truth.loc[truth.applicant_id.isin(set(issued_ids))]
        row = portfolio.loc[portfolio.policy_id == scenario].iloc[0]
        if len(issued_ids) != row.number_approved:
            raise ValueError("saved decision count invalid")
        if math.fsum(issued_truth.expected_profit_true) != row.true_expected_portfolio_profit_USD:
            raise ValueError("saved decision true-profit sum mismatch")
        funded = decisions.loc[decisions.scenario_id == scenario, "funded_principal"]
        if math.fsum(funded) != row.funded_principal_USD:
            raise ValueError("saved funded-principal sum mismatch")
    verify_hashes(originals, root)
    verify_freeze(config, root=root, config_path=config_path)
    report = {"status": "Step_9B_validated", "information_scope": rl.EVALUATOR_SCOPE,
              "scenario_policies": 3, "frozen_reference_policies": 3,
              "decision_rows": len(decisions), "portfolio_rows": len(portfolio), "group_rows": len(groups),
              "pair_count": len(overlap[["policy_first", "policy_second"]].drop_duplicates()),
              "original_files_preserved": len(originals), "originals_byte_identical": True,
              "step9a_scores_exact": True, "unchanged_risk_policies_reproduce_frozen_decisions": True,
              "zero_tie_diagnostics_reproduce_step9a": True,
              "independent_exact_enumeration_all_passed": True, "saved_decision_aggregates_verified": True,
              "bank_freeze_sha256": file_hash(output["freeze"]), "bank_frozen_at_utc": checkpoint["frozen_at_utc"],
              "evaluator_completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "truth_loaded_only_after_bank_freeze": True, "model_fits": 0, "random_draws": 0,
              "oracle_optimizations": 0, "frozen_reference_reoptimizations": 0,
              "risk_adjustment_changes": 0, "maximum_budget_violation_USD": 0.}
    write_json(output["validation"], report)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"schema_version": rl.SCHEMA_VERSION, "status": "Step_9B_complete", "information_scope": rl.EVALUATOR_SCOPE,
                "configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
                "configuration_file_sha256": file_hash(config_path), "git_revision": revision,
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout),
                "source_code_hashes": source_hashes(), "source_fingerprint": configuration_fingerprint(source_hashes()),
                "python": platform.python_version(), "dependencies": {p: version(p) for p in ("numpy", "pandas", "pyarrow", "matplotlib", "PyYAML")},
                "evaluation_ID_fingerprint": inputs["evaluation_ID_fingerprint"],
                "bank_freeze_sha256": file_hash(output["freeze"]), "preservation_manifest_sha256": file_hash(output["preservation"]),
                "lender_information_boundary": "Only saved estimated_PD_bounded, original contracts and saved capital; no group targets or truth.",
                "event_order": ["risk_and_contracts_verified", "three_bank_policies_optimized", "all_decisions_saved_and_frozen", "evaluator_truth_and_frozen_references_loaded", "saved_decisions_evaluated"],
                "profit_comparison_direction": "policy minus frozen reference; signed, not unrestricted-optimal regret",
                "artifacts": {k: {"path": str(p.relative_to(root)), "sha256": file_hash(p)} for k, p in output.items() if k != "metadata"}}
    write_json(output["metadata"], metadata)
    return report
