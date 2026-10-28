"""Step 9C: preserve old artifacts, freeze valid policies, then release truth."""
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

from . import mechanism_audit as audit
from . import revised_lending as lending
from . import revised_lending_io as prior
from .config import PROJECT_ROOT, configuration_fingerprint
from .revised_scenarios import EXACT_BUDGET
from .risk_io import file_hash

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/mechanism_audit.yaml"
DATA = Path("data/processed/professor_model/mechanism_audit")
TABLES = Path("results/tables/professor_model/mechanism_audit")
METRICS = Path("results/metrics/professor_model/mechanism_audit")
FIGURES = Path("results/figures/professor_model/mechanism_audit")
BANK_KEYS = ("risk_csv", "risk_parquet", "definitions", "risk_diagnostics", "ties", "derivation",
             "decisions_csv", "decisions_parquet", "assessments_csv", "assessments_parquet", "candidates", "selections")


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    expected = {"schema_version": audit.SCHEMA_VERSION, "evaluation_rows": config.get("evaluation_rows"),
                "expected_capital_USD": EXACT_BUDGET, "source_step9b_config": "configs/professor_model/revised_lending.yaml",
                "direct_adjustment_per_B": -.1,
                "positive_floor": {"source": "frozen_blind_control_strictly_positive_bounded_evaluation_scores",
                                   "quantile": .01, "method": "linear",
                                   "replace": "exact_zeros_of_frozen_direct_bounded_scores_only", "outcome_tuning": False},
                "diagnostic_C_upper_bound": "report_above_one_without_upper_clipping_never_lend",
                "policy_variants": list(audit.POLICIES), "diagnostic_variants": [audit.B, audit.C],
                "policy_rule": "unchanged_common_cutoff_exact_ties_and_capital"}
    if config != expected or type(config["evaluation_rows"]) is not int or config["evaluation_rows"] < 1:
        raise ValueError("Step 9C requires its predeclared variants, 1% zero-only floor and original policy/budget")
    return config


def paths(root=PROJECT_ROOT):
    names = {"risk_csv": DATA / "variant_scores.csv", "risk_parquet": DATA / "variant_scores.parquet",
             "decisions_csv": DATA / "policy_decisions.csv", "decisions_parquet": DATA / "policy_decisions.parquet",
             "assessments_csv": DATA / "bank_assessments.csv", "assessments_parquet": DATA / "bank_assessments.parquet",
             "definitions": TABLES / "variant_definitions.csv", "risk_diagnostics": TABLES / "risk_clipping_diagnostics.csv",
             "ties": TABLES / "tie_capital_diagnostics.csv", "candidates": TABLES / "cutoff_candidates.csv",
             "portfolio": TABLES / "portfolio_comparison.csv", "groups": TABLES / "group_outcome_audit.csv",
             "overlap": TABLES / "pairwise_allocation_overlap.csv", "derivation": METRICS / "positive_floor_derivation.json",
             "selections": METRICS / "bank_selections.json", "freeze": METRICS / "bank_freeze_checkpoint.json",
             "preservation": METRICS / "preservation_manifest.json", "validation": METRICS / "validation.json",
             "manifest": METRICS / "run_manifest.json", "risk_figure": FIGURES / "raw_and_transformed_scores.png",
             "tie_figure": FIGURES / "tie_blocks_and_capital.png", "outcome_figure": FIGURES / "approval_and_true_profit.png"}
    return {k: root / p for k, p in names.items()}


def require_new_outputs(root=PROJECT_ROOT):
    if any(p.exists() for k, p in paths(root).items() if k != "preservation"):
        raise FileExistsError("Step 9C never overwrites existing audit outputs")


def preservation_manifest(root=PROJECT_ROOT):
    path = paths(root)["preservation"]
    if not path.exists():
        files = set()
        for folder in ("src", "configs", "experiments", "tests", "docs", "data/synthetic/professor_model",
                       "data/processed/professor_model", "results/tables/professor_model",
                       "results/metrics/professor_model", "results/figures/professor_model"):
            files.update(p for p in (root / folder).rglob("*") if p.is_file() and p.name != ".DS_Store"
                         and "__pycache__" not in p.parts and p != root / "docs/PROJECT_STATUS.md"
                         and not any(p.is_relative_to(root / d) for d in (DATA, TABLES, METRICS, FIGURES)))
        prior.write_json(path, {"schema_version": "professor-mechanism-preservation-v1",
                                "checksums": {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}})
    record = json.loads(path.read_text())
    if record["schema_version"] != "professor-mechanism-preservation-v1" or not record["checksums"]:
        raise ValueError("invalid Step 9C preservation manifest")
    prior.verify_hashes(record["checksums"], root)
    return record["checksums"]


def source_hashes():
    hashes = prior.source_hashes()
    for name in ("mechanism_audit", "mechanism_audit_io", "mechanism_audit_figures"):
        path = f"src/fair_lending/professor_model/{name}.py"
        hashes[path] = file_hash(PROJECT_ROOT / path)
    path = "experiments/run_professor_mechanism_audit.py"
    hashes[path] = file_hash(PROJECT_ROOT / path)
    return hashes


def load_inputs(config, originals, root=PROJECT_ROOT):
    """Verify Step 9B and load frozen risk/contracts, not numerical truth."""
    old_config_path = root / config["source_step9b_config"]
    old_config = prior.load_config(old_config_path)
    prior.verify_freeze(old_config, root=root, config_path=old_config_path)
    old_manifest_path = prior.paths(root)["metadata"]
    manifest = json.loads(old_manifest_path.read_text())
    if manifest["status"] != "Step_9B_complete" or manifest["configuration_fingerprint"] != configuration_fingerprint(old_config) or manifest["source_fingerprint"] != configuration_fingerprint(manifest["source_code_hashes"]):
        raise ValueError("invalid frozen Step 9B manifest")
    prior.verify_hashes(manifest["source_code_hashes"], PROJECT_ROOT)
    for name, item in manifest["artifacts"].items():
        if item["path"] != str(prior.paths(root)[name].relative_to(root)):
            raise ValueError("Step 9B artifact path differs")
        prior.verify_hashes({item["path"]: item["sha256"]}, root)
    inputs, hashes = prior.load_risk_inputs(old_config, originals, root)
    hashes[str(old_manifest_path.relative_to(root))] = file_hash(old_manifest_path)
    for item in manifest["artifacts"].values(): hashes[item["path"]] = item["sha256"]
    if len(inputs["contracts"]) != config["evaluation_rows"] or inputs["budget"] != config["expected_capital_USD"]:
        raise ValueError("held-out cohort or exact capital changed")
    inputs.update(old_config=old_config, old_config_path=old_config_path)
    return inputs, hashes


def _check_bank_bundle(inputs, risks, banks):
    if set(banks) != set(audit.POLICIES): raise ValueError("only A/D/E policies must be frozen")
    for variant, bank in banks.items():
        expected_risk = risks.loc[risks.variant_id == variant].sort_values("applicant_id")
        decisions = bank["decisions"].sort_values("applicant_id")
        np.testing.assert_array_equal(expected_risk.applicant_id, decisions.applicant_id)
        np.testing.assert_array_equal(expected_risk.score, decisions.estimated_PD_bounded)
        expected_columns = tuple("variant_id" if c == "scenario_id" else c for c in lending.DECISION_COLUMNS)
        if tuple(decisions.columns) != expected_columns or not decisions.variant_id.eq(variant).all() or not decisions.policy_id.eq(variant).all() or not decisions.information_scope.eq(lending.BANK_SCOPE).all():
            raise ValueError("invalid bank-only mechanism decisions")
        bounded = inputs["contracts"][["applicant_id"]].merge(expected_risk[["applicant_id", "score"]], on="applicant_id", validate="one_to_one").rename(columns={"score": "bank_PD_bounded"})
        bounded.insert(1, "model_id", lending.MODEL_ID)
        assessment = lending.assess_bank_profit(inputs["contracts"], bounded)
        adapter = inputs["contracts"][["applicant_id", "L_i"]].merge(assessment, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, lending.BANK_INPUT_COLUMNS]
        internal_decisions = decisions.drop(columns=["information_scope", "variant_id", "policy_id"]).rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
        candidates = bank["candidates"].drop(columns=["information_scope", "variant_id", "policy_id"])
        lending.exact_cutoff_diagnostics(adapter, internal_decisions, candidates, bank["selection"], inputs["budget"])
        actual = bank["assessments"].drop(columns=["information_scope", "variant_id", "policy_id"]).rename(columns={"estimated_PD_bounded": "bank_PD_bounded"})
        pd.testing.assert_frame_equal(actual.sort_values("applicant_id").reset_index(drop=True), assessment.drop(columns="model_id").sort_values("applicant_id").reset_index(drop=True), check_exact=True)


def freeze_banks(inputs, risks, derivation, banks, config, hashes, originals, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    require_new_outputs(root)
    prior.verify_hashes(originals, root); prior.verify_hashes(hashes, root)
    checks = audit.validate_variants(risks, inputs["risks"], derivation)
    _check_bank_bundle(inputs, risks, banks)
    output = paths(root)
    if any(str(p.relative_to(root)) in hashes for k, p in output.items() if k != "preservation"):
        raise ValueError("audit output collides with frozen evidence")
    for path in output.values(): path.parent.mkdir(parents=True, exist_ok=True)
    prior.write_csv(output["risk_csv"], risks)
    with output["risk_parquet"].open("xb") as stream: risks.to_parquet(stream, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output["risk_parquet"]), risks, check_exact=True)
    prior.write_csv(output["definitions"], audit.definitions(derivation))
    prior.write_csv(output["risk_diagnostics"], audit.risk_diagnostics(risks))
    combined = {}
    for name in ("decisions", "assessments", "candidates"):
        combined[name] = pd.concat([banks[v][name] for v in audit.POLICIES], ignore_index=True).sort_values(
            ["variant_id", "candidate_id" if name == "candidates" else "applicant_id"]).reset_index(drop=True)
        if name == "candidates": prior.write_csv(output[name], combined[name])
        else:
            prior.write_csv(output[f"{name}_csv"], combined[name])
            with output[f"{name}_parquet"].open("xb") as stream: combined[name].to_parquet(stream, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(output[f"{name}_parquet"]), combined[name], check_exact=True)
    ties = audit.tie_diagnostics(risks, combined["decisions"], inputs["contracts"], banks, inputs["budget"], derivation["floor_raw_probability"])
    prior.write_csv(output["ties"], ties)
    prior.write_json(output["derivation"], derivation)
    prior.write_json(output["selections"], {"status": "frozen_before_truth", "banks": {v: {"selection": banks[v]["selection"], "checks": banks[v]["checks"]} for v in audit.POLICIES}})
    sources = source_hashes()
    checkpoint = {"schema_version": audit.SCHEMA_VERSION, "status": "all_valid_policies_frozen_before_evaluation",
                  "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "configuration_fingerprint": configuration_fingerprint(config),
                  "configuration_file_sha256": file_hash(config_path), "source_code_hashes": sources,
                  "source_fingerprint": configuration_fingerprint(sources), "preservation_manifest_sha256": file_hash(output["preservation"]),
                  "input_checksums": hashes, "risk_checks": checks,
                  "artifacts": {k: {"path": str(output[k].relative_to(root)), "sha256": file_hash(output[k])} for k in BANK_KEYS}}
    prior.write_json(output["freeze"], checkpoint)
    verify_freeze(config, root=root, config_path=config_path)
    return checkpoint


def verify_freeze(config, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    output = paths(root)
    if not output["freeze"].is_file(): raise ValueError("Step 9C policies must be frozen before truth access")
    record = json.loads(output["freeze"].read_text())
    if record["schema_version"] != audit.SCHEMA_VERSION or record["status"] != "all_valid_policies_frozen_before_evaluation" or record["configuration_fingerprint"] != configuration_fingerprint(config) or record["configuration_file_sha256"] != file_hash(config_path) or set(record["artifacts"]) != set(BANK_KEYS):
        raise ValueError("invalid mechanism freeze identity")
    if record["source_code_hashes"] != source_hashes() or record["source_fingerprint"] != configuration_fingerprint(record["source_code_hashes"]):
        raise ValueError("invalid or changed mechanism source fingerprint")
    prior.verify_hashes(record["source_code_hashes"], PROJECT_ROOT)
    prior.verify_hashes(record["input_checksums"], root)
    if record["preservation_manifest_sha256"] != file_hash(output["preservation"]): raise ValueError("preservation manifest changed")
    for k, item in record["artifacts"].items():
        if item["path"] != str(output[k].relative_to(root)): raise ValueError("frozen output path changed")
        prior.verify_hashes({item["path"]: item["sha256"]}, root)
    return record


def load_evaluator(inputs, config, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    verify_freeze(config, root=root, config_path=config_path)
    # Both the NEW freeze above and the preserved Step 9B freeze are required.
    old_policies, truth, _ = prior.load_evaluator_inputs(inputs, inputs["old_config"], root=root, config_path=inputs["old_config_path"])
    decisions = pd.read_parquet(paths(root)["decisions_parquet"])
    saved = json.loads(paths(root)["selections"].read_text())["banks"]
    banks = {v: {"decisions": decisions.loc[decisions.variant_id == v].copy(), **saved[v]} for v in audit.POLICIES}
    for variant, old_scenario in ((audit.A, lending.SCENARIO_IDS[1]), (audit.E, lending.SCENARIO_IDS[2])):
        normalized = lending.evaluation_policy(banks[variant]["decisions"], banks[variant]["selection"], score_column="estimated_PD_bounded", role="step9c_valid_policy")
        pd.testing.assert_frame_equal(normalized["decisions"].sort_values("applicant_id").reset_index(drop=True), old_policies[old_scenario]["decisions"].sort_values("applicant_id").reset_index(drop=True), check_exact=True)
        for key in ("optimal_tau", "number_approved", "total_funded_principal", "bank_expected_portfolio_profit"):
            if banks[variant]["selection"][key] != old_policies[old_scenario]["selection"][key]: raise ValueError("A/E fail exact Step 9B reproduction")
    return banks, {key: old_policies[key] for key in lending.REFERENCE_IDS}, truth


def save_evaluation(inputs, risks, derivation, banks, references, truth, config, originals, *, root=PROJECT_ROOT, config_path=CONFIG_PATH):
    checkpoint = verify_freeze(config, root=root, config_path=config_path)
    output = paths(root)
    pd.testing.assert_frame_equal(risks, pd.read_parquet(output["risk_parquet"]), check_exact=True)
    if derivation != json.loads(output["derivation"].read_text()): raise ValueError("frozen floor derivation changed")
    frozen_decisions = pd.read_parquet(output["decisions_parquet"])
    frozen_selections = json.loads(output["selections"].read_text())["banks"]
    for variant in audit.POLICIES:
        expected = frozen_decisions.loc[frozen_decisions.variant_id == variant].sort_values("applicant_id").reset_index(drop=True)
        actual = banks[variant]["decisions"].sort_values("applicant_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        if banks[variant]["selection"] != frozen_selections[variant]["selection"]: raise ValueError("frozen bank selection changed")
    portfolio, groups, overlap = audit.evaluate_policies(banks, references, inputs["contracts"], inputs["audit"], truth, inputs["budget"])
    for key, frame in (("portfolio", portfolio), ("groups", groups), ("overlap", overlap)): prior.write_csv(output[key], frame)
    from .mechanism_audit_figures import save_figures
    ties = pd.read_csv(output["ties"], float_precision="round_trip")
    save_figures(risks, ties, portfolio, groups, derivation, output)
    decisions = pd.read_parquet(output["decisions_parquet"])
    for variant in audit.POLICIES:
        issued = decisions.loc[(decisions.variant_id == variant) & decisions.approved]
        value = math.fsum(truth.loc[truth.applicant_id.isin(set(issued.applicant_id)), "expected_profit_true"])
        row = portfolio.loc[portfolio.policy_id == variant].iloc[0]
        if len(issued) != row.number_approved or value != row.true_expected_portfolio_profit_USD or math.fsum(issued.funded_principal) != row.funded_principal_USD:
            raise ValueError("independent saved decision aggregation differs")
    prior.verify_hashes(originals, root); verify_freeze(config, root=root, config_path=config_path)
    report = {"status": "Step_9C_validated", "information_scope": lending.EVALUATOR_SCOPE,
              "original_files_preserved": len(originals), "all_originals_byte_identical": True,
              "variant_score_rows": len(risks), "diagnostic_variants": [audit.B, audit.C],
              "valid_policy_variants": list(audit.POLICIES), "policy_rows": len(portfolio), "group_rows": len(groups),
              "pair_count": len(overlap[["policy_first", "policy_second"]].drop_duplicates()),
              "A_and_E_exact_Step9A_Step9B_reproduction": True, "positive_floor_derivation": derivation,
              "negative_diagnostic_scores_never_used_in_lending": True, "independent_exact_enumeration_passed": True,
              "saved_decision_aggregate_checks_passed": True, "maximum_budget_violation_USD": 0.,
              "bank_frozen_at_utc": checkpoint["frozen_at_utc"], "evaluator_completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "truth_loaded_only_after_new_freeze": True, "model_fits": 0, "random_draws": 0,
              "new_populations": 0, "oracle_reoptimizations": 0,
              "limitation": "No additive allocation decomposition: B is invalid as a probability policy and D preserves a shared tie."}
    prior.write_json(output["validation"], report)
    sources = source_hashes()
    metadata = {"schema_version": audit.SCHEMA_VERSION, "status": "Step_9C_complete", "information_scope": lending.EVALUATOR_SCOPE,
                "configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
                "configuration_file_sha256": file_hash(config_path), "source_code_hashes": sources, "source_fingerprint": configuration_fingerprint(sources),
                "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout),
                "python": platform.python_version(), "dependencies": {p: version(p) for p in ("numpy", "pandas", "pyarrow", "matplotlib", "PyYAML")},
                "evaluation_ID_fingerprint": inputs["evaluation_ID_fingerprint"], "positive_floor_derivation": derivation,
                "event_order": ["predeclared_1pct_zero_only_rule", "frozen_sources_verified", "scores_and_floor_derived_without_truth", "valid_policies_saved_and_frozen", "truth_released", "frozen_masks_evaluated"],
                "bank_freeze_sha256": file_hash(output["freeze"]),
                "artifacts": {k: {"path": str(p.relative_to(root)), "sha256": file_hash(p)} for k, p in output.items() if k != "manifest"}}
    prior.write_json(output["manifest"], metadata)
    return report
