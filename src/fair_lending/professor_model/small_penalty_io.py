"""Immutable smaller-penalty successor: frozen scores, bank calibration, evaluator.

No estimator fitting, target generation or population generation is performed.
Parent truth is hashed as provenance but numeric true profit is loaded only
following the selected bank-artifact freeze.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np
import pandas as pd
import yaml

from . import nov2_followup as old, nov2_followup_io as parent, small_penalty as core
from .config import PROJECT_ROOT, configuration_fingerprint

NAMESPACE = "small_penalty"
CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/small_penalty.yaml"
COMPLETE = "complete_small_penalty_prediction_only"
FROZEN_CONFIG = {
    "schema_version": core.EXPERIMENT_ID, "parent_study": "nov2_followup",
    "penalty_candidates": list(core.CANDIDATES), "common_coefficient": True,
    "selection_rule": "first_descending_candidate_with_positive_B_and_lower_rate_than_A_in_both_worlds",
    "selection_information": "prediction_cohort_bank_approval_counts_only",
    "minimum_B_approvals": 1, "profit_used_for_penalty_selection": False,
    "training_rows": 80000, "prediction_rows": 20000, "budget": old.EXACT_BUDGET,
    "refit": False, "new_noise": False, "new_population": False,
    "policy": "bounded_PD_common_cutoff_full_requests_indivisible_ties_reject_all",
    "reporting": "total_true_expected_profit_from_funded_prediction_loans_by_group",
}
file_hash = parent.file_hash
write_json = parent.write_json
write_csv = parent.write_csv
write_pair = parent.write_pair
verify_hashes = parent.verify_hashes


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    if configuration_fingerprint(config) != configuration_fingerprint(FROZEN_CONFIG):
        raise ValueError("small-penalty settings differ from the predeclared specification")
    return config


def paths(root=PROJECT_ROOT):
    return {name: Path(root) / rel for name, rel in {
        "processed": "data/processed/professor_model/small_penalty",
        "tables": "results/tables/professor_model/small_penalty",
        "metrics": "results/metrics/professor_model/small_penalty",
        "figures": "results/figures/professor_model/small_penalty"}.items()}


def source_hashes(root=PROJECT_ROOT):
    files = [f"src/fair_lending/professor_model/{name}.py" for name in
             ("small_penalty", "small_penalty_io", "small_penalty_figures")]
    files += ["configs/professor_model/small_penalty.yaml", "experiments/run_professor_small_penalty.py"]
    return {**parent.source_hashes(root), **{name: file_hash(Path(root) / name) for name in files}}


def _preservation(root):
    supplied = root / "tmp/small_penalty/preexisting_hashes.json"
    if supplied.exists():
        result = json.loads(supplied.read_text())
    else:
        files = {root / p for p in subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()}
        for name in ("data/synthetic/professor_model", "data/processed/professor_model",
                     "results/tables/professor_model", "results/metrics/professor_model",
                     "results/models/professor_model", "results/figures/professor_model"):
            files.update(p for p in (root / name).rglob("*") if p.is_file() and NAMESPACE not in p.parts)
        result = {"hashes": {str(p.relative_to(root)): file_hash(p) for p in sorted(files) if p.is_file()},
                  "allowed_documentation_edits": ["docs/CURRENT_RESEARCH.md", "docs/PROJECT_STATUS.md"]}
    verify_hashes(result["hashes"], root, allowed=result["allowed_documentation_edits"])
    return result


def _progress(stage, **extra):
    print(json.dumps({"stage": stage, **extra}), flush=True)


def _parent_bindings(root):
    output = parent.paths(root)
    path = output["metrics"] / "run_manifest.json"
    record = json.loads(path.read_text())
    config = parent.load_config()
    if (record["status"] != parent.STATUS
            or record["configuration_fingerprint"] != configuration_fingerprint(config)
            or record["source_hashes"] != parent.source_hashes(root)
            or record["dependency_versions"] != parent._dependencies()):
        raise ValueError("parent scientific identity differs")
    # Hash-only provenance verification: do not parse evaluator-bearing parent
    # summary tables while choosing the successor penalty.
    for key in ("artifacts", "input_hashes", "source_hashes"):
        verify_hashes(record[key], root)
    for name, expected in record["external_inputs"].items():
        if file_hash(Path(name)) != expected: raise ValueError("parent reference changed")
    parent.verify_freeze(config, root=root)
    preservation = json.loads((output["metrics"] / "preservation_manifest.json").read_text())
    verify_hashes(preservation["hashes"], root, allowed=preservation["allowed_documentation_edits"])
    return output, record, {str(path.relative_to(root)): file_hash(path), **record["artifacts"],
                             **record["input_hashes"], **record["source_hashes"]}


def _selection_record(result, config):
    attempted = sorted(set(result["attempts"].candidate_index))
    return {"status": result["status"], "selected_delta": result["selected_delta"],
            "attempted_candidates": [config["penalty_candidates"][i] for i in attempted],
            "candidate_count": len(attempted), "scenario_attempt_count": len(result["attempts"]),
            "criterion": "approved_B > 0 and approved_B*n_A < approved_A*n_B in BOTH scenarios",
            "selection_uses_true_profit": False,
            "interpretation": "illustrative approval-based calibration on prediction cohort; not untouched confirmatory policy evaluation"}


def _augment(frame, delta):
    """Add new experiment context; inherited method labels retain their meaning."""
    result = frame.copy()
    result.insert(0, "experiment_id", core.EXPERIMENT_ID)
    if "method_id" in result:
        result["penalty_delta"] = np.where(result.method_id.eq("ols_penalty"), delta, 0.)
    else:
        result["penalty_delta"] = delta
    return result


def _plain(frame):
    return frame.drop(columns=[name for name in ("experiment_id", "penalty_delta") if name in frame])


def _saved_banks(output):
    frames = {name: _plain(pd.read_parquet(output["processed"] / f"{name}.parquet"))
              for name in ("predictions", "assessments", "decisions")}
    candidates = _plain(pd.read_csv(output["tables"] / "cutoff_candidates.csv", float_precision="round_trip"))
    selections = json.loads((output["metrics"] / "bank_selections.json").read_text())
    worlds = {}
    for scenario in core.SCENARIOS:
        worlds[scenario] = {}
        for method in core.METHODS:
            bundle = {name: frame.loc[frame.scenario_id.eq(scenario) & frame.method_id.eq(method)].reset_index(drop=True)
                      for name, frame in frames.items()}
            bundle["candidates"] = candidates.loc[candidates.scenario_id.eq(scenario) & candidates.method_id.eq(method)].reset_index(drop=True)
            bundle.update(selections[scenario][method]); worlds[scenario][method] = bundle
    return worlds


def verify_freeze(config, *, root=PROJECT_ROOT):
    output = paths(root)
    frozen = json.loads((output["metrics"] / "bank_freeze_checkpoint.json").read_text())
    if (frozen["status"] != "selected_small_penalty_bank_policies_frozen_before_truth"
            or frozen["configuration_fingerprint"] != configuration_fingerprint(config)):
        raise ValueError("invalid bank freeze identity")
    for name in ("artifacts", "parent_hashes", "source_hashes"):
        verify_hashes(frozen[name], root)
    if frozen["source_hashes"] != source_hashes(root) or frozen["dependency_versions"] != parent._dependencies():
        raise ValueError("frozen scientific source/dependencies changed")
    return frozen


def _manifest(config, config_path, output, root, status, parent_hashes, sources, started, extra):
    paths_to_hash = [p for folder in output.values() for p in folder.rglob("*") if p.is_file()]
    record = {"schema_version": core.EXPERIMENT_ID, "status": status,
              "configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
              "configuration_file_sha256": file_hash(config_path), "parent_hashes": parent_hashes,
              "source_hashes": sources, "dependency_versions": parent._dependencies(),
              "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
              "artifacts": {str(p.relative_to(root)): file_hash(p) for p in paths_to_hash},
              "new_fits": 0, "new_noise_draws": 0, "new_populations": 0, **extra}
    write_json(output["metrics"] / "run_manifest.json", record)


def run_experiment(config_path=CONFIG_PATH, *, root=PROJECT_ROOT):
    """Reuse four policies, calibrate only bank-side penalty, then evaluate."""
    from .small_penalty_figures import save_figures
    root = Path(root); config_path = Path(config_path); config = load_config(config_path)
    output = paths(root)
    if any(p.is_file() for folder in output.values() for p in folder.rglob("*")):
        raise FileExistsError("small-penalty outputs exist; use read-only --summarize-only")
    originals = _preservation(root)
    parent_out, parent_manifest, parent_hashes = _parent_bindings(root)
    sources = source_hashes(root); started = datetime.now(timezone.utc).isoformat()
    for folder in output.values(): folder.mkdir(parents=True, exist_ok=True)
    write_json(output["metrics"] / "resolved_configuration.json", config)
    write_json(output["metrics"] / "preservation_manifest.json", originals)
    write_json(output["metrics"] / "parent_binding.json", {
        "parent_status": parent_manifest["status"], "parent_manifest_sha256": file_hash(parent_out["metrics"] / "run_manifest.json"),
        "parent_hashes": parent_hashes, "source_run_revision": parent_manifest["git_revision"]})
    stage = "bank_source_verification"
    try:
        _progress(stage)
        splits = pd.read_csv(parent_out["processed"] / "applicant_splits.csv")
        contracts = pd.read_parquet(parent_out["processed"] / "loan_contracts.parquet")
        original = parent._read_frozen_banks(parent_out)
        metadata = json.loads((parent_out["metrics"] / "model_metadata.json").read_text())
        audits, controls, split_rows = {}, {}, []
        for scenario in core.SCENARIOS:
            bank = pd.read_parquet(parent_out["synthetic"] / scenario / "bank.parquet")
            prediction_ids = set(contracts.applicant_id)
            audit = bank.loc[bank.applicant_id.isin(prediction_ids), ["applicant_id", "B"]].sort_values("applicant_id").reset_index(drop=True)
            controls[scenario] = original[scenario]["ols_control"]["predictions"]
            audits[scenario] = audit
            pieces = pd.concat([original[scenario][m]["predictions"] for m in ("ols_control", "ml_histgb")], ignore_index=True)
            check = core.validate_prediction_cohort(splits, contracts, pieces, audit)
            if check["training_rows"] != config["training_rows"] or check["prediction_rows"] != config["prediction_rows"]:
                raise ValueError("original cohort sizes changed")
            model = metadata[scenario]["ml"]
            if (model["training_count"] != 80000 or model["evaluation_count"] != 20000
                    or model["ML"]["fit_rows"] != 64000 or model["ML"]["validation_rows"] != 16000
                    or model["ML"]["final_refit_rows"] != 80000
                    or model["features"] != list(old.FEATURES[scenario])
                    or model["training_ID_fingerprint"] != check["training_ID_fingerprint"]
                    or model["evaluation_ID_fingerprint"] != check["prediction_ID_fingerprint"]):
                raise ValueError("saved fitting and prediction identities differ")
            split_rows.append({"scenario_id": scenario, **check, "OLS_features": ",".join(old.FEATURES[scenario]),
                               "ML_features": ",".join(old.FEATURES[scenario]), "ML_inner_fit": 64000,
                               "ML_inner_validation": 16000, "ML_final_refit": 80000,
                               "profits_prediction_only": True})
        stage = "bank_penalty_search"; _progress(stage)
        result = core.select_common_penalty(contracts, controls, audits, config["budget"],
                                            candidates=config["penalty_candidates"])
        selection = _selection_record(result, config)
        write_csv(output["tables"] / "penalty_search.csv", result["attempts"])
        write_csv(output["tables"] / "split_audit.csv", pd.DataFrame(split_rows))
        write_json(output["metrics"] / "selection.json", selection)
        write_json(output["metrics"] / "attempt_records.json", result["attempt_records"])
        write_json(output["metrics"] / "model_reuse.json", {
            "new_fits": 0, "new_ML_selection": False, "original_model_metadata": metadata,
            "parent_model_metadata_sha256": file_hash(parent_out["metrics"] / "model_metadata.json"),
            "source_models": {str(p.relative_to(root)): file_hash(p) for p in parent_out["models"].glob("*.joblib")}})
        if result["status"] == "criterion_unmet":
            verify_hashes(originals["hashes"], root, allowed=originals["allowed_documentation_edits"])
            _manifest(config, config_path, output, root, "criterion_unmet", parent_hashes, sources, started,
                      {"selected_delta": None, "stage_order": ["verified_parent", "bank_search_exhausted"],
                       "truth_numeric_loads": 0, "candidate_count": len(config["penalty_candidates"])})
            return summarize_only(config_path, root=root)
        delta = result["selected_delta"]
        worlds = {s: {m: original[s][m] if m != "ols_penalty" else result["selected_policies"][s]
                      for m in core.METHODS} for s in core.SCENARIOS}
        selections, frames = {}, {name: [] for name in ("predictions", "assessments", "decisions", "candidates")}
        for scenario, methods in worlds.items():
            selections[scenario] = {}
            for method, bundle in methods.items():
                parent.validate_policy_checks(bundle["selection"], bundle["checks"], config["budget"])
                for name in frames: frames[name].append(bundle[name])
                selections[scenario][method] = {"selection": bundle["selection"], "checks": bundle["checks"]}
            check = core.group_criterion(methods["ols_penalty"]["decisions"], audits[scenario])
            if check["criterion_met"] is not True: raise ValueError("selected common penalty fails approval criterion")
        for name, pieces in frames.items():
            frame = pd.concat(pieces, ignore_index=True)
            frame = frame.sort_values(["scenario_id", "method_id", "candidate_id" if name == "candidates" else "applicant_id"]).reset_index(drop=True)
            if name == "candidates": write_csv(output["tables"] / "cutoff_candidates.csv", _augment(frame, delta))
            else: write_pair(output["processed"] / name, _augment(frame, delta))
        write_pair(output["processed"] / "loan_contracts", contracts)
        write_csv(output["processed"] / "applicant_splits.csv", splits)
        write_csv(output["processed"] / "inner_training_splits.csv", pd.read_csv(parent_out["processed"] / "inner_training_splits.csv"))
        write_csv(output["processed"] / "group_audit.csv", pd.concat([audit.assign(scenario_id=s) for s, audit in audits.items()], ignore_index=True))
        write_json(output["metrics"] / "bank_selections.json", selections)
        stage = "bank_freeze"
        bank_files = [p for folder in output.values() for p in folder.rglob("*") if p.is_file()]
        write_json(output["metrics"] / "bank_freeze_checkpoint.json", {
            "status": "selected_small_penalty_bank_policies_frozen_before_truth",
            "configuration_fingerprint": configuration_fingerprint(config),
            "selected_delta": delta, "dependency_versions": parent._dependencies(),
            "source_hashes": sources, "parent_hashes": parent_hashes,
            "artifacts": {str(p.relative_to(root)): file_hash(p) for p in bank_files},
            "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        verify_freeze(config, root=root)
        # Numeric evaluator truth is released only after every chosen policy is sealed.
        stage = "evaluator"; _progress(stage, selected_delta=delta)
        saved = _saved_banks(output)
        outputs = {name: [] for name in ("portfolio", "groups", "risk_diagnostics", "penalty_comparisons")}
        for scenario in core.SCENARIOS:
            truth = pd.read_parquet(parent_out["processed"] / f"{scenario}_true_profit_EVALUATOR_ONLY.parquet")
            write_pair(output["processed"] / f"{scenario}_true_profit_EVALUATOR_ONLY", truth)
            target = pd.read_parquet(parent_out["synthetic"] / scenario / "targets.parquet")
            target = target.loc[target.applicant_id.isin(set(contracts.applicant_id))]
            evaluation = old.evaluate_policies(saved[scenario], contracts, audits[scenario], truth, targets=target)
            for name in outputs: outputs[name].append(evaluation[name])
        merged = {name: pd.concat(pieces, ignore_index=True) for name, pieces in outputs.items()}
        parent.validate_result_cells(merged["portfolio"], merged["groups"])
        baseline_checks = []
        for name, stem in (("portfolio", "portfolio_results"), ("groups", "group_results"),
                           ("risk_diagnostics", "risk_diagnostics"), ("penalty_comparisons", "penalty_comparisons")):
            frame = merged[name]
            if name in ("portfolio", "groups", "risk_diagnostics"):
                prior = pd.read_csv(parent_out["tables"] / f"{stem}.csv", float_precision="round_trip")
                reused = frame.loc[~frame.method_id.eq("ols_penalty")].reset_index(drop=True)
                prior = prior.loc[~prior.method_id.eq("ols_penalty")].reset_index(drop=True)
                pd.testing.assert_frame_equal(reused, prior, check_exact=True, check_dtype=False)
                baseline_checks.append(stem)
            write_csv(output["tables"] / f"{stem}.csv", _augment(frame, delta))
        for stem in ("ml_validation", "systemic_risk_audit"):
            source = parent_out["tables"] / f"{stem}.csv"
            target = output["tables"] / f"{stem}.csv"
            with target.open("xb") as stream: stream.write(source.read_bytes())
        stage = "figures"; save_figures(merged["groups"], output["figures"], delta)
        verify_freeze(config, root=root)
        verify_hashes(originals["hashes"], root, allowed=originals["allowed_documentation_edits"])
        write_json(output["metrics"] / "validation.json", {
            "status": "validated", "criterion_met_in_both_scenarios": True,
            "selected_delta": delta, "new_fits": 0, "new_noise": 0,
            "prediction_only_profits": True, "parent_baseline_metrics_exact": baseline_checks,
            "parent_control_ML_predictions_and_masks_exact": True,
            "all_six_cutoff_checks_valid": True,
            "prior_files_in_inventory": len(originals["hashes"]),
            "allowed_documentation_edits": originals["allowed_documentation_edits"],
            "calibration": "prediction-cohort approval-based illustrative selection; no true-profit selection"})
        _manifest(config, config_path, output, root, COMPLETE, parent_hashes, sources, started,
                  {"selected_delta": delta, "candidate_count": selection["candidate_count"],
                   "stage_order": ["parent_and_cohorts_verified", "bank_approval_search", "bank_artifacts_frozen", "evaluator_truth_loaded", "results_verified"],
                   "policies": 6, "group_cells": 12})
        return summarize_only(config_path, root=root)
    except Exception as exc:
        failure = output["metrics"] / "failure.json"
        if not failure.exists(): write_json(failure, {"status": "failed", "stage": stage,
            "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
            "source_hashes": sources, "configuration_fingerprint": configuration_fingerprint(config)})
        raise


def summarize_only(config_path=CONFIG_PATH, *, root=PROJECT_ROOT):
    """Verify completed or unmet evidence; never fits, optimizes or writes."""
    root = Path(root); config_path = Path(config_path); config = load_config(config_path)
    output = paths(root); path = output["metrics"] / "run_manifest.json"
    if not path.exists(): raise FileNotFoundError("No saved smaller-penalty study; summary-only starts no work")
    manifest = json.loads(path.read_text())
    if (manifest["status"] not in (COMPLETE, "criterion_unmet")
            or manifest["configuration_fingerprint"] != configuration_fingerprint(config)
            or manifest["configuration_file_sha256"] != file_hash(config_path)
            or manifest["source_hashes"] != source_hashes(root)
            or manifest["dependency_versions"] != parent._dependencies()):
        raise ValueError("incompatible smaller-penalty evidence")
    for name in ("artifacts", "source_hashes", "parent_hashes"): verify_hashes(manifest[name], root)
    _parent_bindings(root)
    preservation = json.loads((output["metrics"] / "preservation_manifest.json").read_text())
    verify_hashes(preservation["hashes"], root, allowed=preservation["allowed_documentation_edits"])
    selected = json.loads((output["metrics"] / "selection.json").read_text())
    search = pd.read_csv(output["tables"] / "penalty_search.csv", float_precision="round_trip")
    count = int(search.candidate_index.max()) + 1
    if (len(search) != 2 * count or count > len(core.CANDIDATES)
            or set(search.scenario_id) != set(core.SCENARIOS)
            or set(search.candidate_index) != set(range(count))
            or search.duplicated(["candidate_index", "scenario_id"]).any()):
        raise ValueError("incomplete penalty search")
    for index, frame in search.groupby("candidate_index", sort=True):
        if set(frame.scenario_id) != set(core.SCENARIOS) or not frame.delta.eq(core.CANDIDATES[int(index)]).all():
            raise ValueError("candidate order or coefficient mismatch")
        expected = frame.approved_B.gt(0) & (frame.approved_B * frame.n_A < frame.approved_A * frame.n_B)
        if not np.array_equal(frame.criterion_met, expected): raise ValueError("group criterion mismatch")
        common = bool(expected.all())
        if not np.array_equal(frame.common_criterion_met, np.full(2, common)):
            raise ValueError("common criterion mismatch")
        if not np.array_equal(frame.selected, np.full(2, common)):
            raise ValueError("selected candidate flag mismatch")
        if index < count - 1 and expected.all(): raise ValueError("search continued after first success")
    if manifest["status"] == COMPLETE:
        if (selected["status"] != "criterion_met" or not search.iloc[-2:].criterion_met.all()
                or selected["selected_delta"] != core.CANDIDATES[count - 1]
                or manifest["selected_delta"] != selected["selected_delta"]):
            raise ValueError("selected penalty does not meet common criterion")
        verify_freeze(config, root=root)
        p = _plain(pd.read_csv(output["tables"] / "portfolio_results.csv", float_precision="round_trip"))
        g = _plain(pd.read_csv(output["tables"] / "group_results.csv", float_precision="round_trip"))
        parent.validate_result_cells(p, g)
        return {"status": COMPLETE, "selected_delta": selected["selected_delta"], "candidate_count": count,
                "scenarios": 2, "policies": 6, "group_cells": 12, "new_fits": 0,
                "new_computations": 0, "writes": 0, "tables_directory": str(output["tables"])}
    if selected["selected_delta"] is not None or count != len(core.CANDIDATES) or search.criterion_met.groupby(search.candidate_index).all().any():
        raise ValueError("unmet criterion was not a complete exhausted search")
    return {"status": "criterion_unmet", "selected_delta": None, "candidate_count": count,
            "new_computations": 0, "writes": 0, "new_fits": 0}
