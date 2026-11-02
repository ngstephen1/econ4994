"""Immutable two-world orchestration: simulator, bank freeze, then evaluator.

This is a new experiment family. It never changes canonical population files,
old model artifacts, the historical -0.1 treatment, or existing policies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import PROJECT_ROOT, configuration_fingerprint

NAMESPACE = "nov2_followup"
SCENARIOS = ("scenario_1_no_systemic", "scenario_2_systemic")
METHODS = ("ols_control", "ols_penalty", "ml_histgb")
STATUS = "complete_two_scenario_single_realization"
CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/nov2_followup.yaml"
REFERENCE_PDF = Path("/Users/macbook/Desktop/nov_2_followup_slide_update.pptx.pdf")
BASE = Path("data/synthetic/professor_model/baseline")
FROZEN_CONFIG = {
    "schema_version": "professor-nov2-followup-v1", "population_rows": 100000,
    "training_rows": 80000, "evaluation_rows": 20000, "direct_penalty": .01,
    "epsilon_shift": 1., "signal_scale": .8, "credit_epsilon_coefficient": .4,
    "budget": 1826087220.7466326,
    "scenario_features": {SCENARIOS[0]: ["Y", "D", "C"], SCENARIOS[1]: ["Y", "D", "C", "B"]},
    "inner_split": {"fit_rows": 64000, "validation_rows": 16000, "master_seed": 4994,
                    "spawn_key": [1102, 0], "bit_generator": "PCG64"},
    "ml": {"loss": "squared_error", "learning_rate": .05, "min_samples_leaf": 1000,
           "l2_regularization": 10., "early_stopping": False, "random_state": 4994,
           "max_leaf_nodes": [3, 7], "max_iter": [25, 75], "selection": "validation_PN_MSE",
           "exact_tie_rule": "fewer_leaves_then_fewer_iterations"},
    "policy": {"ordering": "bounded_PD", "probabilities": "unchanged_clip_raw_0_1",
               "full_requests": True, "exact_ties": "indivisible", "reject_all": "separate_action",
               "objective": "perceived_expected_profit", "optimality_check": "independent_Fraction_enumeration"},
    "reporting": {"primary_profit": "total_true_expected_profit_from_funded_loans_by_group",
                  "additional_profit": "true_profit_per_eligible_group_applicant", "noise_realizations": 1},
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, dict): return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_value(v) for v in value]
    if isinstance(value, np.generic): return _json_value(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and np.isnan(value): return None
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(_json_value(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.astype(object).where(frame.notna(), None).to_csv(path, index=False, mode="x")
    restored = pd.read_csv(path, float_precision="round_trip")
    expected = frame.copy()
    for name in expected:
        if expected[name].isna().all(): expected[name] = np.nan
    pd.testing.assert_frame_equal(restored, expected, check_exact=True, check_dtype=False)


def write_pair(stem, frame):
    write_csv(stem.with_suffix(".csv"), frame)
    with stem.with_suffix(".parquet").open("xb") as stream:
        frame.to_parquet(stream, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(stem.with_suffix(".parquet")), frame, check_exact=True)


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    # JSON distinguishes booleans from integer parameters and rejects NaN/inf.
    if configuration_fingerprint(config) != configuration_fingerprint(FROZEN_CONFIG):
        raise ValueError("November specification differs from the predeclared two-world design")
    return config


def paths(root=PROJECT_ROOT):
    return {name: root / value for name, value in {
        "synthetic": f"data/synthetic/professor_model/{NAMESPACE}",
        "processed": f"data/processed/professor_model/{NAMESPACE}",
        "tables": f"results/tables/professor_model/{NAMESPACE}",
        "metrics": f"results/metrics/professor_model/{NAMESPACE}",
        "figures": f"results/figures/professor_model/{NAMESPACE}",
        "models": f"results/models/professor_model/{NAMESPACE}",
    }.items()}


def verify_hashes(hashes, root=PROJECT_ROOT, *, allowed=()):
    for name, expected in hashes.items():
        if name in allowed: continue
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("checksum path must stay within repository")
        p = root / rel
        if not p.is_file() or file_hash(p) != expected:
            raise ValueError(f"checksum mismatch: {name}")


def _dependencies():
    return {"python": platform.python_version(), **{p: version(p) for p in (
        "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "pyarrow", "PyYAML", "joblib", "matplotlib", "threadpoolctl")}}


def source_hashes(root=PROJECT_ROOT):
    names = ("nov2_followup", "nov2_followup_io", "nov2_followup_figures", "bank_regression",
             "group_blind", "risk", "config", "contracts", "profit", "lending", "noise_pilot")
    files = [root / f"src/fair_lending/professor_model/{n}.py" for n in names]
    files += [root / "experiments/run_professor_nov2_followup.py", root / "configs/professor_model/nov2_followup.yaml"]
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def _preservation(root):
    supplied = root / "tmp/nov2_followup/preexisting_hashes.json"
    if supplied.exists():
        record = json.loads(supplied.read_text())
    else:
        files = {root / p for p in subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()}
        for folder in ("data/synthetic/professor_model", "data/processed/professor_model",
                       "results/tables/professor_model", "results/metrics/professor_model",
                       "results/models/professor_model", "results/figures/professor_model"):
            files.update(p for p in (root / folder).rglob("*") if p.is_file() and NAMESPACE not in p.parts)
        record = {"hashes": {str(p.relative_to(root)): file_hash(p) for p in sorted(files) if p.is_file()},
                  "allowed_documentation_edits": ["docs/CURRENT_RESEARCH.md", "docs/PROJECT_STATUS.md"]}
    verify_hashes(record["hashes"], root, allowed=record["allowed_documentation_edits"])
    return record


def _read_pair(stem):
    frame = pd.read_parquet(stem.with_suffix(".parquet"))
    pd.testing.assert_frame_equal(frame, pd.read_csv(stem.with_suffix(".csv"), float_precision="round_trip"), check_exact=True, check_dtype=False)
    return frame


def _registered_inputs(root):
    """Validate saved content identities, never current-HEAD-derived run IDs."""
    files = [root / BASE / f"{n}.{ext}" for n in (
        "applicants_bank_view", "applicants_simulator_private", "training_signals",
        "risk_measurement_noise_private", "simulation_truth", "bank_risk_estimates_group_blind")
        for ext in ("csv", "parquet")]
    files += [root / BASE / "applicant_splits.csv", root / "results/metrics/professor_model/group_blind_regression_model.json",
              root / "results/metrics/professor_model/lending_selection.json"]
    files += [root / f"data/processed/professor_model/loan_contracts.{ext}" for ext in ("csv", "parquet")]
    # These original manifests bind canonical inputs and the frozen model.
    registered = {}
    for rel in (BASE / "population_metadata.json", BASE / "true_risk_metadata.json",
                BASE / "noisy_signal_metadata.json", Path("data/processed/professor_model/group_blind_comparison_metadata.json")):
        p = root / rel; data = json.loads(p.read_text()); files.append(p)
        for key in ("input_checksums", "input_data_checksums", "source_code_hashes"):
            if key in data:
                verify_hashes(data[key], root)
                for name, expected in data[key].items():
                    if name in registered and registered[name] != expected:
                        raise ValueError("canonical manifests disagree on an input identity")
                    registered[name] = expected
        for artifact in data["artifacts"].values():
            verify_hashes({artifact["path"]: artifact["sha256"]}, root)
            registered[artifact["path"]] = artifact["sha256"]
    return {**registered, **{str(p.relative_to(root)): file_hash(p) for p in files}}


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}), flush=True)


def _simulate(config, output, root):
    from .nov2_followup import construct_world
    source = {key: _read_pair(root / BASE / name) for key, name in {
        "bank": "applicants_bank_view", "private": "applicants_simulator_private",
        "targets": "training_signals", "eta": "risk_measurement_noise_private", "truth": "simulation_truth"}.items()}
    if any(len(frame) != config["population_rows"] for frame in source.values()):
        raise ValueError("canonical population row count changed")
    checks = {}
    for scenario in SCENARIOS:
        world = construct_world(**source, scenario_id=scenario)
        folder = output["synthetic"] / scenario
        for name in ("bank", "targets", "private", "truth"):
            write_pair(folder / name, world[name])
        # Saved shared eta is explicit simulator-only evidence, never a feature.
        write_pair(folder / "noise_private", source["eta"])
        checks[scenario] = world["checks"]
        _progress("simulator_world_saved", scenario=scenario)
    return checks


def _read_frozen_banks(output):
    frames = {name: pd.read_parquet(output["processed"] / f"{name}.parquet") for name in ("predictions", "assessments", "decisions")}
    candidates = pd.read_csv(output["tables"] / "cutoff_candidates.csv", float_precision="round_trip")
    selections = json.loads((output["metrics"] / "bank_selections.json").read_text())
    worlds = {}
    for scenario in SCENARIOS:
        worlds[scenario] = {}
        for method in METHODS:
            bundle = {name: frame.loc[(frame.scenario_id == scenario) & (frame.method_id == method)].reset_index(drop=True)
                      for name, frame in frames.items()}
            bundle["candidates"] = candidates.loc[(candidates.scenario_id == scenario) & (candidates.method_id == method)].reset_index(drop=True)
            bundle.update(selections[scenario][method])
            worlds[scenario][method] = bundle
    return worlds


def verify_freeze(config, *, root=PROJECT_ROOT):
    p = paths(root)["metrics"] / "bank_freeze_checkpoint.json"
    frozen = json.loads(p.read_text())
    if frozen["status"] != "all_six_policies_frozen_before_evaluator_load" or frozen["configuration_fingerprint"] != configuration_fingerprint(config):
        raise ValueError("bank stage has not been frozen under this configuration")
    verify_hashes(frozen["artifacts"], root)
    verify_hashes(frozen["input_hashes"], root)
    verify_hashes(frozen["source_hashes"], root)
    if frozen["dependency_versions"] != _dependencies():
        raise ValueError("frozen bank dependencies changed")
    selections = json.loads((paths(root)["metrics"] / "bank_selections.json").read_text())
    if set(selections) != set(SCENARIOS) or any(set(world) != set(METHODS) for world in selections.values()):
        raise ValueError("bank freeze must contain all six declared policy cells")
    for world in selections.values():
        for policy in world.values(): validate_policy_checks(policy["selection"], policy["checks"], config["budget"])
    return frozen


def validate_policy_checks(selection, checks, budget):
    """Separate acceptance flags from legitimate false/zero-valued diagnostics."""
    required = ("exact_threshold_rule", "tied_risks_together", "no_skipped_applicants", "full_loans_only",
                "original_budget_not_exceeded", "all_distinct_candidates_and_no_lending",
                "maximum_perceived_profit_among_feasible_cutoffs", "truth_free_decisions",
                "independent_exact_enumeration_passed")
    if any(checks.get(name) is not True for name in required):
        raise ValueError("a required exact policy validation did not pass")
    if (checks["maximum_budget_violation_USD"] != 0 or checks["selected_budget_residual_USD"] > 0
            or selection["available_bank_capital"] != budget or selection["total_funded_principal"] > budget):
        raise ValueError("policy checks do not certify the original capital constraint")


def run_experiment(config_path=CONFIG_PATH, *, root=PROJECT_ROOT):
    """Execute only the two approved matched worlds; existing output is fatal."""
    from . import nov2_followup as core
    from .nov2_followup_figures import save_figures
    from .profit import assess_true_profit

    config_path = Path(config_path); config = load_config(config_path); output = paths(root)
    occupied = [str(p) for folder in output.values() if folder.exists() for p in folder.rglob("*") if p.is_file()]
    if occupied: raise FileExistsError(f"November outputs already exist; use read-only --summarize-only: {occupied[:3]}")
    originals = _preservation(root); inputs = _registered_inputs(root)
    external = {str(REFERENCE_PDF): file_hash(REFERENCE_PDF)}
    sources = source_hashes(root); dependencies = _dependencies(); started = datetime.now(timezone.utc).isoformat()
    for folder in output.values(): folder.mkdir(parents=True, exist_ok=True)
    write_json(output["metrics"] / "preservation_manifest.json", originals)
    write_json(output["metrics"] / "resolved_configuration.json", config)
    stage = "simulation"
    try:
        _progress(stage)
        simulation_checks = _simulate(config, output, root)
        splits = pd.read_csv(root / BASE / "applicant_splits.csv")
        if splits.split.value_counts().to_dict() != {"train": config["training_rows"], "evaluation": config["evaluation_rows"]}:
            raise ValueError("original applicant-level split changed")
        inner = core.make_inner_splits(splits, **{k: v for k, v in {
            "fit_rows": config["inner_split"]["fit_rows"], "validation_rows": config["inner_split"]["validation_rows"],
            "seed": config["inner_split"]["master_seed"], "spawn_key": tuple(config["inner_split"]["spawn_key"])}.items()})
        write_csv(output["processed"] / "applicant_splits.csv", splits)
        write_csv(output["processed"] / "inner_training_splits.csv", inner)
        contracts = _read_pair(root / "data/processed/professor_model/loan_contracts")
        evaluation_ids = set(splits.loc[splits.split.eq("evaluation"), "applicant_id"])
        if len(contracts) != config["evaluation_rows"] or set(contracts.applicant_id) != evaluation_ids:
            raise ValueError("contracts do not match the original evaluation applicants")
        write_pair(output["processed"] / "loan_contracts", contracts)
        original_selection = json.loads((root / "results/metrics/professor_model/lending_selection.json").read_text())
        if original_selection["selection"]["available_bank_capital"] != config["budget"]:
            raise ValueError("original absolute capital changed")
        frozen_blind = json.loads((root / "results/metrics/professor_model/group_blind_regression_model.json").read_text())
        frozen_predictions = _read_pair(root / BASE / "bank_risk_estimates_group_blind")
        stage = "bank_fitting_and_selection"; all_frames = {name: [] for name in ("predictions", "assessments", "decisions", "candidates")}
        model_metadata, selections, grids = {}, {}, []
        for scenario in SCENARIOS:
            _progress("fit_world", scenario=scenario)
            bank = pd.read_parquet(output["synthetic"] / scenario / "bank.parquet")
            targets = pd.read_parquet(output["synthetic"] / scenario / "targets.parquet")
            fitted = core.fit_scenario_models(bank, targets, splits, inner, scenario,
                frozen_blind_model=frozen_blind, frozen_blind_predictions=frozen_predictions,
                ml_settings=config["ml"])
            model_metadata[scenario] = {"ols_model": fitted["ols_model"], "ml": fitted["model_metadata"]}
            with (output["models"] / f"{scenario}_histgb.joblib").open("xb") as stream:
                joblib.dump(fitted["ml_model"], stream)
            grid = fitted["validation_grid"].copy(); grid["scenario_id"] = scenario; grids.append(grid)
            selections[scenario] = {}
            for method in METHODS:
                pred = fitted["predictions"].loc[fitted["predictions"].method_id.eq(method)].reset_index(drop=True)
                bundle = core.optimize_policy(contracts, pred, config["budget"])
                validate_policy_checks(bundle["selection"], bundle["checks"], config["budget"])
                bundle["predictions"] = pred
                for name in all_frames:
                    frame = bundle[name].copy()
                    if "scenario_id" not in frame: frame.insert(0, "scenario_id", scenario)
                    if "method_id" not in frame: frame.insert(1, "method_id", method)
                    all_frames[name].append(frame)
                selections[scenario][method] = {"selection": bundle["selection"], "checks": bundle["checks"]}
                _progress("policy_selected", scenario=scenario, method=method,
                          approvals=bundle["selection"]["number_approved"])
        for name, pieces in all_frames.items():
            combined = pd.concat(pieces, ignore_index=True)
            sort = ["scenario_id", "method_id", "candidate_id" if name == "candidates" else "applicant_id"]
            combined = combined.sort_values(sort).reset_index(drop=True)
            if name == "candidates": write_csv(output["tables"] / "cutoff_candidates.csv", combined)
            else: write_pair(output["processed"] / name, combined)
        write_csv(output["tables"] / "ml_validation.csv", pd.concat(grids, ignore_index=True))
        write_json(output["metrics"] / "model_metadata.json", model_metadata)
        write_json(output["metrics"] / "bank_selections.json", selections)
        bank_files = [p for folder in (output["processed"], output["models"]) for p in folder.rglob("*") if p.is_file()]
        bank_files += [output["tables"] / "cutoff_candidates.csv", output["tables"] / "ml_validation.csv",
                       output["metrics"] / "model_metadata.json", output["metrics"] / "bank_selections.json",
                       output["metrics"] / "preservation_manifest.json", output["metrics"] / "resolved_configuration.json"]
        for scenario in SCENARIOS:
            # Hashes of simulator-only outputs seal the evaluation reference
            # before bank release without exposing any hidden numeric values
            # to fitting, model selection, assessment or cutoff APIs.
            bank_files += [output["synthetic"] / scenario / f"{name}.{ext}"
                           for name in ("bank", "targets", "private", "truth", "noise_private")
                           for ext in ("csv", "parquet")]
        write_json(output["metrics"] / "bank_freeze_checkpoint.json", {
            "schema_version": config["schema_version"], "status": "all_six_policies_frozen_before_evaluator_load",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "configuration_fingerprint": configuration_fingerprint(config),
            "input_hashes": inputs, "source_hashes": sources, "dependency_versions": dependencies,
            "artifacts": {str(p.relative_to(root)): file_hash(p) for p in bank_files},
            "stage_order": ["simulator_worlds_saved", "six_bank_policies_selected", "bank_artifacts_frozen"],
        })
        verify_freeze(config, root=root)
        stage = "evaluator"; _progress(stage)
        # Release fitted objects and reload the immutable bank artifacts before
        # this stage reads evaluator-only truth. No true values select policies.
        frozen_worlds = _read_frozen_banks(output)
        evaluated = {name: [] for name in ("portfolio", "groups", "risk_diagnostics", "penalty_comparisons")}
        audits = []
        for scenario in SCENARIOS:
            bank = pd.read_parquet(output["synthetic"] / scenario / "bank.parquet")
            truth = pd.read_parquet(output["synthetic"] / scenario / "truth.parquet")
            targets = pd.read_parquet(output["synthetic"] / scenario / "targets.parquet")
            truth_eval = contracts[["applicant_id"]].merge(truth[["applicant_id", "true_PD"]], on="applicant_id", validate="one_to_one")
            true_profit = assess_true_profit(contracts, truth_eval)
            write_pair(output["processed"] / f"{scenario}_true_profit_EVALUATOR_ONLY", true_profit)
            audit = bank.loc[bank.applicant_id.isin(evaluation_ids), ["applicant_id", "B"]]
            pn = targets.loc[targets.applicant_id.isin(evaluation_ids)]
            result = core.evaluate_policies(frozen_worlds[scenario], contracts, audit, true_profit, targets=pn)
            for name in evaluated: evaluated[name].append(result[name])
            private = pd.read_parquet(output["synthetic"] / scenario / "private.parquet")
            joined = bank.merge(private, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one")
            for group, frame in joined.groupby("B"):
                audits.append({"scenario_id": scenario, "B": int(group), "n_population": len(frame),
                               "mean_epsilon": float(frame.epsilon.mean()), "mean_C": float(frame.C.mean()),
                               "mean_P_other": float(frame.P_other.mean()), "mean_true_PD": float(frame.true_PD.mean())})
        destinations = {"portfolio": "portfolio_results", "groups": "group_results",
                        "risk_diagnostics": "risk_diagnostics", "penalty_comparisons": "penalty_comparisons"}
        merged = {name: pd.concat(parts, ignore_index=True) for name, parts in evaluated.items()}
        if len(merged["portfolio"]) != 6 or len(merged["groups"]) != 12:
            raise ValueError("incomplete two-world/three-method output")
        validate_result_cells(merged["portfolio"], merged["groups"])
        for name, stem in destinations.items(): write_csv(output["tables"] / f"{stem}.csv", merged[name])
        write_csv(output["tables"] / "systemic_risk_audit.csv", pd.DataFrame(audits))
        stage = "figures"; save_figures(merged["groups"], output["figures"])
        verify_freeze(config, root=root)
        verify_hashes(originals["hashes"], root, allowed=originals["allowed_documentation_edits"])
        for name, expected in external.items():
            if file_hash(Path(name)) != expected: raise ValueError("reference PDF changed")
        validation = {"status": "validated", "simulation_checks": simulation_checks,
                      "scenario_count": 2, "portfolio_rows": 6, "group_rows": 12,
                      "all_six_exact_cutoffs_verified": True, "bank_freeze_verified": True,
                      "prior_files_in_inventory": len(originals["hashes"]),
                      "prior_files_verified_unchanged": len(set(originals["hashes"]) - set(originals["allowed_documentation_edits"])),
                      "allowed_documentation_edits": originals["allowed_documentation_edits"],
                      "single_noise_realization": True, "new_noise_draws": 0,
                      "desired_outcome_signs_enforced": False}
        write_json(output["metrics"] / "validation.json", validation)
        artifacts = [p for folder in output.values() for p in folder.rglob("*") if p.is_file()]
        manifest = {"schema_version": config["schema_version"], "status": STATUS,
                    "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
                    "configuration_file_sha256": file_hash(config_path), "input_hashes": inputs,
                    "external_inputs": external, "source_hashes": sources, "dependency_versions": dependencies,
                    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
                    "stage_order": ["simulator_worlds_saved", "six_bank_policies_selected", "bank_artifacts_frozen", "evaluator_truth_loaded", "figures_written", "preservation_verified"],
                    "artifacts": {str(p.relative_to(root)): file_hash(p) for p in artifacts},
                    "tables": {name: str((output["tables"] / f"{stem}.csv").relative_to(root)) for name, stem in destinations.items()},
                    "cases": 2, "policies": 6, "group_cells": 12}
        write_json(output["metrics"] / "run_manifest.json", manifest)
        _progress("complete", scenarios=2, policies=6)
        return summarize_only(config_path, root=root)
    except Exception as exc:
        failure = output["metrics"] / "failure.json"
        if not failure.exists():
            write_json(failure, {"status": "failed", "stage": stage, "error_type": type(exc).__name__,
                                 "message": str(exc), "traceback": traceback.format_exc(),
                                 "configuration_fingerprint": configuration_fingerprint(config),
                                 "source_hashes": sources, "started_at_utc": started})
        raise


def summarize_only(config_path=CONFIG_PATH, *, root=PROJECT_ROOT):
    """Read/check saved outputs only. Never creates directories or loads models."""
    config = load_config(config_path)
    output = paths(root); path = output["metrics"] / "run_manifest.json"
    if not path.is_file(): raise FileNotFoundError("No complete November experiment exists; summary-only starts no work")
    manifest = json.loads(path.read_text())
    if manifest["status"] != STATUS or manifest["configuration_fingerprint"] != configuration_fingerprint(config) or manifest["configuration_file_sha256"] != file_hash(Path(config_path)):
        raise ValueError("incomplete or incompatible saved experiment")
    if manifest["source_hashes"] != source_hashes(root) or manifest["dependency_versions"] != _dependencies():
        raise ValueError("saved scientific source or dependency identity differs")
    for key in ("source_hashes", "input_hashes", "artifacts"): verify_hashes(manifest[key], root)
    for name, expected in manifest["external_inputs"].items():
        if file_hash(Path(name)) != expected: raise ValueError("reference PDF changed")
    verify_freeze(config, root=root)
    preservation = json.loads((output["metrics"] / "preservation_manifest.json").read_text())
    verify_hashes(preservation["hashes"], root, allowed=preservation["allowed_documentation_edits"])
    portfolios = pd.read_csv(output["tables"] / "portfolio_results.csv", float_precision="round_trip")
    groups = pd.read_csv(output["tables"] / "group_results.csv", float_precision="round_trip")
    validate_result_cells(portfolios, groups)
    return {"status": STATUS, "scenarios": 2, "policies": 6, "group_cells": 12,
            "new_computations": 0, "writes": 0, "tables_directory": str(output["tables"]),
            "manifest": str(path)}


def validate_result_cells(portfolios, groups):
    """Require exact declared cells, denominators and additive group totals."""
    import math
    expected_p = {(s, m) for s in SCENARIOS for m in METHODS}
    expected_g = {(s, m, b) for s, m in expected_p for b in (0, 1)}
    if (len(portfolios) != 6 or len(groups) != 12
            or portfolios.duplicated(["scenario_id", "method_id"]).any()
            or groups.duplicated(["scenario_id", "method_id", "B"]).any()
            or set(portfolios[["scenario_id", "method_id"]].itertuples(index=False, name=None)) != expected_p
            or set(groups[["scenario_id", "method_id", "B"]].itertuples(index=False, name=None)) != expected_g):
        raise ValueError("saved summary has incomplete, unknown or duplicate cells")
    for _, row in portfolios.iterrows():
        cell = groups.loc[groups.scenario_id.eq(row.scenario_id) & groups.method_id.eq(row.method_id)]
        for name in ("n_applicants", "number_approved"):
            if int(cell[name].sum()) != int(row[name]): raise ValueError("group counts do not aggregate")
        for name in ("true_expected_portfolio_profit_USD", "funded_principal_USD"):
            if not math.isclose(math.fsum(cell[name]), float(row[name]), rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError("group economic values do not aggregate")
        for _, group in cell.iterrows():
            if group.n_applicants <= 0 or not np.isfinite(group.approval_rate) or group.approval_rate != group.number_approved / group.n_applicants:
                raise ValueError("invalid within-group approval denominator")
            if not math.isclose(float(group.true_profit_per_eligible_applicant_USD),
                                group.true_expected_portfolio_profit_USD / group.n_applicants,
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("invalid per-eligible profit denominator")
