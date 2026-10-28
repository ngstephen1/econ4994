"""Step 9A toy fixtures replay declared coefficients; no model is fitted."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import bank_regression as bank
from fair_lending.professor_model import group_blind as gb
from fair_lending.professor_model import revised_scenarios as rs
from fair_lending.professor_model import revised_scenarios_io as io
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


@pytest.fixture
def toy():
    i = np.arange(48)
    observed = pd.DataFrame({"applicant_id": [f"toy_{j:03d}" for j in i],
                             "Y": 30000. + 1000 * i, "D": .2 + (i % 7) / 30,
                             "B": (i % 2).astype(np.int8), "C": np.sin(i)})
    splits = observed[["applicant_id"]].assign(split=np.where(i < 32, "train", "evaluation"))
    models = {}
    for kind, features, coefficients, model_id in (
        ("aware", bank.FEATURES, {"const": .02, "Y": -1e-7, "D": .02, "C": .002, "B": .004}, bank.MODEL_ID),
        ("blind", gb.FEATURES, {"const": .021, "Y": -1e-7, "D": .019, "C": .0021}, gb.MODEL_ID),
    ):
        models[kind] = {"model_id": model_id, "feature_order": list(features),
                        "design_order": ["const", *features], "intercept": True,
                        "coefficients": coefficients, "family": "statsmodels_OLS", "solver": "pinv",
                        "regularization": None, "target": "PN", "target_transform": "none",
                        "prediction_rescaling": "none", "bounding": "clip_0_1_separate_column",
                        "frozen_before_evaluator_truth": True, "training_count": 32,
                        "training_ID_fingerprint": bank.ids_fingerprint(splits.loc[splits.split == "train", "applicant_id"]),
                        "evaluation_ID_fingerprint": bank.ids_fingerprint(splits.loc[splits.split == "evaluation", "applicant_id"]),
                        "training_mean_PN": .017,
                        "units": {term: term + " original units" for term in ["const", *features]}}
    full = observed.merge(splits, on="applicant_id")
    aware = bank.risk_estimates(full, models["aware"])
    blind = gb.risk_estimates(full, models["blind"])
    evaluation = observed.loc[i >= 32].reset_index(drop=True)
    return {"full": observed, "splits": splits, "applicants": evaluation, "models": models,
            "aware": aware.loc[aware.split == "evaluation"].reset_index(drop=True),
            "blind": blind.loc[blind.split == "evaluation"].reset_index(drop=True),
            "all_aware": aware, "all_blind": blind,
            "contracts": make_loan_contracts(evaluation[["applicant_id", "Y"]], T=30, theta=.1, r=.05),
            "budget": rs.EXACT_BUDGET}


def build(toy):
    return rs.build_scenario_risk_estimates(toy["applicants"], toy["aware"], toy["blind"])


def scenario(frame, index):
    return frame.loc[frame.scenario_id == rs.SCENARIOS[index]["scenario_id"]].reset_index(drop=True)


@pytest.fixture
def saved(toy, tmp_path):
    root = tmp_path
    config = io.load_config()
    config.update(population_rows=48, training_rows=32, evaluation_rows=16)
    config_path = root / "configs/professor_model/revised_scenarios.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    frames = {
        io.SYNTHETIC / "applicants_bank_view": toy["full"],
        io.SYNTHETIC / "bank_risk_estimates": toy["all_aware"],
        io.SYNTHETIC / "bank_risk_estimates_group_blind": toy["all_blind"],
        io.PROCESSED / "loan_contracts": toy["contracts"],
    }
    for stem, frame in frames.items():
        (root / stem).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / stem.with_suffix(".csv"), index=False)
        frame.to_parquet(root / stem.with_suffix(".parquet"), index=False)
    toy["splits"].to_csv(root / io.SYNTHETIC / "applicant_splits.csv", index=False)
    write_json(root / io.METRICS / "lending_selection.json", {"selection": {"available_bank_capital": rs.EXACT_BUDGET}})
    private_path = io.SYNTHETIC / "private_sentinel.parquet"
    (root / private_path).write_bytes(b"PRIVATE_SENTINEL_NOT_A_DATAFRAME")
    source = Path("src/fair_lending/professor_model/frozen_toy_source.py")
    (root / source).parent.mkdir(parents=True)
    (root / source).write_text("# Immutable toy source, never executed.\n")
    common = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file() and p != config_path}
    for kind, model_name, coef_name, meta_path, config_name, status in (
        ("aware", "bank_regression_model.json", "bank_regression_coefficients.csv", io.SYNTHETIC / "bank_regression_metadata.json", "bank_regression.yaml", "Step_5_validated"),
        ("blind", "group_blind_regression_model.json", "group_blind_regression_coefficients.csv", io.PROCESSED / "group_blind_comparison_metadata.json", "group_blind_comparison.yaml", "Step_7B_validated"),
    ):
        old_config = {"unchanged_toy_configuration": kind}
        old_path = root / "configs/professor_model" / config_name
        old_path.write_text(yaml.safe_dump(old_config))
        model = deepcopy(toy["models"][kind])
        model["configuration_fingerprint"] = configuration_fingerprint(old_config)
        write_json(root / io.METRICS / model_name, model)
        coef_path = root / io.TABLES / coef_name
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"term": t, "coefficient": model["coefficients"][t], "predictor_units": model["units"][t]} for t in model["design_order"]]).to_csv(coef_path, index=False)
        artifact_paths = [io.METRICS / model_name, io.TABLES / coef_name]
        artifact_paths += [io.SYNTHETIC / ("bank_risk_estimates" + ("_group_blind" if kind == "blind" else "") + ext) for ext in (".csv", ".parquet")]
        artifacts = {str(j): {"path": str(p), "sha256": file_hash(root / p)} for j, p in enumerate(artifact_paths)}
        freeze = {"status": "frozen_before_truth", "configuration_file_sha256": file_hash(old_path), "artifacts": artifacts}
        metadata = {"status": status, "resolved_configuration": old_config,
                    "configuration_fingerprint": model["configuration_fingerprint"],
                    "configuration_file_sha256": file_hash(old_path), "source_code_hashes": {str(source): file_hash(root / source)},
                    "source_code_fingerprint": configuration_fingerprint({str(source): file_hash(root / source)}),
                    "input_checksums": dict(common), "artifacts": artifacts, "freeze_checkpoint": freeze,
                    "exact_saved_budget": rs.EXACT_BUDGET}
        if kind == "blind":
            checkpoint_path = io.METRICS / "group_blind_freeze_checkpoint.json"
            write_json(root / checkpoint_path, freeze)
            metadata["input_checksums"][str(checkpoint_path)] = file_hash(root / checkpoint_path)
        write_json(root / meta_path, metadata)
    return root, config, config_path


def test_config_is_exact():
    config = io.load_config()
    rs.validate_config(config)
    assert config["direct_adjustment_per_B"] == -.1
    assert len(config["scenarios"]) == 3
    assert config["exact_saved_budget_USD"] == 1826087220.7466326


@pytest.mark.parametrize("key,value", [("direct_adjustment_per_B", .1), ("direct_adjustment_per_B", -.001),
                                      ("adjustment_scale", "log_odds"), ("bounding", "sigmoid"),
                                      ("exact_saved_budget_USD", 1826087220.75)])
def test_scientific_config_changes_rejected(key, value):
    config = io.load_config()
    config[key] = value
    with pytest.raises(ValueError): rs.validate_config(config)


def test_professor_branches_and_control_mapping():
    config = io.load_config()
    assert config["scenarios"][0]["base_model_id"] == bank.MODEL_ID
    assert config["scenarios"][1]["base_model_id"] == gb.MODEL_ID
    assert config["scenarios"][2]["role"] == "analytical_control"
    config["scenarios"][0]["base_model_id"] = gb.MODEL_ID
    with pytest.raises(ValueError): rs.validate_config(config)


def test_exact_no_direct_and_blind_frozen_reproduction(toy):
    result = build(toy)
    np.testing.assert_array_equal(scenario(result, 0).estimated_PD_adjusted_raw, toy["aware"].bank_PD_raw)
    np.testing.assert_array_equal(scenario(result, 2).estimated_PD_adjusted_raw, toy["blind"].bank_PD_group_blind_raw)
    np.testing.assert_array_equal(scenario(result, 0).estimated_PD_bounded, toy["aware"].bank_PD_bounded)
    np.testing.assert_array_equal(scenario(result, 2).estimated_PD_bounded, toy["blind"].bank_PD_group_blind_bounded)


def test_direct_B0_unchanged_and_B1_exact_minus_point_one(toy):
    direct, reference = scenario(build(toy), 1), scenario(build(toy), 2)
    mask = direct.B.eq(1)
    np.testing.assert_array_equal(direct.loc[~mask, "estimated_PD_adjusted_raw"], reference.loc[~mask, "estimated_PD_adjusted_raw"])
    np.testing.assert_array_equal(direct.loc[mask, "estimated_PD_adjusted_raw"], reference.loc[mask, "estimated_PD_adjusted_raw"] - .1)
    assert direct.loc[mask, "group_adjustment"].eq(-.1).all()
    assert direct.loc[~mask, "group_adjustment"].eq(0).all()


def test_raw_and_clipping_identities_and_cell_coverage(toy):
    result = build(toy)
    assert rs.validate_scenario_estimates(result, toy["applicants"], toy["aware"], toy["blind"])["valid"]
    assert len(result) == 3 * len(toy["applicants"])
    assert not result.duplicated(["scenario_id", "applicant_id"]).any()
    np.testing.assert_array_equal(result.estimated_PD_adjusted_raw, result.estimated_PD_base_raw + result.group_adjustment)
    np.testing.assert_array_equal(result.estimated_PD_bounded, np.clip(result.estimated_PD_adjusted_raw, 0, 1))
    assert scenario(result, 1).loc[lambda f: f.B == 1, "estimated_PD_adjusted_raw"].lt(0).all()


def test_corrupt_scenario_value_or_missing_cell_is_rejected(toy):
    result = build(toy)
    result.loc[0, "group_adjustment"] = .1
    with pytest.raises(AssertionError): rs.validate_scenario_estimates(result, toy["applicants"], toy["aware"], toy["blind"])
    with pytest.raises(AssertionError): rs.validate_scenario_estimates(build(toy).iloc[1:], toy["applicants"], toy["aware"], toy["blind"])


def test_duplicate_cell_rejected(toy):
    result = build(toy)
    with pytest.raises(ValueError): rs.validate_scenario_estimates(pd.concat([result, result.iloc[:1]]), toy["applicants"], toy["aware"], toy["blind"])


def test_determinism_and_input_row_order_independence(toy):
    reordered = rs.build_scenario_risk_estimates(toy["applicants"].iloc[::-1], toy["aware"].iloc[::-1], toy["blind"].iloc[::-1])
    pd.testing.assert_frame_equal(build(toy), reordered, check_exact=True)
    pd.testing.assert_frame_equal(build(toy), build(toy), check_exact=True)


@pytest.mark.parametrize("column", ["PN", "true_PD", "epsilon", "eta", "approved"])
def test_nonobservable_columns_rejected_at_scenario_boundary(toy, column):
    with pytest.raises(ValueError): rs.build_scenario_risk_estimates(toy["applicants"].assign(**{column: 0}), toy["aware"], toy["blind"])


@pytest.mark.parametrize("change", ["missing_id", "duplicate_id", "train_row", "wrong_model", "wrong_bounded", "wrong_flag"])
def test_bad_reference_inputs_rejected(toy, change):
    frame = toy["blind"].copy()
    if change == "missing_id": frame = frame.iloc[1:]
    elif change == "duplicate_id": frame.loc[0, "applicant_id"] = frame.loc[1, "applicant_id"]
    elif change == "train_row": frame.loc[0, "split"] = "train"
    elif change == "wrong_model": frame.loc[0, "model_id"] = bank.MODEL_ID
    elif change == "wrong_bounded": frame.loc[0, gb.BOUNDED] = .99
    else: frame.loc[0, "prediction_out_of_bounds"] = True
    with pytest.raises(ValueError): rs.build_scenario_risk_estimates(toy["applicants"], toy["aware"], frame)


def test_group_statistics_units_and_adjustment_count(toy):
    result = build(toy)
    overall, groups = rs.summarize_risks(result)
    assert len(overall) == 3 and len(groups) == 6
    direct = groups.loc[(groups.scenario_id == rs.SCENARIOS[1]["scenario_id"]) & groups.B.eq(1)].iloc[0]
    values = scenario(result, 1).loc[lambda f: f.B == 1]
    assert direct.mean_raw_PD_pp == 100 * values.estimated_PD_adjusted_raw.mean()
    assert direct.median_raw_PD_pp == 100 * values.estimated_PD_adjusted_raw.median()
    assert direct.mean_adjustment_pp == -10
    assert direct.adjusted_applicants == len(values)
    assert direct.bounded_zero_share == 1


def test_boundary_zeros_ones_distinct_from_actual_clipping():
    raw = np.array([-.1, 0, .5, 1, 1.2])
    frame = pd.DataFrame({"estimated_PD_base_raw": raw, "group_adjustment": 0., "estimated_PD_adjusted_raw": raw,
                          "estimated_PD_bounded": np.clip(raw, 0, 1), "clipped_below_zero": raw < 0, "clipped_above_one": raw > 1})
    summary = rs._summary(frame)
    assert summary["negative_raw_count"] == summary["clipped_to_zero_count"] == 1
    assert summary["above_one_raw_count"] == summary["clipped_to_one_count"] == 1
    assert summary["bounded_zero_count"] == summary["bounded_one_count"] == 2


def test_zero_tie_capital_math_without_optimization(toy):
    result = build(toy)
    ties = rs.zero_tie_diagnostics(result, toy["contracts"][["applicant_id", "L_i"]], rs.EXACT_BUDGET)
    direct = ties.loc[ties.scenario_id == rs.SCENARIOS[1]["scenario_id"]].iloc[0]
    expected = math.fsum(toy["contracts"].loc[toy["applicants"].B.eq(1), "L_i"])
    assert direct.zero_score_requested_principal_USD == expected
    assert direct.B1_zero_score_count == 8
    assert direct.diagnostic_only and direct.entire_zero_score_group_feasible


def test_zero_tie_exact_budget_and_rounding_edge():
    estimates = pd.DataFrame({"applicant_id": ["x", "y"], "scenario_id": ["toy"] * 2,
                              "B": [0, 1], "estimated_PD_bounded": [0., 0.]})
    principal = pd.DataFrame({"applicant_id": ["x", "y"], "L_i": [1e9, 1e-9]})
    assert math.fsum(principal.L_i) == 1e9  # Display rounding hides the excess.
    assert not rs.zero_tie_diagnostics(estimates, principal, 1e9).iloc[0].entire_zero_score_group_feasible
    principal["L_i"] = [5e8, 5e8]
    assert rs.zero_tie_diagnostics(estimates, principal, 1e9).iloc[0].entire_zero_score_group_feasible


def test_zero_tie_missing_principal_rejected(toy):
    with pytest.raises(ValueError): rs.zero_tie_diagnostics(build(toy), toy["contracts"][["applicant_id", "L_i"]].iloc[1:], rs.EXACT_BUDGET)


def test_preview_is_balanced_deterministic_and_evaluator_only(toy):
    preview = rs.evaluator_preview(toy["applicants"], build(toy), per_group=5)
    assert len(preview) == 30
    assert preview.groupby("B").applicant_id.nunique().to_dict() == {0: 5, 1: 5}
    assert preview.information_scope.eq("EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING").all()
    assert not {"PN", "true_PD", "epsilon", "approved"}.intersection(preview.columns)


def test_frozen_loader_identity_and_information_boundary(saved, monkeypatch):
    root, config, _ = saved
    parquet_read, csv_read = pd.read_parquet, pd.read_csv
    seen = []
    def read_parquet(path, *args, **kwargs):
        seen.append(Path(path).name)
        assert Path(path).name in {"applicants_bank_view.parquet", "bank_risk_estimates.parquet", "bank_risk_estimates_group_blind.parquet", "loan_contracts.parquet"}
        return parquet_read(path, *args, **kwargs)
    def read_csv(path, *args, **kwargs):
        assert not any(s in Path(path).name for s in ("training_signals", "simulation_truth", "private", "outcomes", "expected_profit"))
        return csv_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    def forbidden(*args, **kwargs): raise AssertionError("No fitting or optimization allowed")
    monkeypatch.setattr(bank, "fit_bank_ols", forbidden)
    monkeypatch.setattr(gb, "fit_group_blind", forbidden)
    monkeypatch.setattr(gb, "choose_cutoff", forbidden)
    inputs, hashes = io.load_frozen_inputs(config, root)
    assert len(inputs["applicants"]) == 16 and seen
    assert verify_input_hashes(hashes, root)
    assert len(build(inputs)) == 48


def test_tampered_model_is_not_reused(saved):
    root, config, _ = saved
    p = root / io.METRICS / "group_blind_regression_model.json"
    model = json.loads(p.read_text()); model["coefficients"]["const"] += .01
    write_json(p, model)
    with pytest.raises(ValueError, match="checksum mismatch"): io.load_frozen_inputs(config, root)


def test_fingerprint_or_split_drift_is_rejected(saved):
    root, config, _ = saved
    path = root / io.SYNTHETIC / "bank_regression_metadata.json"
    data = json.loads(path.read_text()); data["source_code_fingerprint"] = "wrong"
    write_json(path, data)
    with pytest.raises(ValueError, match="source fingerprint"): io.load_frozen_inputs(config, root)


def test_output_roundtrip_provenance_preservation_and_overwrite_guard(saved):
    root, config, config_path = saved
    inputs, hashes = io.load_frozen_inputs(config, root)
    report = io.save_artifacts(inputs, build(inputs), config, hashes, root=root, config_path=config_path)
    assert report["valid"] and report["original_files_unchanged"]
    assert report["model_fits"] == report["random_draws"] == report["lending_optimizations"] == 0
    paths = io.output_paths(root)
    assert all(p.is_file() for p in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["input_checksums"] == hashes
    assert metadata["evaluation_ID_fingerprint"] == inputs["evaluation_ID_fingerprint"]
    assert not {"PN", "true_PD", "epsilon", "approved", "expected_profit"}.intersection(pd.read_parquet(paths["risk_parquet"]).columns)
    assert verify_input_hashes(hashes, root)
    before = {str(p): file_hash(p) for p in paths.values()}
    with pytest.raises(FileExistsError): io.save_artifacts(inputs, build(inputs), config, hashes, root=root, config_path=config_path)
    assert before == {str(p): file_hash(p) for p in paths.values()}


def test_changed_original_blocks_export(saved):
    root, config, config_path = saved
    inputs, hashes = io.load_frozen_inputs(config, root)
    (root / io.SYNTHETIC / "applicants_bank_view.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed before export"):
        io.save_artifacts(inputs, build(inputs), config, hashes, root=root, config_path=config_path)
    assert not any(p.exists() for p in io.output_paths(root).values())
