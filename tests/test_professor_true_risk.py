"""Hand-checked risk equations, hidden information, and preserved input data."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model.config import load_population_config, resolve_config, configuration_fingerprint
from fair_lending.professor_model.risk import (
    FROZEN_EQUATION, TRUTH_COLUMNS, TRUTH_PREVIEW_COLUMNS, TRUE_RISK_INPUTS,
    compute_true_risk, load_true_risk_config, validate_true_risk_config,
)
from fair_lending.professor_model.risk_io import (
    file_hash, load_existing_applicants, save_true_risk_artifacts, verify_input_hashes,
)
from fair_lending.professor_model.risk_validation import validate_true_risk


@pytest.fixture
def applicants():
    # Explicit examples, not a regenerated random population.
    log_income = np.array([11., 10., 12., 11., 11., 11.])
    epsilon = np.array([0., 0., 0., -1., 1., 0.])
    omega = np.array([.05, -.02, .01, .02, .03, -.01])
    return pd.DataFrame({
        "applicant_id": [f"example_{i}" for i in range(6)],
        "Y": np.exp(log_income), "log_Y": log_income,
        "D": [.35, .35, .35, .35, .35, .65],
        "B": [0, 1, 0, 1, 0, 1], "epsilon": epsilon,
        "omega": omega, "C": .4 * epsilon + omega,
    })


@pytest.fixture
def config():
    result = load_true_risk_config()
    result["expected_input_rows"] = 6
    return result


def _save_existing_fixture(applicants, config, root):
    """Persist manual Step 2-shaped inputs; never call a population generator."""
    directory = root / config["input_directory"]
    directory.mkdir(parents=True)
    bank = applicants.loc[:, ["applicant_id", "Y", "D", "B", "C"]]
    private = applicants.loc[:, ["applicant_id", "log_Y", "epsilon", "omega"]]
    paths = []
    for name, frame in (("applicants_bank_view", bank), ("applicants_simulator_private", private)):
        for suffix in ("csv", "parquet"):
            path = directory / f"{name}.{suffix}"
            if suffix == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False)
            paths.append(path)
    population_config = resolve_config(load_population_config(), n_samples=len(applicants))
    metadata = {
        "resolved_configuration": population_config,
        "configuration_sha256": configuration_fingerprint(population_config),
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": file_hash(path)} for path in paths},
    }
    path = directory / "population_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_frozen_config_contains_exact_equation():
    config = load_true_risk_config()
    assert config["equation"] == FROZEN_EQUATION
    assert config["expected_input_rows"] == 100000
    assert config["preview_rows"] == 150
    assert TRUE_RISK_INPUTS == ("log_Y", "D", "epsilon")
    assert not {"B", "C", "Y", "omega"}.intersection(TRUE_RISK_INPUTS)


@pytest.mark.parametrize("coefficient", tuple(FROZEN_EQUATION))
def test_coefficient_retuning_is_rejected(config, coefficient):
    config["equation"][coefficient] += .01
    with pytest.raises(ValueError, match="frozen equation"):
        compute_true_risk(pd.DataFrame(), config)


def test_additional_group_term_is_rejected(config):
    config["equation"]["group_coefficient"] = -.1
    with pytest.raises(ValueError, match="frozen equation"):
        validate_true_risk_config(config)


def test_exact_sigmoid_formulas_on_manual_examples(applicants, config):
    truth = compute_true_risk(applicants, config)
    for index, row in applicants.iterrows():
        # Independent scalar arithmetic, not a call to the implementation helper.
        income = 1 / (1 + math.exp(-(.52 - .65 * row.log_Y)))
        other = 1 / (1 + math.exp(-(4 + .06 * row.D + row.epsilon)))
        assert truth.loc[index, "P_income"] == pytest.approx(income, rel=1e-14)
        assert truth.loc[index, "P_other"] == pytest.approx(other, rel=1e-14)
        assert truth.loc[index, "true_PD"] == pytest.approx(income * other, rel=1e-14)
    assert truth.loc[0, "true_PD"] == pytest.approx(.0012951932354248225, rel=1e-12)


def test_output_schema_product_and_strict_probability_bounds(applicants, config):
    truth = compute_true_risk(applicants, config)
    assert tuple(truth.columns) == TRUTH_COLUMNS
    assert len(truth) == 6 and truth.applicant_id.is_unique
    np.testing.assert_array_equal(truth.applicant_id, applicants.applicant_id)
    for variable in TRUTH_COLUMNS[1:]:
        assert truth[variable].between(0, 1, inclusive="neither").all()
    np.testing.assert_allclose(truth.true_PD, truth.P_income * truth.P_other, rtol=1e-14, atol=0)
    assert not truth.isna().any().any()
    assert np.isfinite(truth.iloc[:, 1:]).all().all()


def test_recalculation_is_bitwise_identical(applicants, config):
    pd.testing.assert_frame_equal(compute_true_risk(applicants, config), compute_true_risk(applicants, config), check_exact=True)


@pytest.mark.parametrize("variable", ["B", "C", "omega", "Y"])
def test_excluded_variables_do_not_affect_true_risk(applicants, config, variable):
    baseline = compute_true_risk(applicants, config)
    changed = applicants.copy(deep=True)
    changed[variable] = 1 - changed[variable] if variable == "B" else changed[variable] + 123
    pd.testing.assert_frame_equal(compute_true_risk(changed, config), baseline, check_exact=True)


def test_input_frame_unchanged_after_calculation_and_validation(applicants, config):
    original = applicants.copy(deep=True)
    truth = compute_true_risk(applicants, config)
    assert validate_true_risk(truth, applicants, config)["valid"]
    pd.testing.assert_frame_equal(applicants, original, check_exact=True)


@pytest.mark.parametrize("variable,delta,expected_sign", [("log_Y", .1, -1), ("D", .1, 1), ("epsilon", .1, 1)])
def test_fixed_feature_directions(applicants, config, variable, delta, expected_sign):
    original = compute_true_risk(applicants, config).true_PD
    changed = applicants.copy(deep=True)
    changed[variable] += delta
    difference = compute_true_risk(changed, config).true_PD - original
    assert (expected_sign * difference > 0).all()


def test_stable_sigmoid_does_not_overflow_or_apply_probability_floor(applicants, config):
    extreme = applicants.copy(deep=True)
    extreme["epsilon"] = [-1000., 1000., -1000., 1000., 0., 0.]
    with np.errstate(over="raise"):
        truth = compute_true_risk(extreme, config)
    assert np.isfinite(truth.iloc[:, 1:]).all().all()
    assert truth.P_other.iloc[0] == 0 and truth.true_PD.iloc[0] == 0
    assert truth.P_other.iloc[1] == 1
    assert not validate_true_risk(truth, extreme, config)["valid"]


@pytest.mark.parametrize("mutation", ["missing_input", "infinite_input", "duplicate_input"])
def test_bad_inputs_are_rejected(applicants, config, mutation):
    if mutation == "missing_input":
        applicants.loc[0, "epsilon"] = np.nan
    elif mutation == "infinite_input":
        applicants.loc[0, "log_Y"] = np.inf
    else:
        applicants.loc[0, "applicant_id"] = applicants.loc[1, "applicant_id"]
    with pytest.raises(ValueError):
        compute_true_risk(applicants, config)


@pytest.mark.parametrize("mutation", ["extra_truth_column", "missing_truth", "infinite_truth", "wrong_product", "wrong_income_factor", "wrong_ids", "wrong_count"])
def test_truth_validation_rejects_bad_data(applicants, config, mutation):
    truth = compute_true_risk(applicants, config)
    if mutation == "extra_truth_column":
        truth["B"] = applicants.B
    elif mutation == "missing_truth":
        truth.loc[0, "P_income"] = np.nan
    elif mutation == "infinite_truth":
        truth.loc[0, "true_PD"] = np.inf
    elif mutation == "wrong_product":
        truth.loc[0, "true_PD"] += .001
    elif mutation == "wrong_income_factor":
        truth.loc[0, "P_income"] += .001
        truth["true_PD"] = truth.P_income * truth.P_other
    elif mutation == "wrong_ids":
        truth.loc[0, "applicant_id"] = "other"
    else:
        truth = truth.iloc[:-1]
    report = validate_true_risk(truth, applicants, config)
    assert not report["valid"] and report["errors"]


def test_validation_aligns_by_id_and_uses_sample_sd(applicants, config):
    truth = compute_true_risk(applicants, config).iloc[::-1].reset_index(drop=True)
    report = validate_true_risk(truth, applicants, config)
    assert report["valid"]
    assert report["distributions"]["true_PD"]["sd"] == pytest.approx(truth.true_PD.std(ddof=1))
    assert sum(item["count"] for item in report["risk_bins"]) == len(truth)
    assert sum(item["fraction"] for item in report["risk_bins"]) == pytest.approx(1)
    for item in report["group_comparison"]:
        rows = truth.set_index("applicant_id").loc[applicants.loc[applicants.B == item["B"], "applicant_id"]]
        assert item["n"] == len(rows)
        assert item["mean_true_PD"] == pytest.approx(rows.true_PD.mean())
        assert item["p95_true_PD"] == pytest.approx(rows.true_PD.quantile(.95))


def test_risk_export_preserves_original_files_and_hides_truth(applicants, config, tmp_path):
    directory = _save_existing_fixture(applicants, config, tmp_path)
    joined, hashes = load_existing_applicants(config, tmp_path)
    truth = compute_true_risk(joined, config)
    report = validate_true_risk(truth, joined, config)
    paths = save_true_risk_artifacts(joined, truth, report, config, hashes, root=tmp_path)
    assert verify_input_hashes(hashes, tmp_path)
    bank = pd.read_parquet(directory / "applicants_bank_view.parquet")
    assert list(bank.columns) == ["applicant_id", "Y", "D", "B", "C"]
    assert not {"epsilon", "P_income", "P_other", "true_PD"}.intersection(bank.columns)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["truth_parquet"]), truth, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(paths["truth_csv"], float_precision="round_trip"), truth, check_exact=True)
    preview = pd.read_csv(paths["preview"])
    assert tuple(preview.columns) == TRUTH_PREVIEW_COLUMNS
    assert "EVALUATOR_ONLY" in paths["preview"].name
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["information_scope"] == "EVALUATOR_ONLY"
    assert metadata["input_data_checksums"] == hashes
    assert metadata["configuration_fingerprint"] == configuration_fingerprint(config)
    assert metadata["source_code_fingerprint"] == configuration_fingerprint(metadata["source_code_hashes"])
    assert metadata["random_draws"] is False
    for name, artifact in metadata["artifacts"].items():
        assert artifact["sha256"] == hashlib.sha256(paths[name].read_bytes()).hexdigest()
    assert len(list((tmp_path / config["outputs"]["figures_directory"]).glob("*.png"))) == 3


def test_checksum_mismatch_is_rejected_without_regeneration(applicants, config, tmp_path):
    directory = _save_existing_fixture(applicants, config, tmp_path)
    (directory / "applicants_bank_view.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_existing_applicants(config, tmp_path)


def test_changed_original_input_blocks_export(applicants, config, tmp_path):
    directory = _save_existing_fixture(applicants, config, tmp_path)
    joined, hashes = load_existing_applicants(config, tmp_path)
    truth = compute_true_risk(joined, config)
    report = validate_true_risk(truth, joined, config)
    (directory / "applicants_bank_view.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        save_true_risk_artifacts(joined, truth, report, config, hashes, root=tmp_path)
    assert not (directory / "simulation_truth.csv").exists()
