"""Deterministic structural and predeclared distribution checks for Step 2."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model import (
    generate_applicants, load_population_config, save_population, validate_population,
)
from fair_lending.professor_model.config import (
    BANK_COLUMNS, DEFAULT_CONFIG_PATH, PREVIEW_COLUMNS, PREVIEW_SCOPE,
    PRIVATE_COLUMNS, STREAM_NAMES, PopulationConfigurationError,
    configuration_fingerprint, resolve_config, validate_config,
)
from fair_lending.professor_model.population import evaluator_preview


@pytest.fixture
def config():
    return load_population_config()


@pytest.fixture
def population(config):
    return generate_applicants(config, n_samples=1000)


def test_configuration_matches_supplied_distributions(config):
    assert config["simulation"] == {"n_samples": 100000, "random_seed": 4994, "audit_preview_rows": 150}
    assert config["population"]["independent_primitives"] is True
    for name, mean, sd in (("log_Y", 11, .7), ("D", .35, .16), ("epsilon", 0, 1), ("omega", 0, .2)):
        assert config["population"][name] == {"distribution": "normal", "mean": mean, "sd": sd}
    assert config["population"]["B"] == {"distribution": "bernoulli", "probability": .15}
    assert config["credit_signal"]["epsilon_coefficient"] == .4


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "n_samples", 0), ("simulation", "n_samples", 1.5),
    ("simulation", "n_samples", True), ("simulation", "random_seed", -1),
    ("simulation", "audit_preview_rows", 201),
    ("population", "independent_primitives", False),
    ("credit_signal", "epsilon_coefficient", float("nan")),
])
def test_invalid_settings_are_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(PopulationConfigurationError):
        generate_applicants(config)


@pytest.mark.parametrize("name,key,value", [
    ("log_Y", "sd", 0), ("D", "sd", -.16),
    ("omega", "mean", float("inf")), ("D", "distribution", "truncated_normal"),
    ("B", "probability", 1.01), ("B", "distribution", "forced_exact_counts"),
])
def test_invalid_distribution_parameters_are_rejected(config, name, key, value):
    config["population"][name][key] = value
    with pytest.raises(PopulationConfigurationError):
        validate_config(config)


def test_outputs_cannot_target_version_two(config):
    config["outputs"]["data_directory"] = "data/synthetic/economic_lending/v2_baseline"
    with pytest.raises(PopulationConfigurationError, match="professor_model"):
        validate_config(config)


def test_loaded_configuration_is_a_mapping(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a_mapping\n")
    with pytest.raises(PopulationConfigurationError, match="mapping"):
        load_population_config(invalid)


def test_seed_and_row_overrides_do_not_mutate_config(config):
    original = copy.deepcopy(config)
    resolved = resolve_config(config, n_samples=31, seed=17)
    assert config == original
    assert resolved["simulation"]["n_samples"] == 31
    assert resolved["simulation"]["random_seed"] == 17


def test_schema_and_unique_ids_are_exact(population):
    bank, private, _ = population
    assert tuple(bank.columns) == BANK_COLUMNS
    assert tuple(private.columns) == PRIVATE_COLUMNS
    assert len(bank) == len(private) == 1000
    assert bank.applicant_id.nunique() == private.applicant_id.nunique() == 1000
    pd.testing.assert_series_equal(bank.applicant_id, private.applicant_id)
    assert set(bank.B.unique()) == {0, 1}


def test_hidden_and_future_information_is_excluded_from_bank_view(population):
    bank, private, _ = population
    assert {"epsilon", "omega", "log_Y", "true_PD", "PN", "eta", "default_period", "approved", "bank_PD"}.isdisjoint(bank.columns)
    assert {"epsilon", "omega", "log_Y"}.issubset(private.columns)


def test_positive_income_and_exact_log_exponential_relationship(population):
    bank, private, _ = population
    assert (bank.Y > 0).all()
    np.testing.assert_allclose(bank.Y, np.exp(private.log_Y), rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.log(bank.Y), private.log_Y, rtol=1e-14, atol=1e-14)


def test_credit_signal_equation_and_original_scale(population):
    bank, private, _ = population
    np.testing.assert_allclose(bank.C, .4 * private.epsilon + private.omega, rtol=1e-14, atol=1e-14)
    assert (bank.C < 0).any()
    assert (bank.C > 0).any()


def test_all_numeric_values_are_finite_and_no_values_are_missing(population):
    for frame in population[:2]:
        assert not frame.isna().any().any()
        assert np.isfinite(frame.drop(columns="applicant_id").to_numpy()).all()


def test_same_seed_reproduces_every_generated_value_and_metadata(config):
    first = generate_applicants(config, n_samples=1000)
    second = generate_applicants(config, n_samples=1000)
    for left, right in zip(first[:2], second[:2], strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert first[2] == second[2]


def test_different_seed_changes_all_primitive_draws(config):
    a_bank, a_private, _ = generate_applicants(config, n_samples=1000, seed=4994)
    b_bank, b_private, _ = generate_applicants(config, n_samples=1000, seed=4995)
    for variable in ("Y", "D", "B", "C"):
        assert not np.array_equal(a_bank[variable], b_bank[variable])
    for variable in ("log_Y", "epsilon", "omega"):
        assert not np.array_equal(a_private[variable], b_private[variable])


def test_changing_group_share_leaves_all_other_draws_identical(config):
    all_zero = copy.deepcopy(config)
    all_one = copy.deepcopy(config)
    all_zero["population"]["B"]["probability"] = 0
    all_one["population"]["B"]["probability"] = 1
    a_bank, a_private, _ = generate_applicants(all_zero, n_samples=1000)
    b_bank, b_private, _ = generate_applicants(all_one, n_samples=1000)
    assert a_bank.B.eq(0).all() and b_bank.B.eq(1).all()
    pd.testing.assert_frame_equal(a_bank.drop(columns="B"), b_bank.drop(columns="B"), check_exact=True)
    pd.testing.assert_frame_equal(a_private, b_private, check_exact=True)


def test_changing_income_parameters_does_not_change_other_streams(config):
    changed = copy.deepcopy(config)
    changed["population"]["log_Y"]["mean"] = 12
    a_bank, a_private, _ = generate_applicants(config, n_samples=1000)
    b_bank, b_private, _ = generate_applicants(changed, n_samples=1000)
    assert not np.array_equal(a_bank.Y, b_bank.Y)
    pd.testing.assert_frame_equal(a_bank[["D", "B", "C"]], b_bank[["D", "B", "C"]], check_exact=True)
    pd.testing.assert_frame_equal(a_private[["epsilon", "omega"]], b_private[["epsilon", "omega"]], check_exact=True)


def test_group_draws_are_bernoulli_not_forced_counts(population):
    bank, _, metadata = population
    child = np.random.SeedSequence(metadata["random_seed"]).spawn(5)[STREAM_NAMES.index("B")]
    expected = np.random.default_rng(child).binomial(1, .15, size=len(bank))
    np.testing.assert_array_equal(bank.B, expected)


def test_distribution_moments_and_signal_correlations(config):
    # Declared before generating the production sample. These broad tolerances
    # test SD interpretation and independence without exact sample-mean claims.
    bank, private, metadata = generate_applicants(config, n_samples=50000)
    for frame, variable, mean, sd, tolerance in (
        (private, "log_Y", 11, .7, .02), (bank, "D", .35, .16, .006),
        (private, "epsilon", 0, 1, .025), (private, "omega", 0, .2, .006),
    ):
        assert abs(frame[variable].mean() - mean) < tolerance
        assert abs(frame[variable].std() - sd) < tolerance
    assert abs(bank.B.mean() - .15) < .008
    assert abs(bank.C.std() - np.sqrt(.2)) < .015
    assert abs(bank.C.corr(private.epsilon) - 2 / np.sqrt(5)) < .015
    assert abs(bank.C.corr(private.omega) - 1 / np.sqrt(5)) < .015
    report = validate_population(bank, private, metadata["resolved_configuration"])
    assert report["independence_diagnostics"]["max_absolute_primitive_correlation"] < .03


def test_dti_tails_are_retained_and_warning_is_not_validation_failure(config):
    bank, private, metadata = generate_applicants(config, n_samples=10000)
    # Verify the exact unbounded draws, not just a plausibly shaped distribution.
    child = np.random.SeedSequence(4994).spawn(5)[STREAM_NAMES.index("D")]
    np.testing.assert_array_equal(bank.D, np.random.default_rng(child).normal(.35, .16, size=10000))
    assert (bank.D < 0).any()
    report = validate_population(bank, private, metadata["resolved_configuration"])
    assert report["valid"] and not report["errors"]
    assert report["warnings"]
    assert not report["D"]["clipping_applied"]
    assert report["D"]["count_below_zero"] == (bank.D < 0).sum()
    assert report["D"]["count_above_one"] == (bank.D > 1).sum()


@pytest.mark.parametrize("mutation", ["missing", "infinite", "duplicate", "mismatched_ids", "leaked_column", "wrong_signal", "wrong_income", "wrong_group", "wrong_count"])
def test_validation_rejects_structural_failures(population, mutation):
    bank, private, metadata = population
    if mutation == "missing":
        private.loc[0, "epsilon"] = np.nan
    elif mutation == "infinite":
        bank.loc[0, "Y"] = np.inf
    elif mutation == "duplicate":
        bank.loc[0, "applicant_id"] = bank.loc[1, "applicant_id"]
    elif mutation == "mismatched_ids":
        private.loc[0, "applicant_id"] = "unmatched"
    elif mutation == "leaked_column":
        bank["epsilon"] = private.epsilon
    elif mutation == "wrong_signal":
        bank.loc[0, "C"] += .1
    elif mutation == "wrong_income":
        bank.loc[0, "Y"] = 0
    elif mutation == "wrong_group":
        bank.loc[0, "B"] = 2
    else:
        bank = bank.iloc[:-1]
    report = validate_population(bank, private, metadata["resolved_configuration"])
    assert not report["valid"] and report["errors"]


def test_group_audit_matches_independent_calculation(population):
    bank, private, metadata = population
    report = validate_population(bank, private, metadata["resolved_configuration"])
    joined = bank.merge(private, on="applicant_id", validate="one_to_one")
    for item in report["group_comparison"]:
        subset = joined.loc[joined.B == item["B"]]
        assert item["n"] == len(subset)
        assert item["share"] == len(subset) / len(bank)
        for variable in ("Y", "D", "epsilon", "C"):
            assert item[f"mean_{variable}"] == pytest.approx(subset[variable].mean())
    for variable in ("Y", "D", "epsilon", "C"):
        assert report["group_1_minus_0_mean_differences"][variable] == pytest.approx(joined.loc[joined.B == 1, variable].mean() - joined.loc[joined.B == 0, variable].mean())


def test_preview_is_small_and_explicitly_evaluator_only(population):
    bank, private, _ = population
    preview = evaluator_preview(bank, private)
    assert len(preview) == 150
    assert tuple(preview.columns) == PREVIEW_COLUMNS
    assert preview.information_scope.eq(PREVIEW_SCOPE).all()
    assert set(bank.columns) | set(private.columns) <= set(preview.columns)


def test_config_fingerprint_is_order_independent_and_sensitive(config):
    assert configuration_fingerprint(config) == configuration_fingerprint(dict(reversed(list(config.items()))))
    assert configuration_fingerprint(config) != configuration_fingerprint(resolve_config(config, seed=4995))


def test_exports_roundtrip_and_manifest_records_source_and_hashes(population, tmp_path):
    bank, private, metadata = population
    paths, report = save_population(bank, private, metadata, output_root=tmp_path)
    assert report["valid"]
    for name, original in (("bank", bank), ("private", private)):
        pd.testing.assert_frame_equal(pd.read_parquet(paths[f"{name}_parquet"]), original, check_exact=True)
        csv = pd.read_csv(paths[f"{name}_csv"], float_precision="round_trip")
        pd.testing.assert_frame_equal(csv, original, check_dtype=False, check_exact=True)
    manifest = json.loads(paths["metadata"].read_text())
    assert manifest["resolved_configuration"] == metadata["resolved_configuration"]
    assert manifest["configuration_sha256"] == configuration_fingerprint(metadata["resolved_configuration"])
    assert len(manifest["random_streams"]) == 5
    assert manifest["source_sha256"]["configs/professor_model/baseline.yaml"] == hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
    assert "numpy" in manifest["dependencies"]
    for key, artifact in manifest["artifacts"].items():
        assert artifact["sha256"] == hashlib.sha256(paths[key].read_bytes()).hexdigest()
    assert manifest["artifacts"]["audit_preview_csv"]["information_scope"] == "EVALUATOR_ONLY"


def test_existing_exports_require_explicit_overwrite(population, tmp_path):
    bank, private, metadata = population
    paths, _ = save_population(bank, private, metadata, output_root=tmp_path)
    before = paths["bank_csv"].read_bytes()
    with pytest.raises(FileExistsError):
        save_population(bank, private, metadata, output_root=tmp_path)
    assert paths["bank_csv"].read_bytes() == before
    save_population(bank, private, metadata, output_root=tmp_path, overwrite=True)


def test_failed_validation_writes_no_outputs(population, tmp_path):
    bank, private, metadata = population
    bank.loc[0, "Y"] = -1
    with pytest.raises(ValueError, match="validation failed"):
        save_population(bank, private, metadata, output_root=tmp_path)
    assert not list(tmp_path.rglob("*.csv"))


def test_reproducibility_failure_prevents_export(population, tmp_path):
    bank, private, metadata = population
    with pytest.raises(ValueError, match="reproducibility"):
        save_population(bank, private, metadata, output_root=tmp_path, reproducibility_checks={"same_seed_identical": False})
    assert not list(tmp_path.rglob("*.csv"))
