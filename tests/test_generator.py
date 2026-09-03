"""Behavioral and structural tests for the version-1 synthetic generator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import logit

from fair_lending.simulation.approval import counterfactual_direct_probabilities
from fair_lending.simulation.calibration import solve_intercept
from fair_lending.simulation.config import (
    ConfigurationError,
    resolve_simulation_config,
    stable_fingerprint,
)
from fair_lending.simulation.generator import (
    OUTPUT_COLUMNS,
    generate_synthetic_data,
    save_synthetic_dataset,
)
from fair_lending.simulation.validation import (
    AGE_GROUP_MAX_EMPLOYMENT,
    validate_generated_data,
)


TEST_INTERCEPT = 1.25
TEST_ROWS = 4_000


@pytest.fixture(scope="module")
def generated_worlds() -> dict[str, tuple[pd.DataFrame, dict]]:
    return {
        scenario: generate_synthetic_data(
            scenario=scenario,
            effect_level="moderate",
            n_rows=TEST_ROWS,
            seed=4_994,
            intercept=TEST_INTERCEPT,
        )
        for scenario in (
            "fair_baseline",
            "direct_discrimination",
            "upstream_inequality",
            "mixed_mechanism",
        )
    }


def test_config_resolution_and_switch_matrix() -> None:
    expected = {
        "fair_baseline": (False, False),
        "direct_discrimination": (False, True),
        "upstream_inequality": (True, False),
        "mixed_mechanism": (True, True),
    }
    for scenario, (upstream, direct) in expected.items():
        config = resolve_simulation_config(scenario, "strong", 321, 123)
        assert config["simulation"]["scenario"] == scenario
        assert config["simulation"]["effect_level"] == "strong"
        assert config["simulation"]["n_samples"] == 321
        assert config["simulation"]["random_seed"] == 123
        assert config["scenario_effects"]["upstream"]["enabled"] is upstream
        assert config["scenario_effects"]["direct"]["enabled"] is direct


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scenario": "unknown"}, "Unknown scenario"),
        ({"effect_level": "huge"}, "Unknown effect level"),
        ({"n_rows": 0}, "positive integer"),
        ({"seed": -1}, "non-negative integer"),
    ],
)
def test_invalid_run_configuration_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        resolve_simulation_config(**kwargs)


def test_config_fingerprint_is_deterministic_and_substantive() -> None:
    first = resolve_simulation_config("fair_baseline", "moderate", 100, 1)
    second = resolve_simulation_config("fair_baseline", "moderate", 100, 1)
    changed = resolve_simulation_config("fair_baseline", "moderate", 101, 1)
    assert stable_fingerprint(first) == stable_fingerprint(second)
    assert stable_fingerprint(first) != stable_fingerprint(changed)


def test_fixed_seed_reproduces_exact_data() -> None:
    first, _ = generate_synthetic_data(n_rows=500, seed=77, intercept=TEST_INTERCEPT)
    second, _ = generate_synthetic_data(n_rows=500, seed=77, intercept=TEST_INTERCEPT)
    assert_frame_equal(first, second, check_exact=True)


def test_different_seeds_change_draws() -> None:
    first, _ = generate_synthetic_data(n_rows=500, seed=77, intercept=TEST_INTERCEPT)
    second, _ = generate_synthetic_data(n_rows=500, seed=78, intercept=TEST_INTERCEPT)
    assert not first.drop(columns="application_id").equals(
        second.drop(columns="application_id")
    )


def test_exact_schema_ids_categories_and_outcomes(generated_worlds) -> None:
    data, metadata = generated_worlds["fair_baseline"]
    config = metadata["resolved_configuration"]
    assert list(data.columns) == OUTPUT_COLUMNS
    assert data.shape == (TEST_ROWS, 24)
    assert data["application_id"].is_unique
    assert data["application_id"].iloc[[0, -1]].tolist() == [
        "APP000000001",
        f"APP{TEST_ROWS:09d}",
    ]
    for field, spec in config["population"]["demographics"].items():
        assert set(data[field].astype(object).unique()) <= set(spec["shares"])
    for field in ("loan_purpose", "loan_type", "occupancy_type"):
        assert set(data[field].astype(object).unique()) <= set(
            config["loan_property_variables"][field]["shares"]
        )
    assert data["denial_reason"].isna().all()
    assert data["approval_probability_true"].between(0.0, 1.0).all()
    assert set(data["approved"].unique()) <= {0, 1}


def test_financial_ratio_bounds_and_identities(generated_worlds) -> None:
    data, metadata = generated_worlds["fair_baseline"]
    config = metadata["resolved_configuration"]
    financial = config["financial_variables"]
    loan = config["loan_property_variables"]
    assert data["annual_income"].between(
        financial["annual_income"]["bounds"]["minimum"],
        financial["annual_income"]["bounds"]["maximum"],
    ).all()
    assert data["credit_score"].between(500, 850).all()
    assert (data["liquid_assets"] >= 0).all()
    assert (data["existing_monthly_debt"] >= 0).all()
    assert data["debt_to_income_ratio"].between(0.0, 0.65).all()
    assert data["loan_to_value_ratio"].between(
        loan["loan_to_value_ratio"]["bounds"]["minimum"],
        loan["loan_to_value_ratio"]["bounds"]["maximum"],
    ).all()
    np.testing.assert_allclose(
        data["loan_amount"],
        data["property_value"] * data["loan_to_value_ratio"],
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        data["income_to_loan_ratio"],
        data["annual_income"] / data["loan_amount"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_employment_history_is_age_compatible(generated_worlds) -> None:
    data, _ = generated_worlds["fair_baseline"]
    assert (data["employment_years"] >= 0.0).all()
    for group, maximum in AGE_GROUP_MAX_EMPLOYMENT.items():
        values = data.loc[
            data["age_group"].astype(object) == group, "employment_years"
        ]
        assert (values <= maximum).all()


def test_scenario_feature_paths_are_isolated(generated_worlds) -> None:
    fair, _ = generated_worlds["fair_baseline"]
    direct, _ = generated_worlds["direct_discrimination"]
    upstream, _ = generated_worlds["upstream_inequality"]
    mixed, _ = generated_worlds["mixed_mechanism"]
    pre_outcome = OUTPUT_COLUMNS[:21]
    assert_frame_equal(fair[pre_outcome], direct[pre_outcome], check_exact=True)
    assert_frame_equal(upstream[pre_outcome], mixed[pre_outcome], check_exact=True)
    black = fair["race"].astype(object) == "Black"
    assert upstream.loc[black, "annual_income"].median() < fair.loc[
        black, "annual_income"
    ].median()
    assert upstream.loc[black, "credit_score"].median() < fair.loc[
        black, "credit_score"
    ].median()
    assert upstream.loc[black, "liquid_assets"].median() < fair.loc[
        black, "liquid_assets"
    ].median()


def test_only_declared_upstream_variables_receive_race_shifts(generated_worlds) -> None:
    for scenario in ("upstream_inequality", "mixed_mechanism"):
        _, metadata = generated_worlds[scenario]
        upstream = metadata["resolved_configuration"]["scenario_effects"]["upstream"]
        assert set(upstream["affected_variables"]) == {
            "annual_income",
            "credit_score",
            "liquid_assets",
        }
        assert set(upstream["treatments"]) == set(upstream["affected_variables"])
        for treatment in upstream["treatments"].values():
            shifts = treatment["additive_shift_by_race"]
            assert shifts["Black"] < 0.0
            assert all(value == 0.0 for race, value in shifts.items() if race != "Black")


def test_direct_effect_is_exact_at_fixed_features(generated_worlds) -> None:
    data, metadata = generated_worlds["direct_discrimination"]
    config = metadata["resolved_configuration"]
    result = counterfactual_direct_probabilities(
        data, config, metadata["intercept"]
    )
    reference = result["reference_probability"]
    comparison = result["comparison_probability"]
    configured = config["scenario_effects"]["direct"]["log_odds_by_race"]
    assert configured["White"] == 0.0
    assert configured["Black"] == -0.25
    assert np.all(comparison < reference)
    np.testing.assert_allclose(logit(comparison) - logit(reference), -0.25, atol=1e-12)


def test_scenario_validation_invariants(generated_worlds) -> None:
    for scenario, (data, metadata) in generated_worlds.items():
        report = validate_generated_data(data, metadata)
        assert report["valid"], (scenario, report["errors"])
        assert report["checks"]["scenario_mechanism"]


def test_intercept_solver_reaches_target() -> None:
    predictor = np.linspace(-3.0, 3.0, 20_001)
    intercept, achieved = solve_intercept(predictor, 0.80, -20.0, 20.0, 1e-10)
    assert -20.0 < intercept < 20.0
    assert abs(achieved - 0.80) < 1e-10


def test_same_frozen_intercept_is_used_for_every_scenario(generated_worlds) -> None:
    assert {
        metadata["intercept"] for _, metadata in generated_worlds.values()
    } == {TEST_INTERCEPT}


def test_metadata_contains_reproducibility_fields(generated_worlds) -> None:
    _, metadata = generated_worlds["mixed_mechanism"]
    required = {
        "scenario",
        "effect_level",
        "n_rows",
        "seed",
        "resolved_configuration",
        "intercept",
        "target_baseline_approval_rate",
        "generated_at_utc",
        "package_version",
        "git_revision",
        "config_fingerprint",
        "random_stream_spawn_keys",
        "row_count",
        "schema_column_count",
    }
    assert required <= set(metadata)
    assert set(metadata["random_stream_spawn_keys"]) == {
        "demographics",
        "latent_factors",
        "context",
        "financials",
        "loans",
        "approval",
    }


def test_parquet_and_adjacent_metadata_are_written(tmp_path, generated_worlds) -> None:
    data, metadata = generated_worlds["fair_baseline"]
    path, metadata_path, saved = save_synthetic_dataset(
        data, metadata, tmp_path / "sample.parquet"
    )
    assert path.suffix == ".parquet"
    assert path.exists() and metadata_path.exists()
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == OUTPUT_COLUMNS
    on_disk_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert on_disk_metadata["output_file_path"] == str(path.resolve())
    assert saved["schema_column_count"] == 24


def test_validation_catches_corrupted_derived_identity(generated_worlds) -> None:
    data, metadata = generated_worlds["fair_baseline"]
    corrupted = data.copy()
    corrupted.loc[0, "loan_amount"] += 100.0
    report = validate_generated_data(corrupted, metadata)
    assert not report["valid"]
    assert not report["checks"]["loan_amount_identity"]
    assert any("loan_amount" in error for error in report["errors"])
