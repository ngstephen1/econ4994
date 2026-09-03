"""Deterministic tests for the Prompt 5 DTI dependency recalibration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fair_lending.simulation.calibration import (
    calibrate_intercept,
    expected_calibration_fingerprint,
)
from fair_lending.simulation.config import resolve_simulation_config
from fair_lending.simulation.diagnostics import calculate_dti_components
from fair_lending.simulation.generator import OUTPUT_COLUMNS, generate_synthetic_data
from fair_lending.simulation.population import create_random_streams, generate_population


@pytest.fixture(scope="module")
def large_fair_population():
    config = resolve_simulation_config(
        "fair_baseline", "moderate", n_rows=100_000, seed=4_994
    )
    streams, _ = create_random_streams(4_994)
    data, _ = generate_population(config, streams)
    components = calculate_dti_components(data, config)
    return data, components, config


@pytest.fixture(scope="module")
def recalibrated_artifact(tmp_path_factory):
    artifact_path = tmp_path_factory.mktemp("calibration") / "intercept.json"
    artifact_path.write_text(
        json.dumps(
            {
                "intercept": 2.2422267822548747,
                "target_mean_probability": 0.80,
                "achieved_mean_probability": 0.80,
                "calibration_population_size": 1_000_000,
                "calibration_seed": 499_400,
                "config_fingerprint": "pre-recalibration-fingerprint",
                "generated_at_utc": "2026-09-03T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return calibrate_intercept(
        force=True,
        artifact_path=artifact_path,
        reason="population/DTI recalibration",
    )


def test_final_dti_ceiling_mass_is_below_five_percent(large_fair_population) -> None:
    _, components, _ = large_fair_population
    assert float((components["pre_clip_dti"] >= 0.65).mean()) < 0.05


def test_pre_clip_dti_is_diagnostic_only(large_fair_population) -> None:
    data, components, _ = large_fair_population
    assert "pre_clip_dti" in components
    assert "pre_clip_dti" not in data
    assert "pre_clip_dti" not in OUTPUT_COLUMNS
    assert len(OUTPUT_COLUMNS) == 24


def test_dti_arithmetic_matches_persisted_winsorized_value(
    large_fair_population,
) -> None:
    data, components, config = large_fair_population
    payment = config["internal_housing_payment"]
    expected_pi = data["loan_amount"] * payment[
        "monthly_principal_interest_factor"
    ]
    expected_tax_insurance = data["property_value"] * payment[
        "monthly_tax_insurance_factor_of_property_value"
    ]
    expected_raw = (
        data["existing_monthly_debt"] + expected_pi + expected_tax_insurance
    ) / (data["annual_income"] / 12.0)
    bounds = config["financial_variables"]["debt_to_income_ratio"]["bounds"]
    np.testing.assert_allclose(components["pre_clip_dti"], expected_raw, rtol=1e-13)
    np.testing.assert_allclose(
        data["debt_to_income_ratio"],
        np.clip(expected_raw, bounds["minimum"], bounds["maximum"]),
        rtol=1e-13,
    )


def test_recalibration_changes_only_declared_property_dependencies() -> None:
    config = resolve_simulation_config("fair_baseline", "moderate", 100, 1)
    property_spec = config["loan_property_variables"]["property_value"]
    assert property_spec["income_elasticity"] == 0.90
    assert property_spec["log_residual_standard_deviation"] == 0.18
    assert property_spec["reference_median_usd"] == 400_000.0
    assert config["internal_housing_payment"]["annual_interest_rate"] == 0.0675
    assert config["internal_housing_payment"][
        "monthly_principal_interest_factor"
    ] == 0.00648598
    assert config["internal_housing_payment"][
        "monthly_tax_insurance_factor_of_property_value"
    ] == 0.0012


def test_experimental_treatment_values_are_unchanged() -> None:
    direct = resolve_simulation_config("direct_discrimination", "moderate", 100, 1)
    upstream = resolve_simulation_config("upstream_inequality", "moderate", 100, 1)
    assert direct["scenario_effects"]["direct"]["log_odds_by_race"]["Black"] == -0.25
    treatments = upstream["scenario_effects"]["upstream"]["treatments"]
    assert treatments["annual_income"]["additive_shift_by_race"]["Black"] == -0.105361
    assert treatments["credit_score"]["additive_shift_by_race"]["Black"] == -25.0
    assert treatments["liquid_assets"]["additive_shift_by_race"]["Black"] == -0.287682


def test_fair_baseline_remains_treatment_free() -> None:
    config = resolve_simulation_config("fair_baseline", "strong", 100, 1)
    effects = config["scenario_effects"]
    assert effects["direct"]["enabled"] is False
    assert effects["upstream"]["enabled"] is False
    assert all(value == 0.0 for value in effects["direct"]["log_odds_by_race"].values())
    assert all(
        value == 0.0
        for treatment in effects["upstream"]["treatments"].values()
        for value in treatment["additive_shift_by_race"].values()
    )
    assert config["context_variables"]["neighborhood_minority_share"][
        "baseline_race_dependency"
    ] is False


def test_recalibrated_artifact_hits_target_and_preserves_provenance(
    recalibrated_artifact,
) -> None:
    artifact = recalibrated_artifact
    assert artifact["calibration_population_size"] == 1_000_000
    assert artifact["calibration_seed"] == 499_400
    assert artifact["config_fingerprint"] == expected_calibration_fingerprint()
    assert abs(
        artifact["achieved_mean_probability"]
        - artifact["target_mean_probability"]
    ) < 5e-4
    assert artifact["recalibration_reason"] == "population/DTI recalibration"
    assert math.isclose(
        artifact["previous_calibration"]["intercept"],
        2.2422267822548747,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_all_scenarios_use_the_same_recalibrated_intercept(
    recalibrated_artifact,
) -> None:
    intercepts = set()
    for scenario in (
        "fair_baseline",
        "direct_discrimination",
        "upstream_inequality",
        "mixed_mechanism",
    ):
        data, metadata = generate_synthetic_data(
            scenario=scenario,
            effect_level="moderate",
            n_rows=200,
            seed=77,
            intercept=recalibrated_artifact["intercept"],
        )
        assert data.shape == (200, 24)
        intercepts.add(metadata["intercept"])
    assert len(intercepts) == 1
