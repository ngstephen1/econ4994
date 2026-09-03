"""Deterministic tests for descriptive and statsmodels recovery analysis."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from fair_lending.analysis.descriptive import (
    black_white_raw_gap,
    independent_proportion_difference,
)
from fair_lending.analysis.estimands import standardized_black_white_contrast
from fair_lending.analysis.logit import (
    FORBIDDEN_PRIMARY_FIELDS,
    build_design_matrix,
    fit_logit_model,
    prepare_regression_sample,
)
from fair_lending.simulation.generator import generate_synthetic_data


INTERCEPT = 2.0362202310934663
ANALYSIS_N = 50_000
SEED = 4_994
COEFFICIENT_TOLERANCE = 0.10


@lru_cache(maxsize=None)
def scenario_data(scenario: str) -> tuple[pd.DataFrame, dict]:
    data, metadata = generate_synthetic_data(
        scenario,
        "moderate",
        ANALYSIS_N,
        SEED,
        intercept=INTERCEPT,
    )
    return data, metadata["resolved_configuration"]


@lru_cache(maxsize=None)
def scenario_fit(scenario: str, model: str) -> dict:
    data, config = scenario_data(scenario)
    return fit_logit_model(data, config, model)


def test_transformed_covariates_match_dgp_units() -> None:
    data, config = scenario_data("fair_baseline")
    sample = prepare_regression_sample(data.head(500), config)

    np.testing.assert_allclose(
        sample["log_income"],
        (np.log(sample["annual_income"]) - np.log(95_000.0)) / 0.50,
    )
    np.testing.assert_allclose(
        sample["credit_score_50"], (sample["credit_score"] - 720.0) / 50.0
    )
    np.testing.assert_allclose(
        sample["employment_years_10"],
        (sample["employment_years"] - 10.0) / 10.0,
    )
    np.testing.assert_allclose(
        sample["log_liquid_assets"],
        np.log(sample["liquid_assets"] + 1_000.0) - np.log(36_000.0),
    )
    np.testing.assert_allclose(
        sample["dti_10pp"], (sample["debt_to_income_ratio"] - 0.36) / 0.10
    )
    np.testing.assert_allclose(
        sample["ltv_10pp"], (sample["loan_to_value_ratio"] - 0.80) / 0.10
    )


def test_white_is_reference_and_black_indicator_is_exact() -> None:
    data, config = scenario_data("fair_baseline")
    sample = prepare_regression_sample(data, config)
    matrix = build_design_matrix(sample, config, "model_3")

    assert set(sample["race"].astype(object).unique()) == {"White", "Black"}
    assert (sample.loc[sample["race"].astype(object) == "White", "black"] == 0).all()
    assert (sample.loc[sample["race"].astype(object) == "Black", "black"] == 1).all()
    assert "race__white" not in matrix
    assert "race__black" not in matrix
    assert "black" in matrix
    assert "loan_purpose__home_purchase" not in matrix
    assert "loan_type__conventional" not in matrix
    assert "occupancy_type__principal_residence" not in matrix
    assert "age_group__35_44" not in matrix
    assert "sex__male" not in matrix
    assert "ethnicity__not_hispanic_or_latino" not in matrix


def test_fair_baseline_adjusted_black_effect_is_near_zero() -> None:
    coefficient = scenario_fit("fair_baseline", "model_2")["summary"][
        "black_coefficient"
    ]
    assert abs(coefficient) < COEFFICIENT_TOLERANCE


def test_direct_discrimination_recovers_configured_effect() -> None:
    coefficient = scenario_fit("direct_discrimination", "model_2")["summary"][
        "black_coefficient"
    ]
    assert abs(coefficient - (-0.25)) < COEFFICIENT_TOLERANCE


def test_upstream_adjustment_moves_association_materially_toward_zero() -> None:
    unadjusted = scenario_fit("upstream_inequality", "model_0")["summary"][
        "black_coefficient"
    ]
    adjusted = scenario_fit("upstream_inequality", "model_2")["summary"][
        "black_coefficient"
    ]
    assert abs(adjusted) < abs(unadjusted) * 0.40
    assert abs(adjusted) < COEFFICIENT_TOLERANCE


def test_mixed_adjustment_retains_direct_component() -> None:
    coefficient = scenario_fit("mixed_mechanism", "model_2")["summary"][
        "black_coefficient"
    ]
    assert abs(coefficient - (-0.25)) < COEFFICIENT_TOLERANCE


def test_standardized_probability_contrast_matches_manual_calculation() -> None:
    class FixedLogit:
        @staticmethod
        def predict(matrix: pd.DataFrame) -> np.ndarray:
            return expit(-0.4 + 0.7 * matrix["x"] - 0.25 * matrix["black"])

    matrix = pd.DataFrame(
        {"const": 1.0, "black": [0.0, 1.0, 0.0], "x": [-1.0, 0.0, 1.0]}
    )
    expected = np.mean(expit(-0.4 + 0.7 * matrix["x"] - 0.25) - expit(-0.4 + 0.7 * matrix["x"]))
    observed = standardized_black_white_contrast(FixedLogit(), matrix)
    assert observed == expected


def test_primary_models_exclude_proxies_outcomes_and_redundant_fields() -> None:
    data, config = scenario_data("mixed_mechanism")
    sample = prepare_regression_sample(data, config)
    for model in ("model_1", "model_2", "model_3"):
        columns = set(build_design_matrix(sample, config, model).columns)
        assert not columns.intersection(FORBIDDEN_PRIMARY_FIELDS)


def test_raw_gap_interval_and_denial_identity() -> None:
    result = independent_proportion_difference(40, 100, 50, 100)
    assert result["difference"] == pytest.approx(-0.10)
    assert result["ci_low"] < result["difference"] < result["ci_high"]

    data, _ = scenario_data("direct_discrimination")
    gap = black_white_raw_gap(data, "direct_discrimination")
    assert gap["raw_denial_gap"] == -gap["raw_approval_gap"]
    assert gap["denial_ci_low"] == -gap["ci_high"]


def test_deterministic_reruns_reproduce_data_and_analysis() -> None:
    first, first_metadata = generate_synthetic_data(
        "direct_discrimination", "moderate", 2_000, SEED, intercept=INTERCEPT
    )
    second, second_metadata = generate_synthetic_data(
        "direct_discrimination", "moderate", 2_000, SEED, intercept=INTERCEPT
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_metadata["config_fingerprint"] == second_metadata["config_fingerprint"]
    assert black_white_raw_gap(first, "direct_discrimination") == black_white_raw_gap(
        second, "direct_discrimination"
    )
    first_fit = fit_logit_model(
        first, first_metadata["resolved_configuration"], "model_2"
    )
    second_fit = fit_logit_model(
        second, second_metadata["resolved_configuration"], "model_2"
    )
    assert first_fit["summary"]["black_coefficient"] == second_fit["summary"][
        "black_coefficient"
    ]
    assert standardized_black_white_contrast(
        first_fit["result"], first_fit["design_matrix"]
    ) == standardized_black_white_contrast(
        second_fit["result"], second_fit["design_matrix"]
    )
