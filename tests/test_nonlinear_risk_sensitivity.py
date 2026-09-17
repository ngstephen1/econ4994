"""Structural tests for the frozen nonlinear-risk sensitivity world."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.config import load_economic_config
from fair_lending.economic_lending.contracts import make_requested_loan_options
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.ml import ML_FEATURES
from fair_lending.economic_lending.nonlinear import (
    NONLINEAR_ALLOWED_OBSERVABLES,
    NONLINEAR_WORLD_ID,
    calibrate_nonlinear_intercept,
    load_nonlinear_config,
    nonlinear_repayment_probabilities,
    nonlinear_score_components,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes_from_uniforms
from fair_lending.economic_lending.population import generate_economic_population
from fair_lending.economic_lending.repayment import transformed_true_risk_predictors
from fair_lending.economic_lending.traditional import (
    TRADITIONAL_LENDER_FEATURES,
    build_policy_assessments,
)


@pytest.fixture(scope="module")
def matched_worlds():
    baseline = load_economic_config()
    nonlinear = load_nonlinear_config()
    population = generate_economic_population(400, baseline, seed=4994)
    truth = nonlinear_repayment_probabilities(
        population.applicants, population.loan_options, baseline["true_risk"], nonlinear
    )
    return baseline, nonlinear, population, truth


def test_baseline_fingerprint_and_artifact_path_are_preserved() -> None:
    baseline = load_economic_config()
    nonlinear = load_nonlinear_config()
    assert baseline["metadata"]["config_fingerprint"] == "68869774a4c19e6b81fbd660c71e387b75d0d9a504d3aba9ac905c2a00ad55f2"
    assert nonlinear["baseline_config_fingerprint"] == baseline["metadata"]["config_fingerprint"]
    assert baseline["artifacts"]["data_directory"] != nonlinear["artifacts"]["data_directory"]


def test_nonlinear_world_has_distinct_id_and_frozen_intercept() -> None:
    config = load_nonlinear_config()
    assert config["risk_world"] == NONLINEAR_WORLD_ID
    assert config["risk_world"] != config["baseline_world"]
    assert isinstance(config["calibration"]["solved_intercept"], float)


def test_same_applicants_and_contracts_are_used_across_truth_worlds(matched_worlds) -> None:
    baseline, nonlinear, population, nonlinear_truth = matched_worlds
    assert nonlinear_truth["applicant_id"].equals(population.applicants["applicant_id"])
    assert nonlinear_truth["applicant_id"].equals(population.loan_options["applicant_id"])
    assert len(nonlinear_truth) == len(population.simulation_truth)


def test_nonlinear_truth_is_group_invariant_and_uses_only_allowed_observables(matched_worlds) -> None:
    baseline, nonlinear, population, truth = matched_worlds
    changed = population.applicants.copy()
    changed["group"] = np.where(changed["group"].eq("A"), "B", "A")
    alternate = nonlinear_repayment_probabilities(changed, population.loan_options, baseline["true_risk"], nonlinear)
    pd.testing.assert_frame_equal(truth, alternate)
    assert NONLINEAR_ALLOWED_OBSERVABLES == frozenset(ML_FEATURES)


def test_nonlinear_probabilities_are_valid(matched_worlds) -> None:
    *_, truth = matched_worlds
    assert truth["repayment_probability_per_period_true"].between(0, 1).all()
    assert truth["full_repayment_probability_true"].between(0, 1).all()


def test_intercept_calibration_is_reproducible(matched_worlds) -> None:
    baseline, nonlinear, population, _ = matched_worlds
    first = calibrate_nonlinear_intercept(population.applicants, population.loan_options, baseline["true_risk"], nonlinear)
    second = calibrate_nonlinear_intercept(population.applicants, population.loan_options, baseline["true_risk"], nonlinear)
    assert first == pytest.approx(second, abs=1e-12)


def test_declared_threshold_interaction_and_asset_terms_are_active(matched_worlds) -> None:
    baseline, nonlinear, population, _ = matched_worlds
    transformed = transformed_true_risk_predictors(population.applicants, population.loan_options, baseline["true_risk"])
    components = nonlinear_score_components(transformed, nonlinear)
    assert tuple(components.columns) == (
        "high_dti_convex_penalty", "weak_credit_high_ltv_interaction", "bounded_asset_buffer"
    )
    assert (components["high_dti_convex_penalty"] <= 0).all()
    assert (components["weak_credit_high_ltv_interaction"] <= 0).all()
    assert components["bounded_asset_buffer"].nunique() > 1


def test_paired_uniform_outcomes_are_absorbing_and_cohorts_unchanged(matched_worlds) -> None:
    baseline, nonlinear, population, truth = matched_worlds
    uniforms = np.random.default_rng(7).random((len(truth), int(population.loan_options.term_periods.max())))
    outcomes = simulate_repayment_outcomes_from_uniforms(population.loan_options, truth, uniforms)
    history = build_at_risk_payment_history(population.applicants, population.loan_options, outcomes)
    defaults = outcomes.dropna(subset=["default_period"]).set_index("applicant_id")["default_period"]
    contributed = history.groupby("applicant_id").period_index.max()
    assert (contributed.loc[defaults.index].to_numpy() == defaults.to_numpy()).all()
    assert (history.groupby("applicant_id").cohort.nunique() == 1).all()


def test_traditional_has_no_nonlinear_engineered_terms_and_ml_has_no_extra_fields() -> None:
    assert not any("threshold" in x or "interaction" in x for x in TRADITIONAL_LENDER_FEATURES)
    assert set(ML_FEATURES) == set(NONLINEAR_ALLOWED_OBSERVABLES)


def test_nonlinear_policy_is_truth_free_and_accounting_unchanged(matched_worlds) -> None:
    _, nonlinear, population, truth = matched_worlds
    probabilities = truth[["applicant_id", "repayment_probability_per_period_true"]].rename(
        columns={"repayment_probability_per_period_true": "estimate"}
    )
    policy = build_policy_assessments(
        population.loan_options, probabilities,
        policy_id=nonlinear["policies"]["ml"], probability_column="estimate"
    )
    assert not any("true" in column for column in policy.columns)
    assert np.array_equal(policy.repayment_probability_base, policy.repayment_probability_used)
    assert np.isfinite(policy.expected_profit_perceived).all()


def test_comparison_error_arithmetic() -> None:
    truth = np.array([.8, .9]); traditional = np.array([.7, .95]); ml = np.array([.82, .86])
    difference = np.abs(ml - truth) - np.abs(traditional - truth)
    assert difference.mean() == pytest.approx(-.045)
    assert (difference < 0).mean() == pytest.approx(1.0)
