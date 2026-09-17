"""Deterministic structural tests for the upstream-opportunity experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.config import load_economic_config
from fair_lending.economic_lending.contracts import make_requested_loan_options
from fair_lending.economic_lending.ml import ML_FEATURES, ml_feature_matrix
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes_from_uniforms
from fair_lending.economic_lending.portfolio import solve_fixed_request_portfolio
from fair_lending.economic_lending.repayment import TRUE_RISK_PREDICTORS, true_repayment_probabilities
from fair_lending.economic_lending.systemic import (
    SYSTEMIC_LENDER_FORBIDDEN,
    calibrate_opportunity_intercept,
    controlled_funding_audit,
    draw_systemic_population_inputs,
    generate_systemic_world,
    identify_opportunity_switchers,
    load_systemic_config,
    opportunity_probability,
    validate_matched_worlds,
)
from fair_lending.economic_lending.traditional import (
    TRADITIONAL_LENDER_FEATURES,
    build_policy_assessments,
)


@pytest.fixture(scope="module")
def matched():
    baseline = load_economic_config()
    systemic = load_systemic_config()
    draws = draw_systemic_population_inputs(800, baseline, systemic, seed=71)
    neutral = generate_systemic_world(draws, baseline, systemic, 0.0)
    treated = generate_systemic_world(draws, baseline, systemic, 0.4)
    return baseline, systemic, draws, neutral, treated


def test_strength_zero_is_neutral_opportunity_process(matched) -> None:
    _, systemic, draws, neutral, _ = matched
    expected = opportunity_probability(draws.group, draws.financial_stability_latent, 0.0, systemic["opportunity_model"])
    np.testing.assert_allclose(neutral["mechanism"].opportunity_probability, expected)
    np.testing.assert_array_equal(neutral["mechanism"].opportunity_access, draws.opportunity_uniform.to_numpy() <= expected)


def test_group_a_opportunity_probability_and_realization_are_invariant(matched) -> None:
    _, _, draws, neutral, treated = matched
    group_a = draws.group.eq("A").to_numpy()
    np.testing.assert_array_equal(neutral["mechanism"].loc[group_a, "opportunity_probability"], treated["mechanism"].loc[group_a, "opportunity_probability"])
    np.testing.assert_array_equal(neutral["mechanism"].loc[group_a, "opportunity_access"], treated["mechanism"].loc[group_a, "opportunity_access"])


def test_group_a_downstream_characteristics_are_exactly_invariant(matched) -> None:
    _, _, draws, neutral, treated = matched
    group_a = draws.group.eq("A").to_numpy()
    pd.testing.assert_frame_equal(neutral["applicants"].loc[group_a].reset_index(drop=True), treated["applicants"].loc[group_a].reset_index(drop=True))
    pd.testing.assert_frame_equal(neutral["loan_options"].loc[group_a].reset_index(drop=True), treated["loan_options"].loc[group_a].reset_index(drop=True))


def test_group_b_opportunity_log_odds_receive_exact_negative_strength_shift(matched) -> None:
    _, _, draws, neutral, treated = matched
    b = draws.group.eq("B").to_numpy()
    p0 = neutral["mechanism"].loc[b, "opportunity_probability"].to_numpy()
    p1 = treated["mechanism"].loc[b, "opportunity_probability"].to_numpy()
    shift = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
    np.testing.assert_allclose(shift, -0.4, atol=1e-12)


def test_same_opportunity_uniform_and_pretreatment_inputs_are_reused(matched) -> None:
    _, _, draws, neutral, treated = matched
    for column in ["financial_stability_latent", "creditworthiness_latent", "opportunity_uniform"]:
        np.testing.assert_array_equal(neutral["mechanism"][column], treated["mechanism"][column])
        np.testing.assert_array_equal(neutral["mechanism"][column], draws[column])


def test_group_b_switcher_classification_is_exact(matched) -> None:
    _, _, _, neutral, treated = matched
    result = identify_opportunity_switchers(neutral["mechanism"], treated["mechanism"], neutral["applicants"])
    expected = result.group.astype(str).eq("B") & result.opportunity_access_neutral.eq(1) & result.opportunity_access_systemic.eq(0)
    pd.testing.assert_series_equal(result.upstream_displaced, expected, check_names=False)
    assert expected.any()


def test_only_neutral_intercept_calibration_hits_target() -> None:
    systemic = load_systemic_config()
    values = np.linspace(-3, 3, 10001)
    alpha = calibrate_opportunity_intercept(values, systemic["opportunity_model"])
    achieved = np.mean(1 / (1 + np.exp(-(alpha + systemic["opportunity_model"]["beta_stability"] * values))))
    assert achieved == pytest.approx(systemic["opportunity_model"]["neutral_access_target"], abs=1e-11)


def test_true_risk_structure_excludes_group_strength_and_opportunity() -> None:
    forbidden = {"group", "systemic_strength", "opportunity_access"}
    assert forbidden.isdisjoint(TRUE_RISK_PREDICTORS)


def test_final_lender_feature_sets_exclude_group_opportunity_and_strength() -> None:
    assert SYSTEMIC_LENDER_FORBIDDEN.isdisjoint(TRADITIONAL_LENDER_FEATURES)
    assert SYSTEMIC_LENDER_FORBIDDEN.isdisjoint(ML_FEATURES)


def test_ml_allowlist_selects_only_six_observables(matched) -> None:
    _, _, _, neutral, _ = matched
    source = neutral["applicants"].merge(neutral["loan_options"][["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
    assert tuple(ml_feature_matrix(source).columns) == ML_FEATURES


def test_direct_belief_delta_is_zero_and_policy_probabilities_are_undistorted(matched) -> None:
    _, systemic, _, neutral, _ = matched
    assert systemic["direct_belief_delta"] == 0.0
    probability = pd.DataFrame({"applicant_id": neutral["applicants"].applicant_id, "repayment_probability_traditional": 0.995})
    policy = build_policy_assessments(neutral["loan_options"], probability)
    np.testing.assert_array_equal(policy.repayment_probability_base, policy.repayment_probability_used)


def test_downstream_dti_and_ltv_are_recomputed_coherently(matched) -> None:
    _, _, draws, neutral, treated = matched
    report = validate_matched_worlds(draws, neutral, treated)
    assert report["dti_ltv_identities_valid"]


def test_cohort_and_age_boundaries_remain_matched(matched) -> None:
    _, _, _, neutral, treated = matched
    for column in ["applicant_id", "group", "cohort", "age_years", "credit_score"]:
        pd.testing.assert_series_equal(neutral["applicants"][column], treated["applicants"][column])


def test_employment_remains_age_feasible(matched) -> None:
    _, _, _, _, treated = matched
    applicants = treated["applicants"]
    assert (applicants.employment_years >= 0).all()
    assert (applicants.employment_years <= applicants.age_years - 18 + 1e-12).all()


def test_changed_truth_uses_same_paired_repayment_uniforms(matched) -> None:
    baseline, _, _, neutral, treated = matched
    truth0 = true_repayment_probabilities(neutral["applicants"], neutral["loan_options"], baseline["true_risk"])
    truth1 = true_repayment_probabilities(treated["applicants"], treated["loan_options"], baseline["true_risk"])
    uniforms = np.random.default_rng(4).random((len(truth0), int(neutral["loan_options"].term_periods.iloc[0])))
    first = simulate_repayment_outcomes_from_uniforms(neutral["loan_options"], truth0, uniforms)
    repeat = simulate_repayment_outcomes_from_uniforms(neutral["loan_options"], truth0, uniforms)
    changed = simulate_repayment_outcomes_from_uniforms(treated["loan_options"], truth1, uniforms)
    pd.testing.assert_frame_equal(first, repeat)
    assert not first.equals(changed)


def test_absolute_budgets_are_fixed_and_not_world_request_shares(matched) -> None:
    _, systemic, _, neutral, treated = matched
    assert systemic["budgets"] == {"nonbinding_100pct": 583043937.08, "moderate_40pct": 233217574.832, "tight_20pct": 116608787.416}
    assert neutral["loan_options"].requested_principal.sum() != treated["loan_options"].requested_principal.sum()


def test_milp_feasibility_and_optimality_on_systemic_world(matched) -> None:
    _, _, _, neutral, _ = matched
    loans = neutral["loan_options"].head(40).copy()
    probability = pd.DataFrame({"applicant_id": loans.applicant_id, "repayment_probability_traditional": 0.999})
    policy = build_policy_assessments(loans, probability)
    budget = float(loans.requested_principal.sum() * .2)
    result = solve_fixed_request_portfolio(loans, policy, budget, world_id="systemic", budget_id="test", policy_id="group_blind")
    assert result.solver_metadata["budget_feasible"]
    assert result.solver_metadata["optimality_gap"] == pytest.approx(0.0)
    assert result.decisions.funded_amount.sum() <= budget + 1e-6


def test_persistable_decision_fields_are_truth_free(matched) -> None:
    _, _, _, neutral, _ = matched
    loans = neutral["loan_options"].head(40).copy()
    probability = pd.DataFrame({"applicant_id": loans.applicant_id, "repayment_probability_traditional": 0.999})
    policy = build_policy_assessments(loans, probability)
    result = solve_fixed_request_portfolio(loans, policy, 1e9, world_id="w", budget_id="b", policy_id="p")
    persisted = result.decisions.drop(columns=["perceived_expected_profit", "perceived_profit_per_dollar"])
    assert not any("true" in column for column in persisted)


def test_difference_in_gap_and_paired_switcher_arithmetic() -> None:
    neutral_a, neutral_b = .44, .42
    systemic_a, systemic_b = .46, .37
    assert (systemic_b - systemic_a) - (neutral_b - neutral_a) == pytest.approx(-.07)
    changes = np.array([-1.5, -1.5, -1.0])
    assert changes.mean() == pytest.approx(-4 / 3)


def test_controlled_audit_uses_descendant_controls_without_truth(matched) -> None:
    baseline, _, _, neutral, _ = matched
    rng = np.random.default_rng(9)
    decisions = pd.DataFrame({"applicant_id": neutral["applicants"].applicant_id, "approved": rng.random(len(neutral["applicants"])) < .5})
    result = controlled_funding_audit(decisions, neutral["applicants"], neutral["loan_options"], baseline["true_risk"])
    assert set(result.model) == {"U0_race_only", "U1_downstream_controls"}
    assert result.converged.all()
    assert not any("true" in column for column in result)
