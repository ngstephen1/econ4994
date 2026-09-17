"""Tests for the Version 2 traditional lender risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fair_lending.economic_lending.config import load_economic_config
from fair_lending.economic_lending.contracts import make_fixed_loan_options
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.lender_evaluation import (
    group_risk_audit,
    profit_estimation_diagnostics,
    realized_label_metrics,
)
from fair_lending.economic_lending.population import generate_economic_population
from fair_lending.economic_lending.traditional import (
    applicant_row_weights,
    build_policy_assessments,
    fit_traditional_logit,
    lender_design_matrix,
    predict_applicant_risk,
)
from fair_lending.economic_lending.schema import BaselineEconomicParameters


def _toy_history_inputs():
    applicants = pd.DataFrame(
        {
            "applicant_id": ["complete", "default4"],
            "group": ["A", "B"],
            "cohort": ["historical_train", "historical_train"],
            "age_years": [40, 40],
            "annual_income": [90_000.0, 90_000.0],
            "credit_score": [700, 700],
            "employment_years": [15.0, 15.0],
            "liquid_assets": [45_000.0, 45_000.0],
            "existing_monthly_debt": [1_000.0, 1_000.0],
            "property_value": [200_000.0, 200_000.0],
            "requested_loan_amount": [100.0, 100.0],
        }
    )
    loans = make_fixed_loan_options(
        applicants, BaselineEconomicParameters(term_periods=5)
    )
    loans["requested_ltv"] = 0.5
    loans["first_period_dti"] = 0.3
    outcomes = pd.DataFrame(
        {
            "applicant_id": ["complete", "default4"],
            "default_period": pd.array([pd.NA, 4], dtype="Int64"),
            "completed_all_payments": [True, False],
            "realized_total_receipts": [200.0, 120.0],
            "realized_profit": [95.0, 15.0],
        }
    )
    return applicants, loans, outcomes


def test_history_stops_at_default_and_completed_loan_has_t_positive_rows() -> None:
    applicants, loans, outcomes = _toy_history_inputs()
    history = build_at_risk_payment_history(applicants, loans, outcomes)
    complete = history.loc[history["applicant_id"] == "complete"]
    defaulted = history.loc[history["applicant_id"] == "default4"]
    assert len(complete) == 5
    assert complete["paid_this_period"].tolist() == [1, 1, 1, 1, 1]
    assert len(defaulted) == 4
    assert defaulted["period_index"].tolist() == [1, 2, 3, 4]
    assert defaulted["paid_this_period"].tolist() == [1, 1, 1, 0]


def test_history_never_crosses_applicant_cohorts() -> None:
    result = generate_economic_population(100, load_economic_config())
    history = build_at_risk_payment_history(
        result.applicants, result.loan_options, result.loan_outcomes
    )
    assert (history.groupby("applicant_id")["cohort"].nunique() == 1).all()


def test_lender_design_excludes_truth_group_identifier_cohort_and_period() -> None:
    config = load_economic_config()
    result = generate_economic_population(100, config)
    history = build_at_risk_payment_history(
        result.applicants, result.loan_options, result.loan_outcomes
    )
    training = history.loc[history["cohort"] == "historical_train"]
    design = lender_design_matrix(training, config["true_risk"])
    forbidden = {
        "group",
        "applicant_id",
        "cohort",
        "period_index",
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
    }
    assert forbidden.isdisjoint(design.columns)
    leaked = training.copy()
    leaked["repayment_probability_per_period_true"] = 0.9
    with pytest.raises(ValueError, match="hidden truth leaked"):
        lender_design_matrix(leaked, config["true_risk"])


def test_logit_probabilities_are_valid_and_reproducible() -> None:
    config = load_economic_config()
    result = generate_economic_population(300, config)
    history = build_at_risk_payment_history(
        result.applicants, result.loan_options, result.loan_outcomes
    )
    training = history.loc[history["cohort"] == "historical_train"]
    evaluation_applicants = result.applicants.loc[result.applicants["cohort"] == "evaluation"]
    evaluation_loans = result.loan_options.loc[
        result.loan_options["applicant_id"].isin(evaluation_applicants["applicant_id"])
    ]
    first = fit_traditional_logit(training, config["true_risk"])
    second = fit_traditional_logit(training, config["true_risk"])
    assert first.coefficients == second.coefficients
    first_predictions = predict_applicant_risk(
        first, evaluation_applicants, evaluation_loans, config["true_risk"]
    )
    second_predictions = predict_applicant_risk(
        second, evaluation_applicants, evaluation_loans, config["true_risk"]
    )
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    assert first_predictions["repayment_probability_traditional"].between(0, 1).all()


def test_evaluation_rows_are_rejected_during_fitting() -> None:
    config = load_economic_config()
    result = generate_economic_population(100, config)
    history = build_at_risk_payment_history(
        result.applicants, result.loan_options, result.loan_outcomes
    )
    with pytest.raises(ValueError, match="historical_train"):
        fit_traditional_logit(history, config["true_risk"])


def test_applicant_weight_sums_equal_one() -> None:
    applicants, loans, outcomes = _toy_history_inputs()
    history = build_at_risk_payment_history(applicants, loans, outcomes)
    weights = applicant_row_weights(history)
    sums = weights.groupby(history["applicant_id"]).sum()
    np.testing.assert_allclose(sums, 1.0)


def test_policy_assessments_are_truth_free_and_use_undistorted_probability() -> None:
    applicants, loans, _ = _toy_history_inputs()
    predictions = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "repayment_probability_traditional": [0.9, 0.8],
        }
    )
    policy = build_policy_assessments(loans, predictions)
    assert not any("true" in column for column in policy.columns)
    assert np.array_equal(
        policy["repayment_probability_base"], policy["repayment_probability_used"]
    )


def test_perceived_receipts_and_profit_use_estimated_rho() -> None:
    applicants = pd.DataFrame({"applicant_id": [1]})
    loans = make_fixed_loan_options(applicants)
    predictions = pd.DataFrame(
        {"applicant_id": [1], "repayment_probability_traditional": [0.9]}
    )
    policy = build_policy_assessments(loans, predictions)
    assert policy.loc[0, "expected_receipts_perceived"] == pytest.approx(119.7)
    assert policy.loc[0, "expected_profit_perceived"] == pytest.approx(14.7)


def test_profit_sign_diagnostics_on_toy_cases() -> None:
    applicants = pd.DataFrame(
        {"applicant_id": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]}
    )
    loans = make_fixed_loan_options(applicants)
    true_rho = [0.9, 0.7, 0.9, 0.7]
    perceived_rho = [0.9, 0.9, 0.7, 0.7]
    truth = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "repayment_probability_per_period_true": true_rho,
            "full_repayment_probability_true": np.square(true_rho),
        }
    )
    predictions = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "repayment_probability_traditional": perceived_rho,
        }
    )
    policy = build_policy_assessments(loans, predictions)
    overall, by_group = profit_estimation_diagnostics(
        loans, policy, truth, applicants
    )
    assert overall["agreement_rate"] == pytest.approx(0.5)
    assert overall["wrong_profit_sign_rate"] == pytest.approx(0.5)
    assert overall["false_profitable_rate"] == pytest.approx(0.5)
    assert overall["false_unprofitable_rate"] == pytest.approx(0.5)
    assert set(by_group["group"]) == {"A", "B"}


def test_group_audit_calculations() -> None:
    applicants = pd.DataFrame(
        {"applicant_id": [1, 2], "group": ["A", "B"]}
    )
    predictions = pd.DataFrame(
        {"applicant_id": [1, 2], "estimated": [0.8, 0.7]}
    )
    truth = pd.DataFrame(
        {
            "applicant_id": [1, 2],
            "repayment_probability_per_period_true": [0.75, 0.8],
        }
    )
    audit = group_risk_audit(applicants, predictions, truth, "estimated").set_index("group")
    assert audit.loc["A", "mean_prediction_error"] == pytest.approx(0.05)
    assert audit.loc["A", "mae"] == pytest.approx(0.05)
    assert audit.loc["B", "mean_prediction_error"] == pytest.approx(-0.10)
    assert audit.loc["B", "rmse"] == pytest.approx(0.10)


def test_oracle_realized_metrics_match_sklearn() -> None:
    history = pd.DataFrame(
        {
            "applicant_id": [1, 1, 2, 2],
            "paid_this_period": [1, 1, 1, 0],
        }
    )
    oracle = pd.DataFrame(
        {"applicant_id": [1, 2], "oracle": [0.9, 0.6]}
    )
    metrics = realized_label_metrics(history, oracle, "oracle")
    target = np.array([1, 1, 1, 0])
    probability = np.array([0.9, 0.9, 0.6, 0.6])
    assert metrics["brier_score"] == pytest.approx(brier_score_loss(target, probability))
    assert metrics["log_loss"] == pytest.approx(log_loss(target, probability))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(target, probability))
