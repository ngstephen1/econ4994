"""Deterministic boundary and accounting tests for the V2 ML benchmark."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fair_lending.economic_lending.contracts import make_fixed_loan_options
from fair_lending.economic_lending.lender_evaluation import (
    break_even_disagreement_table,
    calibration_table,
    group_risk_audit,
    realized_label_metrics,
)
from fair_lending.economic_lending.ml import (
    ML_FEATURES,
    load_ml_config,
    ml_feature_matrix,
    predict_ml_applicant_risk,
    tune_hist_gradient_boosting,
)
from fair_lending.economic_lending.schema import POLICY_ASSESSMENTS_COLUMNS
from fair_lending.economic_lending.traditional import build_policy_assessments


def _history(cohort: str, n_applicants: int = 40) -> pd.DataFrame:
    """Small deterministic at-risk table containing both payment outcomes."""

    applicant = np.repeat(np.arange(n_applicants), 2)
    period = np.tile([1, 2], n_applicants)
    credit = 620.0 + applicant * 4.0
    dti = 0.52 - applicant * 0.005
    paid = ((applicant + period) % 7 != 0).astype(np.int8)
    return pd.DataFrame(
        {
            "applicant_id": [f"{cohort}-{value}" for value in applicant],
            "cohort": cohort,
            "group": np.where(applicant % 2 == 0, "A", "B"),
            "period_index": period,
            "paid_this_period": paid,
            "annual_income": 45_000.0 + applicant * 1_500.0,
            "credit_score": credit,
            "employment_years": 2.0 + applicant * 0.4,
            "liquid_assets": 5_000.0 + applicant * 1_000.0,
            "first_period_dti": dti,
            "requested_ltv": 0.95 - applicant * 0.004,
        }
    )


def _small_ml_config() -> dict:
    config = copy.deepcopy(load_ml_config())
    config["fixed_parameters"]["min_samples_leaf"] = 5
    config["grid"] = {
        "learning_rate": [0.05],
        "max_iter": [20],
        "max_leaf_nodes": [7],
    }
    return config


def test_ml_feature_allowlist_is_exact_and_excludes_nonfeatures() -> None:
    assert ML_FEATURES == (
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "first_period_dti",
        "requested_ltv",
    )
    history = _history("historical_train")
    design = ml_feature_matrix(history)
    assert tuple(design.columns) == ML_FEATURES
    assert {
        "group",
        "applicant_id",
        "cohort",
        "period_index",
        "paid_this_period",
    }.isdisjoint(design.columns)


@pytest.mark.parametrize(
    "forbidden",
    [
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
        "expected_profit_true",
        "future_default_period_potential",
    ],
)
def test_hidden_truth_and_future_outcomes_are_rejected(forbidden: str) -> None:
    history = _history("historical_train")
    history[forbidden] = 0.5
    with pytest.raises(ValueError, match="hidden truth leaked"):
        ml_feature_matrix(history)


def test_training_and_validation_boundaries_are_enforced() -> None:
    training = _history("historical_train")
    validation = _history("historical_validation")
    evaluation = _history("evaluation")
    config = _small_ml_config()
    with pytest.raises(ValueError, match="historical_train"):
        tune_hist_gradient_boosting(evaluation, validation, config)
    with pytest.raises(ValueError, match="historical_validation"):
        tune_hist_gradient_boosting(training, evaluation, config)


def test_fixed_random_state_reproduces_selection_and_predictions() -> None:
    training = _history("historical_train")
    validation = _history("historical_validation")
    config = _small_ml_config()
    first = tune_hist_gradient_boosting(training, validation, config)
    second = tune_hist_gradient_boosting(training, validation, config)
    pd.testing.assert_frame_equal(first.trials, second.trials)
    first_probability = first.estimator.predict_proba(ml_feature_matrix(validation))[:, 1]
    second_probability = second.estimator.predict_proba(ml_feature_matrix(validation))[:, 1]
    np.testing.assert_array_equal(first_probability, second_probability)
    assert np.logical_and(first_probability >= 0, first_probability <= 1).all()


def test_applicant_prediction_is_one_valid_probability_per_request() -> None:
    training = _history("historical_train")
    validation = _history("historical_validation")
    fitted = tune_hist_gradient_boosting(training, validation, _small_ml_config())
    applicant_rows = validation.drop_duplicates("applicant_id").reset_index(drop=True)
    applicants = applicant_rows.loc[
        :, ["applicant_id", "annual_income", "credit_score", "employment_years", "liquid_assets"]
    ]
    loans = applicant_rows.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]]
    predictions = predict_ml_applicant_risk(fitted.estimator, applicants, loans)
    assert len(predictions) == len(applicants)
    assert predictions["applicant_id"].is_unique
    assert predictions["repayment_probability_ml"].between(0, 1).all()


def test_ml_policy_schema_is_truth_free_and_economics_use_ml_rho() -> None:
    applicants = pd.DataFrame({"applicant_id": [1]})
    loans = make_fixed_loan_options(applicants)
    predictions = pd.DataFrame(
        {"applicant_id": [1], "repayment_probability_ml": [0.9]}
    )
    policy = build_policy_assessments(
        loans,
        predictions,
        policy_id="ml_histgb_v1",
        probability_column="repayment_probability_ml",
    )
    assert tuple(policy.columns) == POLICY_ASSESSMENTS_COLUMNS
    assert not any("true" in column for column in policy.columns)
    assert policy.loc[0, "repayment_probability_base"] == pytest.approx(0.9)
    assert policy.loc[0, "repayment_probability_used"] == pytest.approx(0.9)
    assert policy.loc[0, "expected_receipts_perceived"] == pytest.approx(119.7)
    assert policy.loc[0, "expected_profit_perceived"] == pytest.approx(14.7)


def test_break_even_disagreement_table_has_correct_counts_and_truth() -> None:
    applicants = pd.DataFrame({"applicant_id": [1, 2, 3, 4]})
    loans = make_fixed_loan_options(applicants)
    traditional_probability = pd.DataFrame(
        {"applicant_id": [1, 2, 3, 4], "p": [0.9, 0.9, 0.7, 0.7]}
    )
    ml_probability = pd.DataFrame(
        {"applicant_id": [1, 2, 3, 4], "p": [0.9, 0.7, 0.9, 0.7]}
    )
    truth = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4],
            "repayment_probability_per_period_true": [0.9, 0.9, 0.7, 0.7],
        }
    )
    traditional = build_policy_assessments(
        loans, traditional_probability, policy_id="traditional", probability_column="p"
    )
    ml = build_policy_assessments(
        loans, ml_probability, policy_id="ml", probability_column="p"
    )
    table = break_even_disagreement_table(traditional, ml, loans, truth).set_index(
        ["traditional_classification", "ml_classification"]
    )
    assert table.loc[("positive", "positive"), "n_applicants"] == 1
    assert table.loc[("positive", "nonpositive"), "n_applicants"] == 1
    assert table.loc[("nonpositive", "positive"), "n_applicants"] == 1
    assert table.loc[("nonpositive", "nonpositive"), "n_applicants"] == 1
    assert table.loc[("positive", "nonpositive"), "mean_true_expected_profit"] > 0
    assert table.loc[("nonpositive", "positive"), "mean_true_expected_profit"] < 0


def test_group_audit_and_calibration_bins_are_computed_from_declared_units() -> None:
    applicants = pd.DataFrame(
        {"applicant_id": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]}
    )
    predictions = pd.DataFrame(
        {"applicant_id": [1, 2, 3, 4], "estimated": [0.6, 0.8, 0.7, 0.9]}
    )
    truth = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4],
            "repayment_probability_per_period_true": [0.5, 0.9, 0.6, 1.0],
        }
    )
    history = pd.DataFrame(
        {
            "applicant_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "paid_this_period": [1, 0, 1, 1, 1, 0, 1, 1],
        }
    )
    audit = group_risk_audit(applicants, predictions, truth, "estimated").set_index("group")
    assert audit.loc["A", "mean_prediction_error"] == pytest.approx(0.0)
    assert audit.loc["A", "mae"] == pytest.approx(0.1)
    calibration = calibration_table(history, predictions, truth, "estimated", n_bins=2)
    assert calibration["n_applicants"].sum() == 4
    assert calibration["n_payment_rows"].sum() == 8
    assert calibration["observed_payment_rate"].between(0, 1).all()


def test_oracle_reference_includes_average_precision() -> None:
    history = pd.DataFrame(
        {"applicant_id": [1, 1, 2, 2], "paid_this_period": [1, 1, 1, 0]}
    )
    oracle = pd.DataFrame({"applicant_id": [1, 2], "oracle": [0.9, 0.6]})
    metrics = realized_label_metrics(history, oracle, "oracle")
    expected = average_precision_score([1, 1, 1, 0], [0.9, 0.9, 0.6, 0.6])
    assert metrics["average_precision"] == pytest.approx(expected)

