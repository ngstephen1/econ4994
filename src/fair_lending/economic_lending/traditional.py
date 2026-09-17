"""Traditional lender repayment-risk estimators and perceived economics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fair_lending.economic_lending.history import FORBIDDEN_LENDER_FIELDS
from fair_lending.economic_lending.profit import expected_economics_from_probability
from fair_lending.economic_lending.repayment import (
    TRUE_RISK_PREDICTORS,
    sigmoid,
    transform_observable_risk_features,
)
from fair_lending.economic_lending.schema import POLICY_ASSESSMENTS_COLUMNS


TRADITIONAL_POLICY_ID = "traditional_logit_v1"
TRADITIONAL_LENDER_FEATURES = TRUE_RISK_PREDICTORS
PRIMARY_TRAINING_COHORT = "historical_train"


@dataclass(frozen=True)
class FittedTraditionalModel:
    """Transparent coefficient-only representation of a fitted risk model."""

    model_family: str
    coefficients: dict[str, float]
    features: tuple[str, ...]
    weighted_by_applicant: bool = False

    def predict_transformed(self, transformed: pd.DataFrame) -> np.ndarray:
        score = np.full(len(transformed), self.coefficients["intercept"], dtype=float)
        for feature in self.features:
            score += transformed[feature].to_numpy(dtype=float) * self.coefficients[feature]
        if self.model_family == "logistic_regression":
            return np.asarray(sigmoid(score), dtype=float)
        if self.model_family == "linear_probability_model":
            return score
        raise ValueError(f"unsupported model_family: {self.model_family}")


def lender_design_matrix(
    payment_history: pd.DataFrame,
    true_risk_config: dict[str, Any],
) -> pd.DataFrame:
    """Build the allowlisted lender design matrix from observable fields."""

    leaked = FORBIDDEN_LENDER_FIELDS.intersection(payment_history.columns)
    if leaked:
        raise ValueError(f"hidden truth leaked into lender history: {sorted(leaked)}")
    transformed = transform_observable_risk_features(payment_history, true_risk_config)
    design = transformed.loc[:, TRADITIONAL_LENDER_FEATURES].copy()
    if {"group", "applicant_id", "cohort", "period_index"}.intersection(design.columns):
        raise AssertionError("identifier, cohort, group, or time leaked into design matrix")
    return design


def applicant_row_weights(payment_history: pd.DataFrame) -> pd.Series:
    """Give each applicant total weight one across observed at-risk rows."""

    row_counts = payment_history.groupby("applicant_id")["applicant_id"].transform("size")
    return pd.Series(1.0 / row_counts.to_numpy(dtype=float), index=payment_history.index)


def _fit_model(
    payment_history: pd.DataFrame,
    true_risk_config: dict[str, Any],
    *,
    family: str,
    applicant_weighted: bool,
) -> FittedTraditionalModel:
    cohorts = set(payment_history["cohort"].unique())
    if cohorts != {PRIMARY_TRAINING_COHORT}:
        raise ValueError("model fitting accepts historical_train rows only")
    design = lender_design_matrix(payment_history, true_risk_config)
    exog = sm.add_constant(design, has_constant="add").rename(columns={"const": "intercept"})
    target = payment_history["paid_this_period"].to_numpy(dtype=float)
    weights = None
    if applicant_weighted:
        weights = applicant_row_weights(payment_history).to_numpy(dtype=float)

    if family == "logistic_regression":
        fitted = sm.GLM(
            target,
            exog,
            family=sm.families.Binomial(),
            freq_weights=weights,
        ).fit(maxiter=100, disp=0)
    elif family == "linear_probability_model":
        fitted = sm.WLS(
            target,
            exog,
            weights=np.ones(len(target)) if weights is None else weights,
        ).fit()
    else:
        raise ValueError(f"unsupported family: {family}")
    coefficients = {name: float(value) for name, value in fitted.params.items()}
    return FittedTraditionalModel(
        model_family=family,
        coefficients=coefficients,
        features=TRADITIONAL_LENDER_FEATURES,
        weighted_by_applicant=applicant_weighted,
    )


def fit_traditional_logit(
    training_history: pd.DataFrame,
    true_risk_config: dict[str, Any],
    *,
    applicant_weighted: bool = False,
) -> FittedTraditionalModel:
    """Fit the primary logit or its applicant-weighted sensitivity."""

    return _fit_model(
        training_history,
        true_risk_config,
        family="logistic_regression",
        applicant_weighted=applicant_weighted,
    )


def fit_linear_probability_model(
    training_history: pd.DataFrame,
    true_risk_config: dict[str, Any],
) -> FittedTraditionalModel:
    """Fit the professor-aligned, unbounded LPM sensitivity."""

    return _fit_model(
        training_history,
        true_risk_config,
        family="linear_probability_model",
        applicant_weighted=False,
    )


def predict_applicant_risk(
    model: FittedTraditionalModel,
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    true_risk_config: dict[str, Any],
    *,
    probability_column: str = "repayment_probability_traditional",
) -> pd.DataFrame:
    """Produce one constant per-period risk estimate per applicant/request."""

    source = applicants.merge(
        loan_options.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]],
        on="applicant_id",
        validate="one_to_one",
    )
    transformed = transform_observable_risk_features(source, true_risk_config)
    return pd.DataFrame(
        {
            "applicant_id": source["applicant_id"].to_numpy(),
            probability_column: model.predict_transformed(transformed),
        }
    )


def build_policy_assessments(
    evaluation_loans: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy_id: str = TRADITIONAL_POLICY_ID,
    probability_column: str = "repayment_probability_traditional",
) -> pd.DataFrame:
    """Create truth-free perceived economic assessments for evaluation loans."""

    economics = expected_economics_from_probability(
        evaluation_loans,
        predictions,
        probability_column,
        "expected_receipts_perceived",
        "expected_profit_perceived",
    )
    assessed = (
        evaluation_loans.loc[:, ["applicant_id", "option_id"]]
        .merge(predictions, on="applicant_id", validate="one_to_one")
        .merge(economics, on="applicant_id", validate="one_to_one")
    )
    assessed["policy_id"] = policy_id
    assessed["repayment_probability_base"] = assessed[probability_column]
    assessed["repayment_probability_used"] = assessed[probability_column]
    assessed["assessment_status"] = "assessed"
    return assessed.loc[:, POLICY_ASSESSMENTS_COLUMNS]
