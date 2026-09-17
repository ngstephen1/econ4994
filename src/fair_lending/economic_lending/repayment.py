"""Hidden true repayment model for the Version 2 baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fair_lending.economic_lending.schema import SIMULATION_TRUTH_COLUMNS


TRUE_RISK_PREDICTORS = (
    "log_annual_income",
    "credit_score_50",
    "employment_years_10",
    "log_liquid_assets",
    "first_period_dti_10pp",
    "requested_ltv_10pp",
)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic transform."""

    return 1.0 / (1.0 + np.exp(-x))


def survival_probability(
    repayment_probability_per_period: np.ndarray | float,
    period: np.ndarray | int,
) -> np.ndarray | float:
    """Return the probability of making every scheduled payment through a period."""

    probability = np.asarray(repayment_probability_per_period, dtype=float)
    periods = np.asarray(period)
    if np.isnan(probability).any() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("repayment probabilities must be in [0, 1]")
    if ((periods < 0) | (periods != np.floor(periods))).any():
        raise ValueError("period must contain nonnegative integers")
    result = np.power(probability, periods)
    if np.ndim(result) == 0:
        return float(result)
    return result


def true_risk_design_matrix(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
) -> pd.DataFrame:
    """Build the declared true-risk predictors without group or outcomes."""

    joined = applicants.merge(
        loan_options.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]],
        on="applicant_id",
        how="inner",
        validate="one_to_one",
    )
    return joined


def transformed_true_risk_predictors(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    true_risk_config: dict,
) -> pd.DataFrame:
    """Apply the configured, documented transformations used by true risk."""

    source = true_risk_design_matrix(applicants, loan_options)
    transformed = transform_observable_risk_features(source, true_risk_config)
    transformed.insert(0, "applicant_id", source["applicant_id"].to_numpy())
    return transformed.loc[:, ["applicant_id", *TRUE_RISK_PREDICTORS]]


def transform_observable_risk_features(
    source: pd.DataFrame,
    true_risk_config: dict,
) -> pd.DataFrame:
    """Apply shared transformations using observable raw fields only."""

    specs = true_risk_config["predictors"]
    transformed = pd.DataFrame(index=source.index)
    transformed["log_annual_income"] = np.log(
        source["annual_income"] / specs["log_annual_income"]["center"]
    )
    transformed["credit_score_50"] = (
        source["credit_score"] - specs["credit_score_50"]["center"]
    ) / specs["credit_score_50"]["scale"]
    transformed["employment_years_10"] = (
        source["employment_years"] - specs["employment_years_10"]["center"]
    ) / specs["employment_years_10"]["scale"]
    transformed["log_liquid_assets"] = np.log(
        (source["liquid_assets"] + specs["log_liquid_assets"]["offset"])
        / specs["log_liquid_assets"]["center"]
    )
    transformed["first_period_dti_10pp"] = (
        source["first_period_dti"] - specs["first_period_dti_10pp"]["center"]
    ) / specs["first_period_dti_10pp"]["scale"]
    transformed["requested_ltv_10pp"] = (
        source["requested_ltv"] - specs["requested_ltv_10pp"]["center"]
    ) / specs["requested_ltv_10pp"]["scale"]
    return transformed.loc[:, TRUE_RISK_PREDICTORS]


def true_repayment_probabilities(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    true_risk_config: dict,
) -> pd.DataFrame:
    """Calculate the hidden true repayment probabilities for the population.

    The working DGP is logistic-linear in the declared transformed predictors.
    Group and cohort are deliberately absent.
    """

    transformed = transformed_true_risk_predictors(
        applicants, loan_options, true_risk_config
    )
    score = np.full(len(transformed), float(true_risk_config["intercept"]))
    for predictor in TRUE_RISK_PREDICTORS:
        score += (
            transformed[predictor].to_numpy(dtype=float)
            * float(true_risk_config["predictors"][predictor]["coefficient"])
        )
    probabilities = sigmoid(score)
    per_period = pd.DataFrame(
        {
            "applicant_id": transformed["applicant_id"].to_numpy(),
            "repayment_probability_per_period_true": probabilities,
        }
    )
    truth = per_period.merge(
        loan_options.loc[:, ["applicant_id", "term_periods"]],
        on="applicant_id",
        how="inner",
        validate="one_to_one",
    )
    truth["full_repayment_probability_true"] = survival_probability(
        truth["repayment_probability_per_period_true"].to_numpy(),
        truth["term_periods"].to_numpy(),
    )
    return truth.loc[:, SIMULATION_TRUTH_COLUMNS]


def true_repayment_probabilities_from_score(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate fixture truth from a directly supplied repayment score.

    ``sigmoid(repayment_score)`` is the constant conditional probability of
    making the next payment. This development helper preserves the four-row
    hand-check fixture and is not the calibrated population DGP.
    """

    probabilities = sigmoid(applicants["repayment_score"].to_numpy(dtype=float))
    per_period = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"].to_numpy(),
            "repayment_probability_per_period_true": probabilities,
        }
    )
    truth = per_period.merge(
        loan_options.loc[:, ["applicant_id", "term_periods"]],
        on="applicant_id",
        how="inner",
        validate="one_to_one",
    )
    truth["full_repayment_probability_true"] = survival_probability(
        truth["repayment_probability_per_period_true"].to_numpy(),
        truth["term_periods"].to_numpy(),
    )
    return truth.loc[:, SIMULATION_TRUTH_COLUMNS]
