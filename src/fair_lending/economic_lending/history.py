"""Lender-observable at-risk payment histories."""

from __future__ import annotations

import numpy as np
import pandas as pd


LENDER_HISTORY_COLUMNS = (
    "applicant_id",
    "cohort",
    "group",
    "period_index",
    "paid_this_period",
    "annual_income",
    "credit_score",
    "employment_years",
    "liquid_assets",
    "existing_monthly_debt",
    "property_value",
    "requested_loan_amount",
    "requested_ltv",
    "first_period_dti",
)

FORBIDDEN_LENDER_FIELDS = frozenset(
    {
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
        "expected_receipts_true",
        "expected_profit_true",
        "future_default_period_potential",
        "simulation_latent_factor",
    }
)


def build_at_risk_payment_history(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    loan_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per applicant-period while the loan remains at risk."""

    source = (
        applicants.merge(
            loan_options.loc[
                :,
                ["applicant_id", "term_periods", "requested_ltv", "first_period_dti"],
            ],
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(
            loan_outcomes.loc[
                :, ["applicant_id", "default_period", "completed_all_payments"]
            ],
            on="applicant_id",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    default_period = source["default_period"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    term = source["term_periods"].to_numpy(dtype=int)
    observed_periods = np.where(np.isnan(default_period), term, default_period).astype(int)
    if (observed_periods <= 0).any():
        raise ValueError("each loan must contribute at least one at-risk period")

    repeated_index = np.repeat(np.arange(len(source)), observed_periods)
    starts = np.repeat(np.cumsum(observed_periods) - observed_periods, observed_periods)
    period_index = np.arange(observed_periods.sum()) - starts + 1
    repeated_default = default_period[repeated_index]
    paid = np.isnan(repeated_default) | (period_index < repeated_default)

    history = pd.DataFrame(
        {
            "applicant_id": source["applicant_id"].to_numpy()[repeated_index],
            "cohort": source["cohort"].astype(str).to_numpy()[repeated_index],
            "group": source["group"].astype(str).to_numpy()[repeated_index],
            "period_index": period_index.astype(np.int16),
            "paid_this_period": paid.astype(np.int8),
        }
    )
    observable_fields = [
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "requested_loan_amount",
        "requested_ltv",
        "first_period_dti",
    ]
    for field in observable_fields:
        history[field] = source[field].to_numpy()[repeated_index]
    return history.loc[:, LENDER_HISTORY_COLUMNS]
