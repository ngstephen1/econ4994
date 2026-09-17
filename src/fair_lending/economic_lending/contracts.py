"""Loan contract and payment schedule utilities."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.economic_lending.schema import (
    LOAN_OPTIONS_COLUMNS,
    PAYMENT_SCHEDULE_COLUMNS,
    BaselineEconomicParameters,
    WORKING_PAYMENT_RULE_ID,
)


def make_fixed_loan_options(
    applicants: pd.DataFrame,
    params: BaselineEconomicParameters | None = None,
) -> pd.DataFrame:
    """Assign one fixed requested loan option to each applicant."""

    params = params or BaselineEconomicParameters()
    first_period_payment = (
        params.requested_principal
        * (1.0 + params.periodic_interest_rate * params.term_periods)
        / params.term_periods
        if params.requested_principal > 0 and params.term_periods > 0
        else 0.0
    )
    options = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"].to_numpy(),
            "option_id": [f"{value}:fixed" for value in applicants["applicant_id"]],
            "requested_principal": params.requested_principal,
            "periodic_interest_rate": params.periodic_interest_rate,
            "term_periods": params.term_periods,
            "transaction_cost": params.transaction_cost,
            "payment_rule_id": params.payment_rule_id,
            "first_period_payment": first_period_payment,
            "first_period_dti": np.nan,
            "requested_ltv": np.nan,
        }
    )
    return options.loc[:, LOAN_OPTIONS_COLUMNS]


def make_requested_loan_options(
    applicants: pd.DataFrame,
    contract_config: dict[str, Any],
) -> pd.DataFrame:
    """Create one configured contract for each applicant's requested amount."""

    principal = applicants["requested_loan_amount"].to_numpy(dtype=float)
    term = int(contract_config["term_periods"])
    rate = float(contract_config["periodic_interest_rate"])
    if term <= 0:
        raise ValueError("term_periods must be positive")
    first_payment = principal * (1.0 + rate * term) / term
    monthly_income = applicants["annual_income"].to_numpy(dtype=float) / 12.0
    first_period_dti = (
        applicants["existing_monthly_debt"].to_numpy(dtype=float) + first_payment
    ) / monthly_income
    requested_ltv = principal / applicants["property_value"].to_numpy(dtype=float)
    options = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"].to_numpy(),
            "option_id": [f"{value}:requested" for value in applicants["applicant_id"]],
            "requested_principal": principal,
            "periodic_interest_rate": rate,
            "term_periods": term,
            "transaction_cost": float(contract_config["transaction_cost"]),
            "payment_rule_id": contract_config["payment_rule_id"],
            "first_period_payment": first_payment,
            "first_period_dti": first_period_dti,
            "requested_ltv": requested_ltv,
        }
    )
    return options.loc[:, LOAN_OPTIONS_COLUMNS]


def scheduled_payment_amount(loan_options: pd.DataFrame) -> pd.Series:
    """Return the equal scheduled payment for each configured loan option."""

    term = loan_options["term_periods"].to_numpy(dtype=float)
    principal = loan_options["requested_principal"].to_numpy(dtype=float)
    rate = loan_options["periodic_interest_rate"].to_numpy(dtype=float)
    result = np.zeros(len(loan_options), dtype=float)
    positive = (principal > 0) & (term > 0)
    result[positive] = principal[positive] * (1.0 + rate[positive] * term[positive]) / term[positive]
    return pd.Series(result, index=loan_options.index, name="scheduled_payment")


def payment_schedule(loan_options: pd.DataFrame) -> pd.DataFrame:
    """Create an equal-payment schedule for each requested loan.

    The working default total promised receipts are
    principal * (1 + periodic_interest_rate * term_periods). The total is split
    equally across periods. This module owns the formula so later prompts can
    replace it without changing repayment or allocation code.
    """

    rows: list[dict[str, object]] = []
    for loan in loan_options.itertuples(index=False):
        if loan.term_periods < 0:
            raise ValueError("term_periods must be nonnegative")
        if loan.requested_principal < 0:
            raise ValueError("requested_principal must be nonnegative")
        if loan.payment_rule_id != WORKING_PAYMENT_RULE_ID:
            raise ValueError(f"unsupported payment_rule_id: {loan.payment_rule_id}")
        if loan.term_periods == 0 or loan.requested_principal == 0:
            continue
        total_promised = loan.requested_principal * (1.0 + loan.periodic_interest_rate * loan.term_periods)
        scheduled_payment = total_promised / loan.term_periods
        for period in range(1, int(loan.term_periods) + 1):
            rows.append(
                {
                    "applicant_id": loan.applicant_id,
                    "period": period,
                    "scheduled_payment": float(scheduled_payment),
                    "payment_rule_id": loan.payment_rule_id,
                }
            )
    return pd.DataFrame(rows, columns=PAYMENT_SCHEDULE_COLUMNS)


def total_promised_receipts(schedule: pd.DataFrame) -> pd.Series:
    """Return promised receipts by applicant if every payment is made."""

    if schedule.empty:
        return pd.Series(dtype=float, name="promised_receipts")
    return schedule.groupby("applicant_id", sort=True)["scheduled_payment"].sum()


def realized_receipts(
    schedule: pd.DataFrame,
    applicant_id: int,
    default_period: int | float | None,
) -> float:
    """Compute realized receipts when default stops later payments.

    ``default_period=None`` or positive infinity means full repayment.
    ``default_period=1`` means immediate default before the first scheduled
    payment. A finite default period is the first scheduled payment not made.
    """

    applicant_schedule = schedule.loc[schedule["applicant_id"] == applicant_id]
    if default_period is None or (
        isinstance(default_period, Real)
        and math.isinf(float(default_period))
        and float(default_period) > 0
    ):
        return float(applicant_schedule["scheduled_payment"].sum())
    if (
        not isinstance(default_period, Real)
        or not math.isfinite(float(default_period))
        or float(default_period) < 1
        or not float(default_period).is_integer()
    ):
        raise ValueError("default_period must be a positive integer, None, or positive infinity")
    return float(
        applicant_schedule.loc[
            applicant_schedule["period"] < default_period, "scheduled_payment"
        ].sum()
    )
