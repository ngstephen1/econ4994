"""Expected receipt and profit accounting."""

from __future__ import annotations

import pandas as pd
import numpy as np

from fair_lending.economic_lending.contracts import scheduled_payment_amount
from fair_lending.economic_lending.repayment import survival_probability
from fair_lending.economic_lending.schema import TRUE_ECONOMIC_ASSESSMENT_COLUMNS


def expected_receipts(payment_schedule: pd.DataFrame, simulation_truth: pd.DataFrame) -> pd.DataFrame:
    """Expected receipts from survival-weighted scheduled payments.

    The conditional probability of making the next payment is constant within
    a contract, so the probability of receiving payment ``t`` is ``rho ** t``.
    """

    cash_flows = payment_schedule.merge(
        simulation_truth.loc[
            :, ["applicant_id", "repayment_probability_per_period_true"]
        ],
        on="applicant_id",
        how="left",
        validate="many_to_one",
    )
    cash_flows["survival_probability_true"] = survival_probability(
        cash_flows["repayment_probability_per_period_true"].to_numpy(),
        cash_flows["period"].to_numpy(),
    )
    cash_flows["expected_scheduled_receipt_true"] = (
        cash_flows["survival_probability_true"] * cash_flows["scheduled_payment"]
    )
    totals = (
        cash_flows.groupby("applicant_id", sort=True)["expected_scheduled_receipt_true"]
        .sum()
        .rename("expected_receipts_true")
    )
    receipts = simulation_truth.loc[:, ["applicant_id"]].merge(
        totals,
        how="left",
        left_on="applicant_id",
        right_index=True,
        validate="one_to_one",
    )
    receipts["expected_receipts_true"] = (
        receipts["expected_receipts_true"].astype("float64").fillna(0.0)
    )
    return receipts.loc[:, ["applicant_id", "expected_receipts_true"]]


def expected_profit(
    loan_options: pd.DataFrame,
    payment_schedule: pd.DataFrame,
    simulation_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Expected monetary profit for each fixed requested loan."""

    receipts = expected_receipts(payment_schedule, simulation_truth)
    assessed = loan_options.merge(receipts, on="applicant_id", how="left")
    assessed["expected_receipts_true"] = assessed["expected_receipts_true"].fillna(0.0)
    positive_loan = assessed["requested_principal"] > 0
    charged_transaction_cost = assessed["transaction_cost"].where(positive_loan, 0.0)
    assessed["expected_profit_true"] = (
        assessed["expected_receipts_true"]
        - assessed["requested_principal"]
        - charged_transaction_cost
    )
    return assessed.loc[:, TRUE_ECONOMIC_ASSESSMENT_COLUMNS]


def expected_economics_from_probability(
    loan_options: pd.DataFrame,
    probabilities: pd.DataFrame,
    probability_column: str,
    receipts_column: str,
    profit_column: str,
) -> pd.DataFrame:
    """Calculate expected receipts and profit from any per-period probability."""

    assessed = loan_options.merge(
        probabilities.loc[:, ["applicant_id", probability_column]],
        on="applicant_id",
        how="inner",
        validate="one_to_one",
    )
    rho = assessed[probability_column].to_numpy(dtype=float)
    if np.isnan(rho).any() or ((rho < 0) | (rho > 1)).any():
        raise ValueError(f"{probability_column} must be in [0, 1]")
    term = assessed["term_periods"].to_numpy(dtype=int)
    expected_payment_count = np.empty(len(assessed), dtype=float)
    certain = rho == 1.0
    expected_payment_count[certain] = term[certain]
    uncertain = ~certain
    expected_payment_count[uncertain] = (
        rho[uncertain] * (1.0 - np.power(rho[uncertain], term[uncertain]))
        / (1.0 - rho[uncertain])
    )
    payment = scheduled_payment_amount(assessed).to_numpy(dtype=float)
    expected_receipts = payment * expected_payment_count
    principal = assessed["requested_principal"].to_numpy(dtype=float)
    cost = np.where(
        principal > 0,
        assessed["transaction_cost"].to_numpy(dtype=float),
        0.0,
    )
    return pd.DataFrame(
        {
            "applicant_id": assessed["applicant_id"].to_numpy(),
            receipts_column: expected_receipts,
            profit_column: expected_receipts - principal - cost,
        }
    )


def expected_profit_from_options(
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Compute true expected receipts and profit without materializing schedules."""

    return expected_economics_from_probability(
        loan_options,
        simulation_truth,
        "repayment_probability_per_period_true",
        "expected_receipts_true",
        "expected_profit_true",
    )
