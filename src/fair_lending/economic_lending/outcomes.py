"""Observed repayment-history generation for Version 2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fair_lending.economic_lending.contracts import scheduled_payment_amount
from fair_lending.economic_lending.schema import LOAN_OUTCOMES_COLUMNS


def simulate_repayment_outcomes(
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate absorbing default using conditional per-period repayment truth.

    Random draws are generated only for loans still active at each period; no
    repayment outcomes are drawn after the first missed payment.
    """

    loans = loan_options.merge(
        simulation_truth.loc[
            :, ["applicant_id", "repayment_probability_per_period_true"]
        ],
        on="applicant_id",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    n_rows = len(loans)
    term = loans["term_periods"].to_numpy(dtype=int)
    rho = loans["repayment_probability_per_period_true"].to_numpy(dtype=float)
    alive = loans["requested_principal"].to_numpy(dtype=float) > 0
    default_period = np.full(n_rows, np.nan)

    for period in range(1, int(term.max(initial=0)) + 1):
        active_indices = np.flatnonzero(alive & (term >= period))
        if not len(active_indices):
            continue
        made_payment = rng.random(len(active_indices)) <= rho[active_indices]
        defaulted = active_indices[~made_payment]
        default_period[defaulted] = period
        alive[defaulted] = False
        completed_at_period = active_indices[made_payment & (term[active_indices] == period)]
        alive[completed_at_period] = False

    completed = np.isnan(default_period)
    periods_paid = np.where(completed, term, default_period - 1).astype(int)
    payment = scheduled_payment_amount(loans).to_numpy(dtype=float)
    realized_receipts = payment * periods_paid
    principal = loans["requested_principal"].to_numpy(dtype=float)
    transaction_cost = np.where(
        principal > 0,
        loans["transaction_cost"].to_numpy(dtype=float),
        0.0,
    )
    outcomes = pd.DataFrame(
        {
            "applicant_id": loans["applicant_id"].to_numpy(),
            "default_period": pd.array(default_period, dtype="Int64"),
            "completed_all_payments": completed,
            "realized_total_receipts": realized_receipts,
            "realized_profit": realized_receipts - principal - transaction_cost,
        }
    )
    zero_loan = principal == 0
    outcomes.loc[zero_loan, ["realized_total_receipts", "realized_profit"]] = 0.0
    return outcomes.loc[:, LOAN_OUTCOMES_COLUMNS]


def simulate_repayment_outcomes_from_uniforms(
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    uniforms: np.ndarray,
) -> pd.DataFrame:
    """Apply a fixed applicant-period uniform matrix with absorbing default.

    Supplying the same matrix to two risk worlds creates paired counterfactual
    outcomes without reusing either world's realized labels.
    """

    loans = loan_options.merge(
        simulation_truth.loc[:, ["applicant_id", "repayment_probability_per_period_true"]],
        on="applicant_id",
        validate="one_to_one",
    ).reset_index(drop=True)
    n_rows = len(loans)
    term = loans["term_periods"].to_numpy(dtype=int)
    if uniforms.shape != (n_rows, int(term.max(initial=0))):
        raise ValueError("uniform matrix must have shape (n_loans, max_term_periods)")
    rho = loans["repayment_probability_per_period_true"].to_numpy(dtype=float)
    alive = loans["requested_principal"].to_numpy(dtype=float) > 0
    default_period = np.full(n_rows, np.nan)
    for period in range(1, uniforms.shape[1] + 1):
        active = np.flatnonzero(alive & (term >= period))
        made = uniforms[active, period - 1] <= rho[active]
        defaulted = active[~made]
        default_period[defaulted] = period
        alive[defaulted] = False
        completed = active[made & (term[active] == period)]
        alive[completed] = False
    completed = np.isnan(default_period)
    periods_paid = np.where(completed, term, default_period - 1).astype(int)
    payment = scheduled_payment_amount(loans).to_numpy(dtype=float)
    receipts = payment * periods_paid
    principal = loans["requested_principal"].to_numpy(dtype=float)
    cost = np.where(principal > 0, loans["transaction_cost"].to_numpy(dtype=float), 0.0)
    result = pd.DataFrame(
        {
            "applicant_id": loans["applicant_id"].to_numpy(),
            "default_period": pd.array(default_period, dtype="Int64"),
            "completed_all_payments": completed,
            "realized_total_receipts": receipts,
            "realized_profit": receipts - principal - cost,
        }
    )
    result.loc[principal == 0, ["realized_total_receipts", "realized_profit"]] = 0.0
    return result.loc[:, LOAN_OUTCOMES_COLUMNS]
