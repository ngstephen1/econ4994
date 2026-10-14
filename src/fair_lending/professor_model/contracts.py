"""Professor-inspired income-linked contracts; no risk or lending decisions.

The growing scheduled cash flows are a stylized working rule, not mortgage
amortization. r is per period; no calendar duration is inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fair_lending.professor_model.bank_regression import validate_ids

PAYMENT_RULE_ID = "professor_income_compound_schedule_v1"
CONTRACT_COLUMNS = ("applicant_id", "Y", "ell_i", "L_i", "T", "theta", "r", "payment_rule_id")


def validate_terms(T: int, theta: float, r: float) -> None:
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError("T must be a positive integer")
    for name, value, allow_zero in (("theta", theta, False), ("r", r, True)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def make_loan_contracts(applicants: pd.DataFrame, *, T: int, theta: float, r: float) -> pd.DataFrame:
    """One request per existing applicant: ell=theta*Y and L=T*ell."""
    validate_ids(applicants, "contract applicants")
    validate_terms(T, theta, r)
    income = applicants.Y.to_numpy(dtype=float)
    if len(income) == 0 or not np.isfinite(income).all() or not (income > 0).all():
        raise ValueError("contracts require finite positive annual income")
    with np.errstate(over="raise", invalid="raise"):
        ell = theta * income
        principal = T * ell
    result = pd.DataFrame({
        "applicant_id": applicants.applicant_id.to_numpy(), "Y": income,
        "ell_i": ell, "L_i": principal, "T": int(T), "theta": float(theta),
        "r": float(r), "payment_rule_id": PAYMENT_RULE_ID,
    }).loc[:, CONTRACT_COLUMNS]
    validate_contracts(result)
    return result


def validate_contracts(contracts: pd.DataFrame) -> None:
    validate_ids(contracts, "contracts")
    if tuple(contracts.columns) != CONTRACT_COLUMNS or len(contracts) == 0:
        raise ValueError("contracts require the exact schema and nonempty rows")
    if not contracts.payment_rule_id.eq(PAYMENT_RULE_ID).all():
        raise ValueError("unsupported payment rule")
    if not np.isfinite(contracts.loc[:, CONTRACT_COLUMNS[1:-1]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite contract values")
    if not (contracts.Y > 0).all() or not (contracts.ell_i > 0).all() or not (contracts.L_i > 0).all():
        raise ValueError("income, ell and principal must be positive")
    # This baseline has common terms, but heterogeneous income-linked requests.
    for column in ("T", "theta", "r"):
        if contracts[column].nunique() != 1:
            raise ValueError("working baseline requires common contract terms")
    T = contracts["T"].iloc[0]
    if not float(T).is_integer():
        raise ValueError("T must be an integer")
    validate_terms(int(T), float(contracts.theta.iloc[0]), float(contracts.r.iloc[0]))
    if not np.allclose(contracts.ell_i, contracts.theta * contracts.Y, rtol=1e-14, atol=0) or not np.allclose(contracts.L_i, contracts["T"] * contracts.ell_i, rtol=1e-14, atol=0):
        raise ValueError("contract principal/base-payment identity mismatch")


def scheduled_payments(contracts: pd.DataFrame) -> np.ndarray:
    """Return applicant × period cash flows, independent of all risk values.

    Columns correspond to periods 1,...,T (not 0,...,T-1). The 20,000×30
    working schedule is small enough to calculate in memory, not persist.
    """
    validate_contracts(contracts)
    periods = np.arange(1, int(contracts["T"].iloc[0]) + 1)
    with np.errstate(over="raise", invalid="raise"):
        payments = contracts.ell_i.to_numpy()[:, None] * (1 + contracts.r.iloc[0]) ** periods
    if not np.isfinite(payments).all():
        raise ValueError("nonfinite scheduled payments")
    return payments
