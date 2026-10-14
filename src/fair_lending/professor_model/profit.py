"""Survival-weighted receipts with separate bank/evaluator entry points.

No model is fitted, no scores are clipped here, and no request is approved.
These are potential values if each individual request were issued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fair_lending.professor_model.bank_regression import MODEL_ID, validate_ids
from fair_lending.professor_model.contracts import scheduled_payments

BANK_PROFIT_COLUMNS = ("applicant_id", "model_id", "bank_PD_bounded", "expected_receipts_bank", "expected_profit_bank")
TRUE_PROFIT_COLUMNS = ("applicant_id", "true_PD", "expected_receipts_true", "expected_profit_true")
BANK_RISK_COLUMNS = ("applicant_id", "model_id", "bank_PD_bounded")
TRUE_RISK_COLUMNS = ("applicant_id", "true_PD")
PREVIEW_SCOPE = "EVALUATOR_ONLY_NOT_FOR_LENDER_DECISIONS"
PREVIEW_COLUMNS = (
    "information_scope", "applicant_id", "B", "Y", "ell_i", "L_i", "model_id",
    "bank_PD_raw", "bank_PD_bounded", "true_PD", "expected_receipts_bank",
    "expected_receipts_true", "expected_profit_bank", "expected_profit_true", "profit_estimation_error",
)


def payment_survival(default_probability: np.ndarray, T: int) -> np.ndarray:
    """P(receive payment t)=(1-p)^t, including survival of period t itself."""
    probability = np.asarray(default_probability, dtype=float)
    if probability.ndim != 1 or len(probability) == 0 or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("default probabilities must be a finite vector in [0,1]")
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError("T must be a positive integer")
    return (1 - probability[:, None]) ** np.arange(1, T + 1)


def expected_cashflows(contracts: pd.DataFrame, default_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Explicit 30-term sum avoids division by q-1 at the break-even point."""
    schedule = scheduled_payments(contracts)
    survival = payment_survival(default_probability, schedule.shape[1])
    if survival.shape != schedule.shape:
        raise ValueError("exactly one default probability is required per contract")
    receipts = np.sum(schedule * survival, axis=1)
    profit = receipts - contracts.L_i.to_numpy()  # principal once; no other costs
    if not np.isfinite(receipts).all() or not np.isfinite(profit).all() or (receipts < 0).any():
        raise ValueError("invalid expected cash flows")
    return receipts, profit


def break_even_probability(r: float) -> float:
    """For nonnegative r, q=(1-p)(1+r)=1 implies p=r/(1+r)."""
    if isinstance(r, bool) or not isinstance(r, (int, float, np.number)) or not np.isfinite(r) or r < 0:
        raise ValueError("r must be finite and nonnegative")
    return float(r / (1 + r))


def _risk_join(contracts: pd.DataFrame, risk: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    validate_ids(contracts, "contracts")
    validate_ids(risk, "risk view")
    if tuple(risk.columns) != columns or set(risk.applicant_id) != set(contracts.applicant_id):
        raise ValueError("risk view requires exact schema and matching contract IDs")
    return contracts[["applicant_id"]].merge(risk, on="applicant_id", validate="one_to_one")


def assess_bank_profit(contracts: pd.DataFrame, bank_risk: pd.DataFrame) -> pd.DataFrame:
    """The bank accepts bounded estimates only; truth/raw scores are not inputs."""
    risk = _risk_join(contracts, bank_risk, BANK_RISK_COLUMNS)
    if not risk.model_id.eq(MODEL_ID).all():
        raise ValueError("unexpected frozen bank model")
    receipts, profit = expected_cashflows(contracts, risk.bank_PD_bounded.to_numpy())
    return risk.assign(expected_receipts_bank=receipts, expected_profit_bank=profit).loc[:, BANK_PROFIT_COLUMNS]


def assess_true_profit(contracts: pd.DataFrame, truth_risk: pd.DataFrame) -> pd.DataFrame:
    """Evaluator only: identical accounting, evaluated at saved true_PD."""
    risk = _risk_join(contracts, truth_risk, TRUE_RISK_COLUMNS)
    receipts, profit = expected_cashflows(contracts, risk.true_PD.to_numpy())
    return risk.assign(expected_receipts_true=receipts, expected_profit_true=profit).loc[:, TRUE_PROFIT_COLUMNS]


def economic_diagnostics(
    evaluation: pd.DataFrame, contracts: pd.DataFrame, bank: pd.DataFrame, truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Overall/B-group summaries and distribution rows; every result is evaluator-only."""
    for name, frame in (("evaluation", evaluation), ("contracts", contracts), ("bank", bank), ("truth", truth)):
        validate_ids(frame, name)
        if set(frame.applicant_id) != set(evaluation.applicant_id):
            raise ValueError("diagnostic input IDs do not match")
    if tuple(bank.columns) != BANK_PROFIT_COLUMNS or tuple(truth.columns) != TRUE_PROFIT_COLUMNS:
        raise ValueError("profit artifact schema mismatch")
    joined = evaluation[["applicant_id", "B", "bank_PD_raw"]].merge(contracts, on="applicant_id", validate="one_to_one").merge(bank, on="applicant_id", validate="one_to_one").merge(truth, on="applicant_id", validate="one_to_one").sort_values("applicant_id").reset_index(drop=True)
    joined["profit_estimation_error"] = joined.expected_profit_bank - joined.expected_profit_true
    scopes = [("overall", joined)] + [(f"B={int(group)}", frame) for group, frame in joined.groupby("B", sort=True)]
    summaries, distributions = [], []
    variables = ("Y", "ell_i", "L_i", "expected_receipts_bank", "expected_receipts_true", "expected_profit_bank", "expected_profit_true", "profit_estimation_error")
    for scope, frame in scopes:
        error = frame.profit_estimation_error.to_numpy()
        bank_positive = frame.expected_profit_bank > 0
        true_positive = frame.expected_profit_true > 0
        summaries.append({
            "scope": scope, "n_applicants": len(frame),
            **{f"{column}_{stat}": float(getattr(frame[column], stat)()) for column in ("Y", "ell_i", "L_i") for stat in ("mean", "median")},
            "total_requested_principal": float(frame.L_i.sum()),
            **{f"{column}_mean": float(frame[column].mean()) for column in variables[3:7]},
            "fraction_profit_bank_positive": float(bank_positive.mean()),
            "fraction_profit_true_positive": float(true_positive.mean()),
            "fraction_profit_sign_disagreement": float((bank_positive != true_positive).mean()),
            "mean_profit_estimation_error": float(error.mean()),
            "profit_MAE": float(np.abs(error).mean()), "profit_RMSE": float(np.sqrt(np.mean(error ** 2))),
        })
        for column in variables:
            values = frame[column].to_numpy()
            distributions.append({
                "scope": scope, "variable": column, "unit": "synthetic_USD_per_year" if column == "Y" else "synthetic_USD", "n": len(values),
                "mean": float(values.mean()), "median": float(np.median(values)),
                "sample_SD": float(values.std(ddof=1)) if len(values) > 1 else None,
                "minimum": float(values.min()), "maximum": float(values.max()),
                **{f"p{q:02d}": float(np.quantile(values, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)},
            })
    joined.insert(0, "information_scope", PREVIEW_SCOPE)
    return pd.DataFrame(summaries), pd.DataFrame(distributions), joined.loc[:, PREVIEW_COLUMNS]
