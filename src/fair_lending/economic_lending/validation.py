"""Validation helpers for the Version 2 economic lending baseline."""

from __future__ import annotations

import pandas as pd


def require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...], table_name: str) -> None:
    """Raise when a DataFrame does not contain the declared schema columns."""

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def require_unique_applicant_ids(applicants: pd.DataFrame) -> None:
    """Raise unless applicant identifiers are unique."""

    if applicants["applicant_id"].duplicated().any():
        raise ValueError("applicant_id values must be unique")


def require_valid_probabilities(truth: pd.DataFrame) -> None:
    """Raise unless repayment probabilities are in [0, 1]."""

    probability_columns = (
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
    )
    for column in probability_columns:
        probs = truth[column]
        if probs.isna().any() or not ((0 <= probs) & (probs <= 1)).all():
            raise ValueError(f"{column} must be nonmissing and in [0, 1]")
