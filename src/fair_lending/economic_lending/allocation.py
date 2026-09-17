"""Fixed-request lending allocation for the Version 2 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from fair_lending.economic_lending.schema import LENDING_DECISIONS_COLUMNS


@dataclass(frozen=True)
class AllocationResult:
    """Budget-constrained fixed-request allocation result."""

    decisions: pd.DataFrame
    selected_applicant_ids: tuple[int | str, ...]
    total_funded_principal: float
    total_expected_profit_true: float
    unused_funds: float


def _candidate_subsets(
    applicant_ids: list[int | str],
) -> list[tuple[int | str, ...]]:
    subsets: list[tuple[int | str, ...]] = []
    for size in range(len(applicant_ids) + 1):
        subsets.extend(combinations(applicant_ids, size))
    return subsets


def enumerate_allocations(
    loan_options: pd.DataFrame,
    policy_assessments: pd.DataFrame,
    bank_budget: float,
) -> pd.DataFrame:
    """Enumerate every possible subset of fixed loan requests."""

    if bank_budget < 0:
        raise ValueError("bank_budget must be nonnegative")
    loans = loan_options.merge(policy_assessments, on="applicant_id", how="inner")
    applicant_ids = loans["applicant_id"].tolist()
    rows: list[dict[str, object]] = []
    for subset in _candidate_subsets(applicant_ids):
        selected = loans.loc[loans["applicant_id"].isin(subset)]
        total_principal = float(selected["requested_principal"].sum())
        total_profit = float(selected["expected_profit_true"].sum())
        feasible = total_principal <= bank_budget
        rows.append(
            {
                "selected_applicant_ids": tuple(sorted(subset)),
                "total_funded_principal": total_principal,
                "total_expected_profit_true": total_profit,
                "unused_funds": bank_budget - total_principal if feasible else float("nan"),
                "feasible": feasible,
            }
        )
    return pd.DataFrame(rows)


def allocate_fixed_requests(
    loan_options: pd.DataFrame,
    policy_assessments: pd.DataFrame,
    bank_budget: float,
) -> AllocationResult:
    """Choose the feasible subset with maximum expected monetary profit."""

    candidates = enumerate_allocations(loan_options, policy_assessments, bank_budget)
    feasible = candidates.loc[candidates["feasible"]].copy()
    best = feasible.sort_values(
        ["total_expected_profit_true", "total_funded_principal", "selected_applicant_ids"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    selected_ids = tuple(best["selected_applicant_ids"])
    decisions = loan_options.loc[:, ["applicant_id", "requested_principal"]].copy()
    decisions["funded"] = decisions["applicant_id"].isin(selected_ids)
    decisions["funded_principal"] = decisions["requested_principal"].where(
        decisions["funded"], 0.0
    )
    decisions = decisions.loc[:, LENDING_DECISIONS_COLUMNS]
    return AllocationResult(
        decisions=decisions,
        selected_applicant_ids=selected_ids,
        total_funded_principal=float(best["total_funded_principal"]),
        total_expected_profit_true=float(best["total_expected_profit_true"]),
        unused_funds=float(best["unused_funds"]),
    )
